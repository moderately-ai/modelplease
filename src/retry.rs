// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared retry logic with exponential backoff for language model calls.

use std::future::Future;

use crate::error::LanguageModelError;

/// Configuration for retry behavior.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of attempts (including the first).
    pub max_attempts: u32,
    /// Initial backoff duration in milliseconds.
    pub initial_backoff_ms: u64,
    /// Maximum backoff duration in milliseconds.
    pub max_backoff_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff_ms: 500,
            max_backoff_ms: 30_000,
        }
    }
}

/// Returns true if the error is retryable (rate limited or provider error).
const fn is_retryable(err: &LanguageModelError) -> bool {
    matches!(
        err,
        LanguageModelError::RateLimited { .. } | LanguageModelError::Provider { .. }
    )
}

/// Execute an async operation with retry and exponential backoff.
///
/// Retries on [`RateLimited`](LanguageModelError::RateLimited) and
/// [`Provider`](LanguageModelError::Provider) errors. Does NOT retry
/// [`Authentication`](LanguageModelError::Authentication) or
/// [`EmptyResponse`](LanguageModelError::EmptyResponse).
///
/// # Errors
///
/// Returns the last non-retryable error (or the last retryable error if
/// `max_attempts` is exhausted). Returns `EmptyResponse` if no error was
/// captured — only possible when `max_attempts == 0`.
#[tracing::instrument(
    skip(config, f),
    fields(max_attempts = config.max_attempts, attempts_taken = tracing::field::Empty),
    err(Display),
)]
pub async fn with_retry<F, Fut, T>(config: &RetryConfig, f: F) -> Result<T, LanguageModelError>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T, LanguageModelError>>,
{
    let mut last_err = None;
    let mut backoff_ms = config.initial_backoff_ms;
    let max_backoff = std::time::Duration::from_millis(config.max_backoff_ms);

    for attempt in 0..config.max_attempts {
        match f().await {
            Ok(val) => {
                tracing::Span::current().record("attempts_taken", attempt + 1);
                return Ok(val);
            }
            Err(err) => {
                if !is_retryable(&err) || attempt + 1 >= config.max_attempts {
                    tracing::Span::current().record("attempts_taken", attempt + 1);
                    if attempt + 1 >= config.max_attempts && is_retryable(&err) {
                        tracing::error!(
                            attempt = attempt + 1,
                            error = %err,
                            "language-model retry exhausted",
                        );
                    }
                    return Err(err);
                }

                // If the provider sent `Retry-After`, honour it (capped at
                // max_backoff so a hostile/buggy header can't pin the process).
                // Reset the exp-backoff schedule afterwards: the provider's
                // hint replaces this attempt's slot in the schedule.
                let sleep_for = if let LanguageModelError::RateLimited {
                    retry_after: Some(d),
                    ..
                } = &err
                {
                    backoff_ms = config.initial_backoff_ms;
                    (*d).min(max_backoff)
                } else {
                    // Sleep with jitter (±25%). The f64 round-trip is a
                    // best-effort jitter window; saturating on loss/overflow
                    // is the right behavior — worst case is a clamp to
                    // u64::MAX, still just a sleep duration.
                    let jitter = jitter_ms(backoff_ms);
                    let actual_ms = backoff_ms.saturating_sub(jitter)
                        + (simple_hash(attempt) % (jitter * 2 + 1));
                    std::time::Duration::from_millis(actual_ms)
                };
                tracing::warn!(
                    attempt = attempt + 1,
                    backoff_ms = u64::try_from(sleep_for.as_millis()).unwrap_or(u64::MAX),
                    error = %err,
                    "language-model call failed, retrying",
                );
                last_err = Some(err);
                tokio::time::sleep(sleep_for).await;

                backoff_ms = (backoff_ms * 2).min(config.max_backoff_ms);
            }
        }
    }

    tracing::Span::current().record("attempts_taken", config.max_attempts);
    Err(last_err.unwrap_or(LanguageModelError::EmptyResponse))
}

/// Compute a 25% jitter window saturating at `u64::MAX`.
#[expect(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "jitter is a sleep duration, never a semantic value: at f64 precision \
              loss or truncation the worst outcome is a slightly-different sleep \
              length, which is already expected behaviour"
)]
fn jitter_ms(backoff_ms: u64) -> u64 {
    (backoff_ms as f64 * 0.25) as u64
}

/// Simple deterministic hash for jitter — avoids adding rand as a dependency.
const fn simple_hash(attempt: u32) -> u64 {
    let mut h = attempt as u64;
    h = h.wrapping_mul(6_364_136_223_846_793_005);
    h = h.wrapping_add(1_442_695_040_888_963_407);
    h
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    #[tokio::test]
    async fn succeeds_on_first_attempt() {
        let config = RetryConfig {
            max_attempts: 3,
            initial_backoff_ms: 1,
            max_backoff_ms: 10,
        };
        let result = with_retry(&config, || async { Ok::<_, LanguageModelError>(42) }).await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn retries_on_rate_limited() {
        let attempts = AtomicU32::new(0);
        let config = RetryConfig {
            max_attempts: 3,
            initial_backoff_ms: 1,
            max_backoff_ms: 10,
        };

        let result = with_retry(&config, || {
            let n = attempts.fetch_add(1, Ordering::SeqCst);
            async move {
                if n < 2 {
                    Err(LanguageModelError::rate_limited("slow down"))
                } else {
                    Ok(42)
                }
            }
        })
        .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn retries_on_provider_error() {
        let attempts = AtomicU32::new(0);
        let config = RetryConfig {
            max_attempts: 2,
            initial_backoff_ms: 1,
            max_backoff_ms: 10,
        };

        let result = with_retry(&config, || {
            let n = attempts.fetch_add(1, Ordering::SeqCst);
            async move {
                if n < 1 {
                    Err(LanguageModelError::provider("server error"))
                } else {
                    Ok(42)
                }
            }
        })
        .await;

        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn does_not_retry_authentication() {
        let attempts = AtomicU32::new(0);
        let config = RetryConfig {
            max_attempts: 3,
            initial_backoff_ms: 1,
            max_backoff_ms: 10,
        };

        let result = with_retry(&config, || {
            attempts.fetch_add(1, Ordering::SeqCst);
            async { Err::<i32, _>(LanguageModelError::authentication("bad key")) }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 1); // no retry
    }

    #[tokio::test]
    async fn does_not_retry_empty_response() {
        let attempts = AtomicU32::new(0);
        let config = RetryConfig {
            max_attempts: 3,
            initial_backoff_ms: 1,
            max_backoff_ms: 10,
        };

        let result = with_retry(&config, || {
            attempts.fetch_add(1, Ordering::SeqCst);
            async { Err::<i32, _>(LanguageModelError::EmptyResponse) }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn returns_last_error_on_exhaustion() {
        let config = RetryConfig {
            max_attempts: 2,
            initial_backoff_ms: 1,
            max_backoff_ms: 10,
        };

        let result = with_retry(&config, || async {
            Err::<i32, _>(LanguageModelError::rate_limited("always failing"))
        })
        .await;

        assert!(matches!(
            result.unwrap_err(),
            LanguageModelError::RateLimited { .. }
        ));
    }

    #[tokio::test]
    async fn respects_retry_after_when_present() {
        let attempts = AtomicU32::new(0);
        // initial_backoff_ms = 1 means the exp-backoff branch would sleep
        // sub-millisecond — the elapsed-time floor proves we took the
        // Retry-After path instead.
        let config = RetryConfig {
            max_attempts: 3,
            initial_backoff_ms: 1,
            max_backoff_ms: 5_000,
        };

        let start = std::time::Instant::now();
        let result = with_retry(&config, || {
            let n = attempts.fetch_add(1, Ordering::SeqCst);
            async move {
                if n < 1 {
                    Err(LanguageModelError::rate_limited_after(
                        "slow down",
                        std::time::Duration::from_millis(50),
                    ))
                } else {
                    Ok(42)
                }
            }
        })
        .await;

        assert_eq!(result.unwrap(), 42);
        assert!(
            start.elapsed() >= std::time::Duration::from_millis(50),
            "expected to honour 50ms Retry-After, slept {:?}",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn caps_retry_after_at_max_backoff() {
        let attempts = AtomicU32::new(0);
        // Hostile Retry-After of 1 hour gets capped at max_backoff_ms = 30ms.
        let config = RetryConfig {
            max_attempts: 2,
            initial_backoff_ms: 1,
            max_backoff_ms: 30,
        };

        let start = std::time::Instant::now();
        let _ = with_retry(&config, || {
            attempts.fetch_add(1, Ordering::SeqCst);
            async {
                Err::<i32, _>(LanguageModelError::rate_limited_after(
                    "absurd hint",
                    std::time::Duration::from_secs(3600),
                ))
            }
        })
        .await;

        // Cap is 30ms, plus some scheduling overhead. Anything over a
        // second means the cap is broken.
        assert!(
            start.elapsed() < std::time::Duration::from_secs(1),
            "max_backoff_ms cap should bound the sleep, slept {:?}",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn falls_back_to_exp_backoff_when_retry_after_none() {
        // Retry-After absent — verify the existing exponential-backoff path
        // still runs (no panic, retries fire, eventual success).
        let attempts = AtomicU32::new(0);
        let config = RetryConfig {
            max_attempts: 3,
            initial_backoff_ms: 1,
            max_backoff_ms: 10,
        };

        let result = with_retry(&config, || {
            let n = attempts.fetch_add(1, Ordering::SeqCst);
            async move {
                if n < 2 {
                    Err(LanguageModelError::rate_limited("no header"))
                } else {
                    Ok(7)
                }
            }
        })
        .await;

        assert_eq!(result.unwrap(), 7);
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }
}
