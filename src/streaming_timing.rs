// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-stream observability instrumentation for streaming providers.
//! `generate_stream` returns a `Pin<Box<dyn Stream<...>>>` and the
//! `#[instrument]` macro returns once the stream is constructed — so
//! the close event covers the request-build phase only and every
//! subsequent runtime fact (the first-delta moment, the upstream-
//! reported usage on the final delta) is invisible to external
//! observability unless something on the stream records it.
//!
//! [`instrument_stream`] wraps a `Stream<Item = Result<StreamDelta,
//! LanguageModelError>>` and records two distinct facts onto a span
//! the caller supplies:
//!
//! - **`first_token_ms`** on the first `Ok` item (elapsed since the `started_at` instant the caller
//!   captured at function entry). Matches client-side TTFT semantics.
//! - **`prompt_tokens` / `completion_tokens` / `total_tokens` / `cache_creation_input_tokens` /
//!   `cache_read_input_tokens`** on wrapper drop, derived from the most recent delta that carried a
//!   `usage` payload. Providers reliably emit usage on the final delta (or close to it), so
//!   end-of-stream is the right recording moment.
//!
//! Errors are ignored for both metrics: the first successful delta
//! drives `first_token_ms`; usage is recorded from whichever delta most
//! recently carried it. When upstream never surfaces usage on this
//! stream the cache fields stay absent (consistent with the buffered
//! `generate` span when `prediction.usage()` returns `None`).
//!
//! The wrapper holds the `tracing::Span` handle (a cheap Arc-backed
//! reference) so `span.record()` keeps working even after
//! `generate_stream` has returned and the macro's drop-guard exited
//! the span — the span stays open until the wrapped stream itself is
//! dropped.

use std::{
    pin::Pin,
    task::{Context, Poll},
    time::Instant,
};

use futures::Stream;
use pin_project::pin_project;

use crate::{
    LanguageModelError,
    response::{StreamDelta, Usage},
};

/// Wrap `inner` so the wrapped stream records `first_token_ms` on the
/// first `Ok` delta and records the upstream `Usage` (prompt /
/// completion / cache fields) on drop. Subsequent items pass through
/// unchanged.
pub(crate) const fn instrument_stream<S>(
    span: tracing::Span,
    started_at: Instant,
    inner: S,
) -> InstrumentedStream<S>
where
    S: Stream<Item = Result<StreamDelta, LanguageModelError>>,
{
    InstrumentedStream {
        inner,
        span,
        started_at,
        first_seen: false,
        last_usage: None,
    }
}

#[pin_project(PinnedDrop)]
pub(crate) struct InstrumentedStream<S> {
    #[pin]
    inner: S,
    span: tracing::Span,
    started_at: Instant,
    first_seen: bool,
    last_usage: Option<Usage>,
}

impl<S> Stream for InstrumentedStream<S>
where
    S: Stream<Item = Result<StreamDelta, LanguageModelError>>,
{
    type Item = Result<StreamDelta, LanguageModelError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();
        let poll = this.inner.poll_next(cx);
        if let Poll::Ready(Some(Ok(delta))) = &poll {
            if !*this.first_seen {
                let elapsed_ms =
                    u64::try_from(this.started_at.elapsed().as_millis()).unwrap_or(u64::MAX);
                this.span.record("first_token_ms", elapsed_ms);
                *this.first_seen = true;
            }
            if let Some(usage) = delta.usage {
                *this.last_usage = Some(usage);
            }
        }
        poll
    }
}

#[pin_project::pinned_drop]
impl<S> PinnedDrop for InstrumentedStream<S> {
    fn drop(self: Pin<&mut Self>) {
        let this = self.project();
        if let Some(usage) = this.last_usage {
            let cache_creation = usage.cache_creation_input_tokens;
            let cache_read = usage.cache_read_input_tokens;
            let prompt_total = usage.input_tokens + cache_read + cache_creation;
            this.span.record("prompt_tokens", prompt_total);
            this.span.record("completion_tokens", usage.output_tokens);
            this.span
                .record("total_tokens", prompt_total + usage.output_tokens);
            this.span
                .record("cache_creation_input_tokens", cache_creation);
            this.span.record("cache_read_input_tokens", cache_read);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc, time::Duration};

    use futures::{StreamExt, stream};
    use parking_lot::Mutex;
    use tracing_subscriber::{
        Layer,
        layer::{Context as LayerContext, SubscriberExt as _},
        registry::LookupSpan,
    };

    use super::*;

    /// Layer that captures every numeric value recorded via
    /// `span.record(...)`, keyed by field name. Lets tests assert on
    /// what was recorded without coupling to the global tracing
    /// formatter.
    struct FieldCapture {
        captured: Arc<Mutex<BTreeMap<String, Vec<u64>>>>,
    }

    impl<S> Layer<S> for FieldCapture
    where
        S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    {
        fn on_record(
            &self,
            _id: &tracing::span::Id,
            values: &tracing::span::Record<'_>,
            _ctx: LayerContext<'_, S>,
        ) {
            struct V {
                captured: Arc<Mutex<BTreeMap<String, Vec<u64>>>>,
            }
            impl tracing::field::Visit for V {
                fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
                    self.captured
                        .lock()
                        .entry(field.name().to_owned())
                        .or_default()
                        .push(value);
                }
                fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
                    if let Ok(v) = u64::try_from(value) {
                        self.captured
                            .lock()
                            .entry(field.name().to_owned())
                            .or_default()
                            .push(v);
                    }
                }
                fn record_debug(
                    &mut self,
                    _field: &tracing::field::Field,
                    _value: &dyn std::fmt::Debug,
                ) {
                }
            }
            values.record(&mut V {
                captured: Arc::clone(&self.captured),
            });
        }
    }

    fn delta(content: &str, usage: Option<Usage>) -> StreamDelta {
        StreamDelta {
            content: content.to_owned(),
            thinking: None,
            usage,
            model: None,
            stop_reason: None,
            is_final: usage.is_some(),
        }
    }

    fn make_span() -> tracing::Span {
        tracing::info_span!(
            "test_stream",
            first_token_ms = tracing::field::Empty,
            prompt_tokens = tracing::field::Empty,
            completion_tokens = tracing::field::Empty,
            total_tokens = tracing::field::Empty,
            cache_creation_input_tokens = tracing::field::Empty,
            cache_read_input_tokens = tracing::field::Empty,
        )
    }

    #[tokio::test]
    async fn first_token_ms_recorded_on_first_ok_item() {
        let captured: Arc<Mutex<BTreeMap<String, Vec<u64>>>> =
            Arc::new(Mutex::new(BTreeMap::new()));
        let subscriber = tracing_subscriber::registry().with(FieldCapture {
            captured: Arc::clone(&captured),
        });
        let _guard = tracing::subscriber::set_default(subscriber);

        let span = make_span();
        let inner = stream::iter(vec![Ok(delta("hello", None)), Ok(delta("world", None))]);
        let started_at = Instant::now();
        let mut wrapped = instrument_stream(span, started_at, inner);

        // Give the elapsed clock a couple of ms so a too-eager
        // implementation (recording from a captured zero-elapsed) would
        // not happen to pass.
        tokio::time::sleep(Duration::from_millis(2)).await;

        assert!(wrapped.next().await.is_some());
        assert!(wrapped.next().await.is_some());

        let values = captured
            .lock()
            .get("first_token_ms")
            .cloned()
            .unwrap_or_default();
        assert_eq!(
            values.len(),
            1,
            "first_token_ms should have been recorded exactly once"
        );
        assert!(
            values[0] >= 1,
            "first_token_ms should reflect the elapsed sleep, got {}",
            values[0]
        );
    }

    #[tokio::test]
    async fn first_token_ms_only_recorded_once() {
        let captured: Arc<Mutex<BTreeMap<String, Vec<u64>>>> =
            Arc::new(Mutex::new(BTreeMap::new()));
        let subscriber = tracing_subscriber::registry().with(FieldCapture {
            captured: Arc::clone(&captured),
        });
        let _guard = tracing::subscriber::set_default(subscriber);

        let span = make_span();
        let inner = stream::iter(vec![
            Ok(delta("a", None)),
            Ok(delta("b", None)),
            Ok(delta("c", None)),
        ]);
        let started_at = Instant::now();
        let mut wrapped = instrument_stream(span, started_at, inner);
        while wrapped.next().await.is_some() {}
        drop(wrapped);

        assert_eq!(
            captured.lock().get("first_token_ms").map_or(0, Vec::len),
            1,
            "first_token_ms should be recorded exactly once across multiple emissions",
        );
    }

    #[tokio::test]
    async fn errors_do_not_record_first_token_ms() {
        let captured: Arc<Mutex<BTreeMap<String, Vec<u64>>>> =
            Arc::new(Mutex::new(BTreeMap::new()));
        let subscriber = tracing_subscriber::registry().with(FieldCapture {
            captured: Arc::clone(&captured),
        });
        let _guard = tracing::subscriber::set_default(subscriber);

        let span = make_span();
        let inner = stream::iter(vec![
            Err::<StreamDelta, LanguageModelError>(LanguageModelError::provider("boom")),
            Ok(delta("recovered", None)),
        ]);
        let started_at = Instant::now();
        let mut wrapped = instrument_stream(span, started_at, inner);

        // First item is Err — must NOT record.
        assert!(matches!(wrapped.next().await, Some(Err(_))));
        assert!(
            captured
                .lock()
                .get("first_token_ms")
                .is_none_or(Vec::is_empty),
            "Err items must not record first_token_ms",
        );

        // First Ok now records.
        assert!(wrapped.next().await.is_some());
        assert_eq!(
            captured.lock().get("first_token_ms").map_or(0, Vec::len),
            1,
            "first Ok after Err records first_token_ms",
        );
    }

    #[tokio::test]
    async fn usage_fields_recorded_on_drop_from_final_delta() {
        let captured: Arc<Mutex<BTreeMap<String, Vec<u64>>>> =
            Arc::new(Mutex::new(BTreeMap::new()));
        let subscriber = tracing_subscriber::registry().with(FieldCapture {
            captured: Arc::clone(&captured),
        });
        let _guard = tracing::subscriber::set_default(subscriber);

        let span = make_span();
        let inner = stream::iter(vec![
            Ok(delta("a", None)),
            Ok(delta("b", None)),
            Ok(delta(
                "",
                Some(Usage {
                    input_tokens: 17,
                    output_tokens: 7,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 1824,
                }),
            )),
        ]);
        let started_at = Instant::now();
        let mut wrapped = instrument_stream(span, started_at, inner);
        while wrapped.next().await.is_some() {}
        drop(wrapped);

        let by_field = captured.lock().clone();
        // prompt_total = input + cache_read + cache_creation = 17 + 1824 + 0
        assert_eq!(by_field.get("prompt_tokens"), Some(&vec![1841_u64]));
        assert_eq!(by_field.get("completion_tokens"), Some(&vec![7_u64]));
        assert_eq!(by_field.get("total_tokens"), Some(&vec![1848_u64]));
        assert_eq!(
            by_field.get("cache_creation_input_tokens"),
            Some(&vec![0_u64])
        );
        assert_eq!(
            by_field.get("cache_read_input_tokens"),
            Some(&vec![1824_u64])
        );
    }

    #[tokio::test]
    async fn no_usage_fields_recorded_when_upstream_omits_usage() {
        let captured: Arc<Mutex<BTreeMap<String, Vec<u64>>>> =
            Arc::new(Mutex::new(BTreeMap::new()));
        let subscriber = tracing_subscriber::registry().with(FieldCapture {
            captured: Arc::clone(&captured),
        });
        let _guard = tracing::subscriber::set_default(subscriber);

        let span = make_span();
        let inner = stream::iter(vec![Ok(delta("a", None)), Ok(delta("b", None))]);
        let started_at = Instant::now();
        let mut wrapped = instrument_stream(span, started_at, inner);
        while wrapped.next().await.is_some() {}
        drop(wrapped);

        let by_field = captured.lock().clone();
        assert!(
            !by_field.contains_key("prompt_tokens"),
            "no usage chunk on the stream means no token field recorded; got {by_field:?}",
        );
        assert!(!by_field.contains_key("cache_read_input_tokens"));
    }
}
