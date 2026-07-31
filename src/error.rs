// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Language model error types.

use crate::capabilities::CapabilityError;

/// Errors that can occur when calling a language model.
///
/// `Clone` is derived because `moka::future::Cache::try_get_with`
/// returns `Result<_, Arc<Error>>` on cache loader failure — provider
/// `list_models` impls deref-clone the inner error to surface it
/// through the trait's plain `Result` shape.
#[derive(Debug, Clone, thiserror::Error)]
pub enum LanguageModelError {
    /// The request was rate limited by the provider.
    #[error("rate limited: {message}")]
    RateLimited {
        /// Details about the rate limit.
        message: String,
        /// Optional hint from the provider's `Retry-After` header. We parse
        /// integer delta-seconds only — HTTP-date format is treated as
        /// missing and falls back to the caller's exponential backoff.
        retry_after: Option<std::time::Duration>,
    },

    /// Authentication with the provider failed.
    #[error("authentication error: {message}")]
    Authentication {
        /// Details about the auth failure.
        message: String,
    },

    /// The language model returned an empty response.
    #[error("empty response from language model")]
    EmptyResponse,

    /// A provider-specific error occurred.
    #[error("provider error: {message}")]
    Provider {
        /// Details about the provider error.
        message: String,
    },

    /// A capability mismatch was detected before the request reached
    /// the wire (wrong modality, wrong source kind, wrong format, or
    /// oversize payload for the model). Surfaced by
    /// [`LanguageModelProvider::validate_request`](crate::LanguageModelProvider::validate_request).
    #[error(transparent)]
    Capability(#[from] CapabilityError),
}

impl LanguageModelError {
    /// Create a [`RateLimited`](LanguageModelError::RateLimited) error
    /// without a `Retry-After` hint.
    pub fn rate_limited(message: impl Into<String>) -> Self {
        Self::RateLimited {
            message: message.into(),
            retry_after: None,
        }
    }

    /// Create a [`RateLimited`](LanguageModelError::RateLimited) error
    /// carrying a `Retry-After` hint parsed from the provider response.
    pub fn rate_limited_after(
        message: impl Into<String>,
        retry_after: std::time::Duration,
    ) -> Self {
        Self::RateLimited {
            message: message.into(),
            retry_after: Some(retry_after),
        }
    }

    /// Create an [`Authentication`](LanguageModelError::Authentication) error.
    pub fn authentication(message: impl Into<String>) -> Self {
        Self::Authentication {
            message: message.into(),
        }
    }

    /// Create a [`Provider`](LanguageModelError::Provider) error.
    pub fn provider(message: impl Into<String>) -> Self {
        Self::Provider {
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let err = LanguageModelError::rate_limited("try again in 30s");
        assert!(err.to_string().contains("rate limited"));
        assert!(err.to_string().contains("30s"));
    }

    #[test]
    fn empty_response_display() {
        let err = LanguageModelError::EmptyResponse;
        assert!(err.to_string().contains("empty response"));
    }
}
