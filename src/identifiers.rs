// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Newtype wrappers for provider credentials and model identifiers.
//!
//! Both `api_key` and `model` end up as `String` on the wire and are
//! adjacent on every provider config (`AnthropicConfig`, `OpenAiConfig`,
//! `OllamaConfig`). Naming them at struct-literal call sites is the
//! convention but doesn't survive helpers that build configs from
//! sequenced sources (env vars, CLI args). The newtypes elevate the
//! protection from convention to compiler-enforced.
//!
//! `ApiKey` follows the *owned validated* shape from
//! `.claude/rules/rust.md`: the only constructor is
//! [`ApiKey::parse`], which enforces non-empty + no whitespace
//! padding + no embedded control bytes. Once a caller holds an
//! `ApiKey` it carries that proof through every downstream call —
//! provider clients can't be handed a partner-supplied blank string
//! that prints redacted but produces a 401 on first use.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize};

/// Failure mode returned by [`ApiKey::parse`].
///
/// Validation fires before storage / network use so misconfiguration
/// surfaces at config-load (or test fixture) time rather than at the
/// first authenticated request.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ApiKeyError {
    /// Empty string — caller almost certainly meant `None`.
    #[error("api key cannot be empty")]
    Empty,
    /// Leading or trailing whitespace — usually a copy-paste artifact
    /// from a TOML/env value that got quoted with surrounding spaces;
    /// providers reject it as a 401 with no actionable error.
    #[error("api key cannot have leading or trailing whitespace")]
    Whitespace,
    /// Embedded control byte (NUL, CR, LF, etc). HTTP header
    /// serialization would fail loudly; the parse-time reject keeps
    /// the failure local to the caller that fed the bad input.
    #[error("api key cannot contain control characters")]
    ControlChar,
}

/// Provider API key. Validated at parse time; redacted on `Display`.
///
/// Construct via [`ApiKey::parse`] — the inner string is private so
/// every value carries the parse-time proof.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct ApiKey(String);

impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ApiKey(<redacted>)")
    }
}

impl ApiKey {
    /// Validate `value` and wrap it as an `ApiKey`.
    ///
    /// # Errors
    /// Returns [`ApiKeyError`] when the input is empty, padded with
    /// whitespace, or contains an embedded control byte.
    pub fn parse(value: impl Into<String>) -> Result<Self, ApiKeyError> {
        let value = value.into();
        if value.is_empty() {
            return Err(ApiKeyError::Empty);
        }
        if value.trim() != value {
            return Err(ApiKeyError::Whitespace);
        }
        if value.chars().any(char::is_control) {
            return Err(ApiKeyError::ControlChar);
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[must_use]
    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Never echo the key — partner credentials should not leak via
        // accidental Display in logs. Show only a stable redaction.
        f.write_str("<api-key:redacted>")
    }
}

impl<'de> Deserialize<'de> for ApiKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::parse(s).map_err(serde::de::Error::custom)
    }
}

impl From<ApiKey> for String {
    fn from(value: ApiKey) -> Self {
        value.0
    }
}

impl TryFrom<String> for ApiKey {
    type Error = ApiKeyError;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(value)
    }
}

/// Provider model identifier (e.g. `"claude-sonnet-4-6"`,
/// `"gpt-4o-mini"`). Wraps an arbitrary `String` so it cannot be passed
/// where an [`ApiKey`] is expected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelId(pub String);

impl ModelId {
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[must_use]
    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_typical_key() {
        let key = ApiKey::parse("sk-ant-abc123").unwrap();
        assert_eq!(key.as_str(), "sk-ant-abc123");
    }

    #[test]
    fn parse_rejects_empty() {
        assert_eq!(ApiKey::parse(""), Err(ApiKeyError::Empty));
    }

    #[test]
    fn parse_rejects_leading_whitespace() {
        assert_eq!(ApiKey::parse(" sk-ant-abc"), Err(ApiKeyError::Whitespace));
    }

    #[test]
    fn parse_rejects_trailing_whitespace() {
        assert_eq!(ApiKey::parse("sk-ant-abc\n"), Err(ApiKeyError::Whitespace));
    }

    #[test]
    fn parse_rejects_embedded_control_byte() {
        assert_eq!(ApiKey::parse("sk-ant\0abc"), Err(ApiKeyError::ControlChar));
    }

    #[test]
    fn display_redacts() {
        let key = ApiKey::parse("super-secret").unwrap();
        assert_eq!(format!("{key}"), "<api-key:redacted>");
    }

    #[test]
    fn debug_redacts() {
        let key = ApiKey::parse("super-secret").unwrap();
        let rendered = format!("{key:?}");
        assert_eq!(rendered, "ApiKey(<redacted>)");
        assert!(!rendered.contains("super-secret"));
    }

    #[test]
    fn deserialize_revalidates() {
        // Valid string deserializes.
        let ok: ApiKey = serde_json::from_str("\"sk-ant-abc\"").unwrap();
        assert_eq!(ok.as_str(), "sk-ant-abc");

        // Invalid string fails — the parse-time proof can't be bypassed
        // by a wire decode.
        let err = serde_json::from_str::<ApiKey>("\"\"").unwrap_err();
        assert!(err.to_string().contains("empty"), "got: {err}");
    }
}
