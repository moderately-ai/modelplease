// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Media source taxonomy for multimodal content parts.
//!
//! Provides validated newtypes for every kind of media reference that can
//! cross the wire to an LM provider (`HttpsUrl`, `S3Uri`, `MediaType`,
//! `AwsAccountId`, `ProviderFileId`), the [`MediaSource`] sum type that
//! enumerates them, and the [`SourceKind`] discriminant used in
//! capability tables.
//!
//! The newtypes follow the *owned validated* pattern documented in
//! `.claude/rules/rust.md`: the only constructor is `parse(...)`, which
//! returns a `Result<Self, *Error>`. Once a caller holds the wrapper,
//! downstream code cannot be handed a malformed string — `Deserialize`
//! re-validates, so a partner-supplied JSON payload cannot smuggle an
//! invalid value past the wire.

use std::fmt;

use base64::{Engine, engine::general_purpose::STANDARD as BASE64_STANDARD};
use enumset::{EnumSet, EnumSetType};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};
use thiserror::Error;

/// Failure mode returned by [`HttpsUrl::parse`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum HttpsUrlError {
    /// Empty string.
    #[error("https url cannot be empty")]
    Empty,
    /// Missing or non-`https://` scheme. We refuse `http://`, `ftp://`,
    /// and bare hostnames — every supported provider requires TLS.
    #[error("https url must start with `https://`")]
    NonHttpsScheme,
    /// `https://` prefix present but no host component followed.
    #[error("https url is missing a host component")]
    MissingHost,
    /// Embedded control byte (NUL, CR, LF, etc.). HTTP header
    /// serialization would fail loudly; the parse-time reject keeps the
    /// failure local to the caller that fed the bad input.
    #[error("https url cannot contain control characters")]
    ControlChar,
}

/// A URL pinned to the `https://` scheme.
///
/// `parse` rejects empty input, non-https schemes, missing hosts, and
/// embedded control bytes. Every LM provider that accepts URL-sourced
/// media (`OpenAI`, Anthropic) requires TLS, so the validation eliminates
/// a class of "works on staging, breaks in prod behind a redirect"
/// surprises at the construction boundary.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct HttpsUrl(String);

impl HttpsUrl {
    /// Validate `value` and wrap it as an [`HttpsUrl`].
    ///
    /// # Errors
    /// Returns [`HttpsUrlError`] when the input is empty, has the wrong
    /// scheme, is missing a host, or contains a control character.
    pub fn parse(value: impl Into<String>) -> Result<Self, HttpsUrlError> {
        let value = value.into();
        if value.is_empty() {
            return Err(HttpsUrlError::Empty);
        }
        if value.chars().any(char::is_control) {
            return Err(HttpsUrlError::ControlChar);
        }
        let Some(rest) = value.strip_prefix("https://") else {
            return Err(HttpsUrlError::NonHttpsScheme);
        };
        // Host runs to the first `/`, `?`, `#`, or end-of-string; reject
        // the variants where the prefix exists but nothing follows it.
        let host_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
        if host_end == 0 {
            return Err(HttpsUrlError::MissingHost);
        }
        Ok(Self(value))
    }

    /// Borrow the validated string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for HttpsUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for HttpsUrl {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Self::parse(s).map_err(D::Error::custom)
    }
}

impl TryFrom<String> for HttpsUrl {
    type Error = HttpsUrlError;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(value)
    }
}

impl From<HttpsUrl> for String {
    fn from(value: HttpsUrl) -> Self {
        value.0
    }
}

/// Failure mode returned by [`S3Uri::parse`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum S3UriError {
    #[error("s3 uri cannot be empty")]
    Empty,
    #[error("s3 uri must start with `s3://`")]
    BadScheme,
    #[error("s3 uri is missing a bucket component")]
    MissingBucket,
    #[error("s3 uri is missing a key component")]
    MissingKey,
    #[error("s3 uri cannot contain control characters")]
    ControlChar,
}

/// A canonical AWS S3 URI: `s3://<bucket>/<key>`.
///
/// `parse` requires a non-empty bucket *and* a non-empty key. Bedrock's
/// `S3Location` field will reject either-missing-or-empty halves at the
/// SDK boundary; we surface that at construction so the caller learns
/// before request time.
///
/// IAM note: the Bedrock service role still needs `s3:GetObject` on the
/// referenced object. That cannot be verified locally; surface a clear
/// `AccessDenied` mapping in the provider error layer instead.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct S3Uri(String);

impl S3Uri {
    /// Validate `value` and wrap it as an [`S3Uri`].
    ///
    /// # Errors
    /// Returns [`S3UriError`] when the scheme is wrong, the bucket or
    /// key is missing, or a control character is embedded.
    pub fn parse(value: impl Into<String>) -> Result<Self, S3UriError> {
        let value = value.into();
        if value.is_empty() {
            return Err(S3UriError::Empty);
        }
        if value.chars().any(char::is_control) {
            return Err(S3UriError::ControlChar);
        }
        let Some(rest) = value.strip_prefix("s3://") else {
            return Err(S3UriError::BadScheme);
        };
        let Some((bucket, key)) = rest.split_once('/') else {
            return Err(S3UriError::MissingKey);
        };
        if bucket.is_empty() {
            return Err(S3UriError::MissingBucket);
        }
        if key.is_empty() {
            return Err(S3UriError::MissingKey);
        }
        Ok(Self(value))
    }

    /// Borrow the validated string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Return the bucket component (between `s3://` and the first `/`).
    #[must_use]
    pub fn bucket(&self) -> &str {
        // Invariant established by `parse`: `s3://<bucket>/<key>` with
        // non-empty bucket and key; the split is therefore total.
        self.0
            .strip_prefix("s3://")
            .and_then(|rest| rest.split_once('/'))
            .map_or("", |(bucket, _)| bucket)
    }

    /// Return the key component (everything after the first `/` past
    /// `s3://<bucket>`).
    #[must_use]
    pub fn key(&self) -> &str {
        self.0
            .strip_prefix("s3://")
            .and_then(|rest| rest.split_once('/'))
            .map_or("", |(_, key)| key)
    }
}

impl fmt::Display for S3Uri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for S3Uri {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Self::parse(s).map_err(D::Error::custom)
    }
}

impl TryFrom<String> for S3Uri {
    type Error = S3UriError;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(value)
    }
}

impl From<S3Uri> for String {
    fn from(value: S3Uri) -> Self {
        value.0
    }
}

/// Failure mode returned by [`MediaType::parse`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum MediaTypeError {
    #[error("media type cannot be empty")]
    Empty,
    /// No `/` between top-level type and subtype.
    #[error("media type must be `type/subtype` (RFC 6838)")]
    MissingSlash,
    #[error("media type top-level cannot be empty")]
    MissingTopLevel,
    #[error("media type subtype cannot be empty")]
    MissingSubtype,
    #[error("media type contains an invalid character")]
    InvalidCharacter,
}

/// An RFC 6838 media type / MIME (`type/subtype[;parameters]`).
///
/// `parse` validates that the input has the `type/subtype` shape, both
/// halves are non-empty, and the characters are restricted to the
/// RFC-restricted set (`A-Za-z0-9` plus `!#$&^_-+.`). Parameters
/// (`; charset=...`) are preserved verbatim but not further validated.
///
/// Use [`MediaType::top_level`] to bucket into the [`crate::MediaKind`] taxonomy
/// (`image/*` → `Image`, `audio/*` → `Audio`, `video/*` → `Video`,
/// `application/*` / `text/*` → `Document`) and [`MediaType::subtype`]
/// when checking against a provider's accepted-format list (e.g.
/// `["png", "jpeg", "gif", "webp"]`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct MediaType(String);

impl MediaType {
    /// Validate `value` and wrap it as a [`MediaType`].
    ///
    /// # Errors
    /// Returns [`MediaTypeError`] when the input is empty, missing the
    /// `/`, has an empty top-level or subtype, or contains a character
    /// outside the RFC-restricted set.
    pub fn parse(value: impl Into<String>) -> Result<Self, MediaTypeError> {
        let value = value.into();
        if value.is_empty() {
            return Err(MediaTypeError::Empty);
        }
        let Some((top, rest)) = value.split_once('/') else {
            return Err(MediaTypeError::MissingSlash);
        };
        if top.is_empty() {
            return Err(MediaTypeError::MissingTopLevel);
        }
        // Subtype ends at the first `;` (parameter boundary), or
        // at the end of the string.
        let subtype_end = rest.find(';').unwrap_or(rest.len());
        let subtype = &rest[..subtype_end];
        if subtype.is_empty() {
            return Err(MediaTypeError::MissingSubtype);
        }
        if !top.bytes().all(is_mime_token_byte) || !subtype.bytes().all(is_mime_token_byte) {
            return Err(MediaTypeError::InvalidCharacter);
        }
        Ok(Self(value))
    }

    /// Borrow the validated string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Return the top-level type (e.g. `"image"` for `"image/png"`).
    #[must_use]
    pub fn top_level(&self) -> &str {
        // Invariant from parse: contains `/` with non-empty top-level.
        self.0.split_once('/').map_or("", |(top, _)| top)
    }

    /// Return the subtype, stripped of any parameters
    /// (e.g. `"png"` for `"image/png; charset=utf-8"`).
    #[must_use]
    pub fn subtype(&self) -> &str {
        let after_slash = self.0.split_once('/').map_or("", |(_, rest)| rest);
        let subtype_end = after_slash.find(';').unwrap_or(after_slash.len());
        &after_slash[..subtype_end]
    }
}

impl fmt::Display for MediaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for MediaType {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Self::parse(s).map_err(D::Error::custom)
    }
}

impl TryFrom<String> for MediaType {
    type Error = MediaTypeError;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(value)
    }
}

impl From<MediaType> for String {
    fn from(value: MediaType) -> Self {
        value.0
    }
}

const fn is_mime_token_byte(byte: u8) -> bool {
    // RFC 6838 §4.2: restricted-name characters.
    byte.is_ascii_uppercase()
        || byte.is_ascii_lowercase()
        || byte.is_ascii_digit()
        || matches!(
            byte,
            b'!' | b'#' | b'$' | b'&' | b'^' | b'_' | b'-' | b'+' | b'.'
        )
}

/// Failure mode returned by [`AwsAccountId::parse`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum AwsAccountIdError {
    #[error("aws account id must be 12 digits")]
    NotTwelveDigits,
}

/// A 12-digit AWS account id. Used as the optional `bucket_owner` on
/// [`MediaSource::S3`] for cross-account `S3Location` references.
///
/// Cross-account S3 access in Bedrock requires the caller to pass the
/// bucket owner's account id alongside the URI — without it the SDK
/// short-circuits with `Confused deputy`-style errors. Validating the
/// 12-digit shape at construction surfaces the typo (or accidentally
/// passing the role arn) up-front.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct AwsAccountId(String);

impl AwsAccountId {
    /// Validate `value` and wrap it as an [`AwsAccountId`].
    ///
    /// # Errors
    /// Returns [`AwsAccountIdError::NotTwelveDigits`] when the input is
    /// not exactly twelve ASCII digits.
    pub fn parse(value: impl Into<String>) -> Result<Self, AwsAccountIdError> {
        let value = value.into();
        if value.len() == 12 && value.bytes().all(|b| b.is_ascii_digit()) {
            Ok(Self(value))
        } else {
            Err(AwsAccountIdError::NotTwelveDigits)
        }
    }

    /// Borrow the validated string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AwsAccountId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for AwsAccountId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Self::parse(s).map_err(D::Error::custom)
    }
}

impl TryFrom<String> for AwsAccountId {
    type Error = AwsAccountIdError;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(value)
    }
}

impl From<AwsAccountId> for String {
    fn from(value: AwsAccountId) -> Self {
        value.0
    }
}

/// Failure mode returned by [`ProviderFileId::parse`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ProviderFileIdError {
    #[error("provider file id cannot be empty")]
    Empty,
    #[error("provider file id cannot contain control characters")]
    ControlChar,
}

/// An opaque, provider-issued file identifier (Anthropic Files API,
/// `OpenAI` Files API).
///
/// Validated for non-emptiness + no control chars only. Routing to the
/// right provider is implicit in the [`crate::ModelId`] used at call
/// time — a Claude file_id paired with an `OpenAI` model will surface
/// as a provider-side error, not a local one.
///
/// Lifecycle is *not* tracked here. Anthropic file_ids persist until
/// explicit DELETE (no auto-expiry); `OpenAI` file_ids carry a purpose
/// tag (`assistants`, `user_data`) that constrains usage. Both
/// lifecycles live outside this crate (application storage layer).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct ProviderFileId(String);

impl ProviderFileId {
    /// Validate `value` and wrap it as a [`ProviderFileId`].
    ///
    /// # Errors
    /// Returns [`ProviderFileIdError`] when the input is empty or
    /// contains a control character.
    pub fn parse(value: impl Into<String>) -> Result<Self, ProviderFileIdError> {
        let value = value.into();
        if value.is_empty() {
            return Err(ProviderFileIdError::Empty);
        }
        if value.chars().any(char::is_control) {
            return Err(ProviderFileIdError::ControlChar);
        }
        Ok(Self(value))
    }

    /// Borrow the validated string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ProviderFileId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ProviderFileId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Self::parse(s).map_err(D::Error::custom)
    }
}

impl TryFrom<String> for ProviderFileId {
    type Error = ProviderFileIdError;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(value)
    }
}

impl From<ProviderFileId> for String {
    fn from(value: ProviderFileId) -> Self {
        value.0
    }
}

/// Discriminant for [`MediaSource`] variants, suitable for use in
/// [`enumset::EnumSet`]-backed capability masks (e.g. `MediaSupport.sources`).
///
/// The `EnumSet` representation iterates in variant declaration order,
/// matching the workspace's `BTreeMap`-over-`HashMap` determinism
/// preference for any iteration that could leak into output.
#[derive(EnumSetType, Debug, Hash, Serialize, Deserialize)]
#[enumset(serialize_repr = "list")]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// A reachable HTTPS URL the provider fetches itself.
    Url,
    /// Bytes carried inline on the request (base64 over JSON).
    InlineBytes,
    /// A provider-issued file id from a prior upload to the provider's
    /// Files API.
    ProviderFile,
    /// An S3 URI, with an optional explicit bucket-owner for
    /// cross-account access.
    S3,
}

/// Where the bytes of one media content part come from.
///
/// The four variants exhaust the source kinds supported by any backend
/// we care about: HTTPS URL (`OpenAI`, Anthropic), inline base64 (all),
/// provider-issued file id (`OpenAI`, Anthropic), and S3 URI (Bedrock).
/// Provider impls translate the source variant they natively speak and
/// return `CapabilityError::SourceKindUnsupported` for the rest — the
/// validation lives on the provider's capability table, not here.
///
/// The serde representation is internally tagged on `source`:
///
/// ```json
/// {"source": "url", "url": "https://example.com/cat.png"}
/// {"source": "inline_bytes", "mime": "image/png", "data": "<base64>"}
/// {"source": "provider_file", "file_id": "file-..."}
/// {"source": "s3", "uri": "s3://bucket/key", "bucket_owner": "123456789012"}
/// ```
///
/// This in-crate JSON shape is intentionally provider-neutral. Downstream
/// applications can map their own accepted input envelopes into these variants
/// before sending a request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum MediaSource {
    /// A reachable HTTPS URL.
    Url {
        /// The URL the provider will fetch.
        url: HttpsUrl,
    },
    /// Bytes carried inline. Serialized as base64 over JSON.
    InlineBytes {
        /// The media type of `data`. Subtype is matched against the
        /// provider's accepted-format list at validation time.
        mime: MediaType,
        /// Raw bytes. Encoded/decoded as base64 on the JSON wire.
        #[serde(with = "base64_bytes")]
        data: Vec<u8>,
    },
    /// A reference to bytes already uploaded to the provider's Files API.
    ProviderFile {
        /// The provider-issued file identifier.
        file_id: ProviderFileId,
    },
    /// An S3 URI, with an optional explicit cross-account bucket owner.
    S3 {
        /// The S3 URI (`s3://bucket/key`).
        uri: S3Uri,
        /// 12-digit account id of the bucket owner. Required for
        /// cross-account access; omit when the caller's account owns
        /// the bucket.
        #[serde(skip_serializing_if = "Option::is_none", default)]
        bucket_owner: Option<AwsAccountId>,
    },
}

impl MediaSource {
    /// Return the discriminant of this source variant.
    #[must_use]
    pub const fn kind(&self) -> SourceKind {
        match self {
            Self::Url { .. } => SourceKind::Url,
            Self::InlineBytes { .. } => SourceKind::InlineBytes,
            Self::ProviderFile { .. } => SourceKind::ProviderFile,
            Self::S3 { .. } => SourceKind::S3,
        }
    }
}

/// Serde helper: base64-encode `Vec<u8>` as a standard-alphabet base64
/// string on the JSON wire.
mod base64_bytes {
    use super::{BASE64_STANDARD, Deserialize, Deserializer, Engine, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&BASE64_STANDARD.encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(d)?;
        BASE64_STANDARD
            .decode(s.as_bytes())
            .map_err(serde::de::Error::custom)
    }
}

/// `EnumSet<SourceKind>` containing every variant — useful as a default
/// `accepted_sources` for callers that haven't narrowed the slot.
#[must_use]
pub const fn all_source_kinds() -> EnumSet<SourceKind> {
    EnumSet::all()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- HttpsUrl ----

    #[test]
    fn https_url_accepts_typical() {
        let u = HttpsUrl::parse("https://example.com/img.png").unwrap();
        assert_eq!(u.as_str(), "https://example.com/img.png");
    }

    #[test]
    fn https_url_rejects_http_scheme() {
        assert_eq!(
            HttpsUrl::parse("http://example.com/x"),
            Err(HttpsUrlError::NonHttpsScheme)
        );
    }

    #[test]
    fn https_url_rejects_missing_host() {
        assert_eq!(
            HttpsUrl::parse("https:///x"),
            Err(HttpsUrlError::MissingHost)
        );
    }

    #[test]
    fn https_url_rejects_empty() {
        assert_eq!(HttpsUrl::parse(""), Err(HttpsUrlError::Empty));
    }

    #[test]
    fn https_url_rejects_control_chars() {
        assert_eq!(
            HttpsUrl::parse("https://example.com/\n"),
            Err(HttpsUrlError::ControlChar)
        );
    }

    #[test]
    fn https_url_deserialize_revalidates() {
        let err = serde_json::from_str::<HttpsUrl>("\"http://x\"").unwrap_err();
        assert!(err.to_string().contains("https"));
    }

    // ---- S3Uri ----

    #[test]
    fn s3_uri_accepts_typical() {
        let u = S3Uri::parse("s3://my-bucket/path/to/object").unwrap();
        assert_eq!(u.bucket(), "my-bucket");
        assert_eq!(u.key(), "path/to/object");
    }

    #[test]
    fn s3_uri_rejects_wrong_scheme() {
        assert!(matches!(
            S3Uri::parse("https://b/k"),
            Err(S3UriError::BadScheme)
        ));
    }

    #[test]
    fn s3_uri_rejects_missing_key() {
        assert!(matches!(
            S3Uri::parse("s3://my-bucket"),
            Err(S3UriError::MissingKey)
        ));
        assert!(matches!(
            S3Uri::parse("s3://my-bucket/"),
            Err(S3UriError::MissingKey)
        ));
    }

    #[test]
    fn s3_uri_rejects_missing_bucket() {
        assert!(matches!(
            S3Uri::parse("s3:///key"),
            Err(S3UriError::MissingBucket)
        ));
    }

    // ---- MediaType ----

    #[test]
    fn media_type_accepts_typical() {
        let m = MediaType::parse("image/png").unwrap();
        assert_eq!(m.top_level(), "image");
        assert_eq!(m.subtype(), "png");
    }

    #[test]
    fn media_type_strips_parameter_from_subtype() {
        let m = MediaType::parse("text/plain; charset=utf-8").unwrap();
        assert_eq!(m.top_level(), "text");
        assert_eq!(m.subtype(), "plain");
    }

    #[test]
    fn media_type_rejects_missing_slash() {
        assert_eq!(MediaType::parse("image"), Err(MediaTypeError::MissingSlash));
    }

    #[test]
    fn media_type_rejects_empty_halves() {
        assert_eq!(
            MediaType::parse("/png"),
            Err(MediaTypeError::MissingTopLevel)
        );
        assert_eq!(
            MediaType::parse("image/"),
            Err(MediaTypeError::MissingSubtype)
        );
    }

    #[test]
    fn media_type_rejects_disallowed_chars() {
        assert!(matches!(
            MediaType::parse("image/png?"),
            Err(MediaTypeError::InvalidCharacter)
        ));
    }

    // ---- AwsAccountId ----

    #[test]
    fn aws_account_id_accepts_twelve_digits() {
        let a = AwsAccountId::parse("123456789012").unwrap();
        assert_eq!(a.as_str(), "123456789012");
    }

    #[test]
    fn aws_account_id_rejects_wrong_length() {
        assert!(matches!(
            AwsAccountId::parse("123"),
            Err(AwsAccountIdError::NotTwelveDigits)
        ));
    }

    #[test]
    fn aws_account_id_rejects_non_digit() {
        assert!(matches!(
            AwsAccountId::parse("12345678901a"),
            Err(AwsAccountIdError::NotTwelveDigits)
        ));
    }

    // ---- ProviderFileId ----

    #[test]
    fn provider_file_id_accepts_typical() {
        let f = ProviderFileId::parse("file-abc123").unwrap();
        assert_eq!(f.as_str(), "file-abc123");
    }

    #[test]
    fn provider_file_id_rejects_empty() {
        assert!(matches!(
            ProviderFileId::parse(""),
            Err(ProviderFileIdError::Empty)
        ));
    }

    // ---- MediaSource ----

    #[test]
    fn media_source_kind_matches_variant() {
        let url = MediaSource::Url {
            url: HttpsUrl::parse("https://x/y").unwrap(),
        };
        assert_eq!(url.kind(), SourceKind::Url);
        let bytes = MediaSource::InlineBytes {
            mime: MediaType::parse("image/png").unwrap(),
            data: vec![1, 2, 3],
        };
        assert_eq!(bytes.kind(), SourceKind::InlineBytes);
    }

    #[test]
    fn media_source_url_round_trip() {
        let src = MediaSource::Url {
            url: HttpsUrl::parse("https://example.com/a.png").unwrap(),
        };
        let json = serde_json::to_string(&src).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["source"], "url");
        assert_eq!(value["url"], "https://example.com/a.png");
        let back: MediaSource = serde_json::from_str(&json).unwrap();
        assert_eq!(src, back);
    }

    #[test]
    fn media_source_inline_bytes_encodes_base64() {
        let src = MediaSource::InlineBytes {
            mime: MediaType::parse("image/png").unwrap(),
            data: b"hello".to_vec(),
        };
        let json = serde_json::to_string(&src).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["source"], "inline_bytes");
        assert_eq!(value["mime"], "image/png");
        assert_eq!(value["data"], "aGVsbG8="); // base64("hello")
        let back: MediaSource = serde_json::from_str(&json).unwrap();
        assert_eq!(src, back);
    }

    #[test]
    fn media_source_provider_file_round_trip() {
        let src = MediaSource::ProviderFile {
            file_id: ProviderFileId::parse("file-x").unwrap(),
        };
        let json = serde_json::to_string(&src).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["source"], "provider_file");
        assert_eq!(value["file_id"], "file-x");
        let back: MediaSource = serde_json::from_str(&json).unwrap();
        assert_eq!(src, back);
    }

    #[test]
    fn media_source_s3_with_bucket_owner_round_trip() {
        let src = MediaSource::S3 {
            uri: S3Uri::parse("s3://b/k").unwrap(),
            bucket_owner: Some(AwsAccountId::parse("123456789012").unwrap()),
        };
        let json = serde_json::to_string(&src).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["source"], "s3");
        assert_eq!(value["uri"], "s3://b/k");
        assert_eq!(value["bucket_owner"], "123456789012");
        let back: MediaSource = serde_json::from_str(&json).unwrap();
        assert_eq!(src, back);
    }

    #[test]
    fn media_source_s3_without_bucket_owner_omits_field() {
        let src = MediaSource::S3 {
            uri: S3Uri::parse("s3://b/k").unwrap(),
            bucket_owner: None,
        };
        let json = serde_json::to_string(&src).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(value.get("bucket_owner").is_none());
        let back: MediaSource = serde_json::from_str(&json).unwrap();
        assert_eq!(src, back);
    }

    // ---- SourceKind / EnumSet ----

    #[test]
    fn source_kind_enumset_round_trip() {
        let mut set = EnumSet::new();
        set.insert(SourceKind::Url);
        set.insert(SourceKind::InlineBytes);
        let json = serde_json::to_string(&set).unwrap();
        // serialize_repr = "list" → array of variant names in declaration order.
        assert!(json.contains("url"));
        assert!(json.contains("inline_bytes"));
        let back: EnumSet<SourceKind> = serde_json::from_str(&json).unwrap();
        assert_eq!(set, back);
    }

    #[test]
    fn all_source_kinds_contains_every_variant() {
        let all = all_source_kinds();
        assert!(all.contains(SourceKind::Url));
        assert!(all.contains(SourceKind::InlineBytes));
        assert!(all.contains(SourceKind::ProviderFile));
        assert!(all.contains(SourceKind::S3));
    }
}
