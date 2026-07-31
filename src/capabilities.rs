// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-model media capability declarations + the validation error type
//! returned when a request asks for something a model doesn't support.
//!
//! Capability checking has two layers:
//!
//! 1. **Runtime**: every provider impl publishes a static `MODEL_CAPABILITIES` table that maps its
//!    catalog ids to [`ModelCapabilities`] entries.
//!    [`LanguageModelProvider::capabilities`](crate::LanguageModelProvider::capabilities) returns
//!    the entry for a given model; the default impl of
//!    [`LanguageModelProvider::validate_request`](crate::LanguageModelProvider::validate_request)
//!    walks every non-text content part and consults the table, returning a typed
//!    [`CapabilityError`] before any wire call.
//!
//! 2. **Compile-time** (concrete callers only): the `Accepts*` marker traits below let
//!    known-concrete provider call sites (predict optimizers wired to a specific provider,
//!    examples, tests) refuse to compile if they hand the wrong source kind to the wrong provider.
//!    The markers vanish under `Arc<dyn LanguageModelProvider>` — that's by design; the dyn path
//!    relies on Layer 1's runtime check.

use std::{collections::BTreeMap, ops::RangeInclusive};

use enumset::{EnumSet, EnumSetType};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    config::{ReasoningConfig, ReasoningEffort},
    media::{MediaSource, MediaType, SourceKind},
};

/// Bucket for media content parts.
///
/// Discriminates the four modalities we expose at the
/// [`ContentPart`](crate::ContentPart) level. Iteration order is
/// variant declaration order via `Ord`, which matches the workspace's
/// "ordered iteration when observable" rule for any logging or error
/// surfaces that walk a `BTreeMap<MediaKind, ...>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaKind {
    Image,
    Document,
    Audio,
    Video,
}

impl MediaKind {
    /// Bucket an RFC-6838 top-level media type into a [`MediaKind`].
    ///
    /// `image/*` → `Image`, `audio/*` → `Audio`, `video/*` → `Video`,
    /// `application/*` and `text/*` → `Document`. Returns `None` for
    /// anything else (e.g. `multipart/*`, `message/*`) — those need
    /// explicit handling we haven't designed yet.
    #[must_use]
    pub fn from_media_type(media_type: &MediaType) -> Option<Self> {
        match media_type.top_level() {
            "image" => Some(Self::Image),
            "audio" => Some(Self::Audio),
            "video" => Some(Self::Video),
            "application" | "text" => Some(Self::Document),
            _ => None,
        }
    }

    /// Human-readable label used in error messages and prompt strings.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Document => "document",
            Self::Audio => "audio",
            Self::Video => "video",
        }
    }
}

impl std::fmt::Display for MediaKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// What a model accepts for one [`MediaKind`].
///
/// Stored on [`ModelCapabilities::media_support`] keyed by `MediaKind`.
/// A missing entry (vs. an empty `MediaSupport`) means "not supported";
/// an entry with empty `sources` is a bug in the table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaSupport {
    /// Source kinds the provider can carry to this model for this
    /// modality (e.g. Bedrock Image = `{InlineBytes, S3}`, no `Url`).
    pub sources: EnumSet<SourceKind>,
    /// Accepted RFC-6838 subtypes (e.g. `["png", "jpeg", "gif", "webp"]`).
    /// Matched against [`MediaType::subtype`](crate::MediaType::subtype).
    /// Empty means "the provider didn't publish a constraint; accept
    /// any subtype" — use sparingly.
    pub formats: &'static [&'static str],
    /// Max byte size accepted for [`MediaSource::InlineBytes`]. `None`
    /// = no local check (e.g. for `Url`/`ProviderFile`/`S3` sources the
    /// size is opaque locally).
    pub max_bytes: Option<u64>,
    /// Max number of parts of this kind per message. `None` = no
    /// declared limit; counts are still cheap to enforce locally.
    pub max_count_per_message: Option<u8>,
}

impl MediaSupport {
    /// Validate one media source against this support entry.
    ///
    /// `URL`, `ProviderFile`, and `S3` sources skip format/size checks
    /// — the bytes aren't reachable locally and the provider performs
    /// the equivalent check server-side. `InlineBytes` validates
    /// `mime.subtype()` against [`Self::formats`] and `data.len()` against
    /// [`Self::max_bytes`].
    ///
    /// # Errors
    /// Returns the first matching [`CapabilityError`] variant.
    pub fn validate(
        &self,
        model: &str,
        kind: MediaKind,
        source: &MediaSource,
    ) -> Result<(), CapabilityError> {
        let source_kind = source.kind();
        if !self.sources.contains(source_kind) {
            return Err(CapabilityError::SourceKindUnsupported {
                model: model.to_owned(),
                kind,
                attempted: source_kind,
                accepted: self.sources,
            });
        }
        if let MediaSource::InlineBytes { mime, data } = source {
            if !self.formats.is_empty() {
                let subtype = mime.subtype();
                if !self.formats.contains(&subtype) {
                    return Err(CapabilityError::FormatUnsupported {
                        model: model.to_owned(),
                        kind,
                        format: subtype.to_owned(),
                        accepted: self.formats.to_vec(),
                    });
                }
            }
            if let Some(max) = self.max_bytes {
                let bytes = u64::try_from(data.len()).unwrap_or(u64::MAX);
                if bytes > max {
                    return Err(CapabilityError::SizeExceeded {
                        model: model.to_owned(),
                        kind,
                        bytes,
                        max,
                    });
                }
            }
        }
        Ok(())
    }
}

/// What a single model can carry across every [`MediaKind`].
///
/// Returned by
/// [`LanguageModelProvider::capabilities`](crate::LanguageModelProvider::capabilities).
/// `media_support` keys are deterministic via `BTreeMap` so iteration
/// (used in error messages and logging) doesn't change between runs.
///
/// `Eq` is intentionally not derived — `ReasoningCapability` carries an
/// `Option<RangeInclusive<f64>>` for `top_p` clamping and `f64` is not
/// `Eq`. Callers needing equality use `PartialEq`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelCapabilities {
    /// Provider-specific model id (matches
    /// [`ChatModelInfo.id`](crate::ChatModelInfo)).
    pub model_id: String,
    /// One [`MediaSupport`] per [`MediaKind`] the model accepts. Missing
    /// keys → modality unsupported. Empty map → text-only model.
    pub media_support: BTreeMap<MediaKind, MediaSupport>,
    /// Reasoning / extended-thinking capability. `None` = model has no
    /// reasoning surface at all; any non-Off `ReasoningConfig` will fail
    /// validation upstream.
    pub reasoning: Option<ReasoningCapability>,
    /// Whether the model accepts AWS Bedrock latency-optimized
    /// ("accelerated") inference. `false` for every non-Bedrock provider
    /// and for Bedrock models AWS doesn't list as latency-eligible. The
    /// caller gates [`LatencyMode::Optimized`](crate::config::LatencyMode::Optimized)
    /// on this flag and fails loud before the wire.
    pub latency_optimized_supported: bool,
    /// Whether the model supports extended (1-hour) prompt-cache TTL. Basic
    /// 5-minute caching is broader; the 1-hour tier is Anthropic-only (and,
    /// on Bedrock, only the Claude 4.5+ family). The caller gates an
    /// explicit [`CacheTtl::OneHour`](crate::config::CacheTtl::OneHour) on
    /// this flag and fails loud, so a model that can't accept it never
    /// receives a 1-hour `cachePoint`.
    pub extended_cache_ttl_supported: bool,
}

impl ModelCapabilities {
    /// Validate one (kind, source) pair against this model.
    ///
    /// # Errors
    /// - [`CapabilityError::ModalityUnsupported`] when no entry exists for `kind`.
    /// - Anything [`MediaSupport::validate`] can return.
    pub fn validate(&self, kind: MediaKind, source: &MediaSource) -> Result<(), CapabilityError> {
        let support =
            self.media_support
                .get(&kind)
                .ok_or_else(|| CapabilityError::ModalityUnsupported {
                    model: self.model_id.clone(),
                    kind,
                })?;
        support.validate(&self.model_id, kind, source)
    }
}

/// Provider wire-mode classes a model accepts.
///
/// `Adaptive` covers OpenAI's `reasoning_effort` string, Ollama's
/// OpenAI-compat `reasoning_effort`, and Anthropic adaptive-thinking
/// (`thinking: {type: "adaptive", effort: ...}`). `Manual` is
/// Anthropic-family only — `thinking: {type: "enabled", budget_tokens:
/// N}` natively and the same shape on Bedrock via
/// `additionalModelRequestFields`.
/// `EnumSetType` auto-derives `Copy + Clone + PartialEq + Eq`; `Debug`
/// and `Hash` are derived separately.
#[derive(EnumSetType, Debug, Hash)]
pub enum ReasoningMode {
    Adaptive,
    Manual,
}

impl ReasoningMode {
    /// Human-readable label used in `ReasoningValidationError` messages.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Adaptive => "adaptive",
            Self::Manual => "manual",
        }
    }
}

impl std::fmt::Display for ReasoningMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Sampling-parameter restrictions that apply when reasoning is on.
///
/// Anthropic-family upstream rejects `temperature` and `top_k` outright
/// when `thinking` is enabled and clamps `top_p` to `[0.95, 1]`. OpenAI
/// and Ollama have no such restrictions. Carried per-model so the
/// caller can produce precise errors that name the offending field
/// + the constraint upstream actually enforces.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ReasoningParamConflicts {
    /// `true` ⇒ the model rejects a non-`None` `temperature` whenever
    /// reasoning is on.
    pub temperature_forbidden: bool,
    /// `true` ⇒ the model rejects a non-`None` `top_k`.
    pub top_k_forbidden: bool,
    /// `Some(range)` ⇒ `top_p` is only honoured inside the inclusive
    /// range; values outside fail validation. `None` ⇒ unrestricted.
    pub top_p_allowed_range: Option<RangeInclusive<f64>>,
}

/// What reasoning a single model supports.
///
/// Populated alongside [`MediaSupport`] in each provider's static
/// `MODEL_CAPABILITIES` table and exposed on [`ModelCapabilities`].
/// Callers reads this synchronously before any wire call and
/// rejects user configurations whose intent doesn't fit.
#[derive(Debug, Clone, PartialEq)]
pub struct ReasoningCapability {
    /// Modes this model accepts. Non-empty by construction — a model
    /// with no reasoning capability omits the whole `ReasoningCapability`
    /// from `ModelCapabilities.reasoning`.
    pub supported_modes: EnumSet<ReasoningMode>,
    /// Effort levels accepted in `Adaptive` (and OpenAI-shape) mode.
    /// Also non-empty by construction when `Adaptive ∈ supported_modes`.
    pub supported_efforts: EnumSet<ReasoningEffort>,
    /// `budget_tokens` range when `Manual ∈ supported_modes`. `None`
    /// when manual mode isn't supported.
    pub manual_budget_range: Option<RangeInclusive<u32>>,
    /// Sampling-parameter restrictions when reasoning is on.
    pub conflicts: ReasoningParamConflicts,
    /// `true` ⇒ the model removed `temperature` / `top_p` / `top_k` entirely
    /// (the 4.7+ Anthropic generation — Opus 4.7/4.8, Sonnet 5) and rejects
    /// them with a 400 in *every* request, regardless of reasoning state.
    /// The caller suppresses its default-temperature injection and fails
    /// loud on any user-explicit sampling value for such models. `false` ⇒
    /// the `conflicts` above apply only when reasoning is active (the
    /// 4.6-and-earlier rule, where sampling is accepted with thinking off).
    pub sampling_params_removed: bool,
}

impl ReasoningCapability {
    /// Check a resolved [`ReasoningConfig`] against this capability.
    ///
    /// Returns the first violation, if any. `ReasoningConfig::Off` always
    /// passes — disabling reasoning is universally allowed at the
    /// reasoning-capability layer (some models like Mythos Preview reject
    /// `thinking: {type: "disabled"}` at the wire, but that's a per-model
    /// concern the provider impl surfaces, not a general rule here).
    ///
    /// # Errors
    /// Returns the precise [`ReasoningValidationError`] variant identifying
    /// which constraint failed.
    pub fn validate(
        &self,
        model_id: &str,
        config: &ReasoningConfig,
    ) -> Result<(), ReasoningValidationError> {
        match config {
            ReasoningConfig::Off => Ok(()),
            ReasoningConfig::Adaptive { effort } => {
                if !self.supported_modes.contains(ReasoningMode::Adaptive) {
                    return Err(ReasoningValidationError::ModeUnsupported {
                        model: model_id.to_owned(),
                        requested: ReasoningMode::Adaptive,
                        supported: self.supported_modes,
                    });
                }
                if !self.supported_efforts.contains(*effort) {
                    return Err(ReasoningValidationError::EffortUnsupported {
                        model: model_id.to_owned(),
                        requested: *effort,
                        supported: self.supported_efforts,
                    });
                }
                Ok(())
            }
            ReasoningConfig::Manual { budget_tokens } => {
                if !self.supported_modes.contains(ReasoningMode::Manual) {
                    return Err(ReasoningValidationError::ModeUnsupported {
                        model: model_id.to_owned(),
                        requested: ReasoningMode::Manual,
                        supported: self.supported_modes,
                    });
                }
                let range = self.manual_budget_range.as_ref().ok_or_else(|| {
                    ReasoningValidationError::ModeUnsupported {
                        model: model_id.to_owned(),
                        requested: ReasoningMode::Manual,
                        supported: self.supported_modes,
                    }
                })?;
                if !range.contains(budget_tokens) {
                    return Err(ReasoningValidationError::BudgetOutOfRange {
                        model: model_id.to_owned(),
                        requested: *budget_tokens,
                        min: *range.start(),
                        max: *range.end(),
                    });
                }
                Ok(())
            }
        }
    }
}

/// Reasons a [`ReasoningConfig`] can fail [`ReasoningCapability::validate`].
///
/// Each variant names the offending field and what the model actually
/// accepts so the caller can build a clean user-facing error without
/// pulling the model back out of the lookup.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ReasoningValidationError {
    /// Model has no reasoning capability at all (`ModelCapabilities.reasoning ==
    /// None`) but the caller supplied a non-`Off` config.
    #[error("model `{model}` does not support reasoning")]
    Unsupported { model: String },

    /// The requested mode (Adaptive / Manual) is outside the model's
    /// `supported_modes` set.
    #[error(
        "model `{model}` does not support {requested} reasoning mode (supported: {supported:?})"
    )]
    ModeUnsupported {
        model: String,
        requested: ReasoningMode,
        supported: EnumSet<ReasoningMode>,
    },

    /// The requested effort level is outside the model's `supported_efforts`
    /// set for the chosen mode.
    #[error(
        "model `{model}` does not support reasoning effort `{requested}` (supported: {supported:?})"
    )]
    EffortUnsupported {
        model: String,
        requested: ReasoningEffort,
        supported: EnumSet<ReasoningEffort>,
    },

    /// `Manual { budget_tokens }` is outside the model's
    /// `manual_budget_range`.
    #[error(
        "model `{model}` rejects manual budget_tokens={requested} \
         (supported range: {min}..={max})"
    )]
    BudgetOutOfRange {
        model: String,
        requested: u32,
        min: u32,
        max: u32,
    },
}

/// Reasons a request can fail capability validation, by precision.
///
/// Returned from
/// [`LanguageModelProvider::validate_request`](crate::LanguageModelProvider::validate_request)
/// and surfaced both at the application boundary (predict / direct
/// generate) and at the application configuration preflight (where
/// the same validation runs against the *declared* schema rather than
/// the actual values).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CapabilityError {
    /// The model has no [`MediaSupport`] entry for this modality at all.
    #[error("model `{model}` does not accept {kind} content")]
    ModalityUnsupported { model: String, kind: MediaKind },

    /// The model accepts this modality but not via this source kind.
    /// `accepted` enumerates what would work. The field is named
    /// `attempted` (rather than `source`) so thiserror doesn't try to
    /// treat it as an `Error::source()` provider — it's a discriminant,
    /// not a chained error.
    #[error(
        "model `{model}` accepts {kind} content but not via {attempted:?} (accepted: {accepted:?})"
    )]
    SourceKindUnsupported {
        model: String,
        kind: MediaKind,
        attempted: SourceKind,
        accepted: EnumSet<SourceKind>,
    },

    /// The model accepts this modality + source but not this MIME
    /// subtype (e.g. JPEG image submitted to a model that only takes
    /// PNG).
    #[error("model `{model}` rejects {kind} format `{format}` (accepted: {accepted:?})")]
    FormatUnsupported {
        model: String,
        kind: MediaKind,
        format: String,
        accepted: Vec<&'static str>,
    },

    /// `InlineBytes` payload exceeds the model's declared cap.
    #[error("{kind} payload {bytes} exceeds model `{model}` cap {max}")]
    SizeExceeded {
        model: String,
        kind: MediaKind,
        bytes: u64,
        max: u64,
    },

    /// The capability table has no entry for `model` at all — the model
    /// is either too new (unmodeled) or misspelled. Surfaced when a
    /// non-text content part is sent against an unknown model. Pure
    /// text calls do not produce this; they bypass capability lookup.
    #[error("model `{model}` has no capability metadata; cannot validate {kind} content")]
    UnknownModel { model: String, kind: MediaKind },
}

// =============================================================================
// Marker traits — compile-time capability gating for concrete-typed callers.
//
// Each marker trait expresses one (modality, source kind) pair that a
// concrete provider impl publishes. Concrete-typed call sites can bound
// their generic on the markers they need; the bound disappears under
// `Arc<dyn LanguageModelProvider>`, so the dyn path falls back to Layer
// 1's runtime capability check. There is no `dyn`-bridge by design.
//
// Provider impls declare these in their own module. The presence/absence
// here describes the trait shape only.
// =============================================================================

/// Implementor accepts at least one model that takes
/// `MediaSource::Url` for [`MediaKind::Image`]. (OpenAI, Anthropic.)
pub trait AcceptsImageUrl {}

/// Implementor accepts at least one model that takes
/// `MediaSource::InlineBytes` for [`MediaKind::Image`]. (All providers
/// except text-only ones.)
pub trait AcceptsImageBytes {}

/// Implementor accepts at least one model that takes
/// `MediaSource::S3` for [`MediaKind::Image`]. (Bedrock only.)
pub trait AcceptsImageS3 {}

/// Implementor accepts at least one model that takes
/// `MediaSource::InlineBytes` for [`MediaKind::Audio`]. (`OpenAI`
/// gpt-audio family, Bedrock Voxtral.)
pub trait AcceptsAudioBytes {}

/// Implementor accepts at least one model that takes
/// `MediaSource::InlineBytes` for [`MediaKind::Document`]. (Anthropic,
/// `OpenAI` file API, Bedrock.)
pub trait AcceptsDocumentBytes {}

/// Implementor accepts at least one model that takes
/// `MediaSource::InlineBytes` for [`MediaKind::Video`]. (Bedrock Nova
/// Pro/Lite only.)
pub trait AcceptsVideoBytes {}

/// Implementor accepts at least one model that takes
/// `MediaSource::S3` for [`MediaKind::Video`]. (Bedrock only.)
pub trait AcceptsVideoS3 {}

#[cfg(test)]
mod tests {
    use enumset::enum_set;

    use super::*;
    use crate::media::{HttpsUrl, MediaType};

    fn anthropic_image_support() -> MediaSupport {
        MediaSupport {
            sources: enum_set!(
                SourceKind::Url | SourceKind::InlineBytes | SourceKind::ProviderFile
            ),
            formats: &["png", "jpeg", "gif", "webp"],
            max_bytes: Some(5 * 1024 * 1024),
            max_count_per_message: None,
        }
    }

    fn bedrock_image_support() -> MediaSupport {
        MediaSupport {
            sources: enum_set!(SourceKind::InlineBytes | SourceKind::S3),
            formats: &["png", "jpeg", "gif", "webp"],
            max_bytes: Some(3_932_160), // 3.75 MB
            max_count_per_message: None,
        }
    }

    #[test]
    fn media_kind_from_media_type_buckets_correctly() {
        let png = MediaType::parse("image/png").unwrap();
        assert_eq!(MediaKind::from_media_type(&png), Some(MediaKind::Image));
        let mp3 = MediaType::parse("audio/mpeg").unwrap();
        assert_eq!(MediaKind::from_media_type(&mp3), Some(MediaKind::Audio));
        let mp4 = MediaType::parse("video/mp4").unwrap();
        assert_eq!(MediaKind::from_media_type(&mp4), Some(MediaKind::Video));
        let pdf = MediaType::parse("application/pdf").unwrap();
        assert_eq!(MediaKind::from_media_type(&pdf), Some(MediaKind::Document));
        let txt = MediaType::parse("text/plain").unwrap();
        assert_eq!(MediaKind::from_media_type(&txt), Some(MediaKind::Document));
        let multipart = MediaType::parse("multipart/form-data").unwrap();
        assert_eq!(MediaKind::from_media_type(&multipart), None);
    }

    #[test]
    fn support_accepts_url_when_listed() {
        let support = anthropic_image_support();
        let src = MediaSource::Url {
            url: HttpsUrl::parse("https://x/y.png").unwrap(),
        };
        assert!(support.validate("claude", MediaKind::Image, &src).is_ok());
    }

    #[test]
    fn support_rejects_url_when_not_listed() {
        let support = bedrock_image_support();
        let src = MediaSource::Url {
            url: HttpsUrl::parse("https://x/y.png").unwrap(),
        };
        let err = support
            .validate("bedrock-claude", MediaKind::Image, &src)
            .unwrap_err();
        assert!(matches!(
            err,
            CapabilityError::SourceKindUnsupported {
                attempted: SourceKind::Url,
                ..
            }
        ));
    }

    #[test]
    fn support_rejects_inline_bytes_with_wrong_subtype() {
        let support = anthropic_image_support();
        let src = MediaSource::InlineBytes {
            mime: MediaType::parse("image/bmp").unwrap(),
            data: vec![0, 1, 2, 3],
        };
        let err = support
            .validate("claude", MediaKind::Image, &src)
            .unwrap_err();
        match err {
            CapabilityError::FormatUnsupported {
                format, accepted, ..
            } => {
                assert_eq!(format, "bmp");
                assert_eq!(accepted, vec!["png", "jpeg", "gif", "webp"]);
            }
            other => panic!("expected FormatUnsupported, got {other:?}"),
        }
    }

    #[test]
    fn support_rejects_oversize_inline_bytes() {
        let support = MediaSupport {
            sources: enum_set!(SourceKind::InlineBytes),
            formats: &["png"],
            max_bytes: Some(8),
            max_count_per_message: None,
        };
        let src = MediaSource::InlineBytes {
            mime: MediaType::parse("image/png").unwrap(),
            data: vec![0; 16],
        };
        let err = support.validate("m", MediaKind::Image, &src).unwrap_err();
        match err {
            CapabilityError::SizeExceeded { bytes, max, .. } => {
                assert_eq!(bytes, 16);
                assert_eq!(max, 8);
            }
            other => panic!("expected SizeExceeded, got {other:?}"),
        }
    }

    #[test]
    fn support_skips_format_check_for_non_inline_sources() {
        let support = bedrock_image_support();
        // S3 source — format/size unknown locally; validate succeeds.
        let src = MediaSource::S3 {
            uri: crate::media::S3Uri::parse("s3://bucket/key").unwrap(),
            bucket_owner: None,
        };
        assert!(
            support
                .validate("bedrock-claude", MediaKind::Image, &src)
                .is_ok()
        );
    }

    #[test]
    fn capabilities_rejects_unknown_modality() {
        let caps = ModelCapabilities {
            model_id: "claude".to_owned(),
            media_support: BTreeMap::from([(MediaKind::Image, anthropic_image_support())]),
            reasoning: None,
            latency_optimized_supported: false,
            extended_cache_ttl_supported: false,
        };
        let src = MediaSource::InlineBytes {
            mime: MediaType::parse("audio/mpeg").unwrap(),
            data: vec![0],
        };
        let err = caps.validate(MediaKind::Audio, &src).unwrap_err();
        assert!(matches!(
            err,
            CapabilityError::ModalityUnsupported {
                kind: MediaKind::Audio,
                ..
            }
        ));
    }

    #[test]
    fn capabilities_routes_through_to_support_validate() {
        let caps = ModelCapabilities {
            model_id: "claude".to_owned(),
            media_support: BTreeMap::from([(MediaKind::Image, anthropic_image_support())]),
            reasoning: None,
            latency_optimized_supported: false,
            extended_cache_ttl_supported: false,
        };
        let src = MediaSource::Url {
            url: HttpsUrl::parse("https://x/y.png").unwrap(),
        };
        assert!(caps.validate(MediaKind::Image, &src).is_ok());
    }
}
