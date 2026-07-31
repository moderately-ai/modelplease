// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `LanguageModelProvider` trait — single seam for chat / completion providers.
//!
//! Each provider impl (Anthropic, OpenAI, Ollama, Bedrock, ...) holds its
//! credentials and HTTP client privately and serves any model the
//! provider supports. The application router routes
//! `provider/model` model_ids to the right provider based on
//! [`LanguageModelProvider::name`].

use std::{collections::BTreeMap, pin::Pin};

use async_trait::async_trait;
use futures::Stream;

use crate::{
    capabilities::{
        CapabilityError, MediaKind, MediaSupport, ModelCapabilities, ReasoningCapability,
    },
    config::{LanguageModelConfig, ResponseFormat},
    error::LanguageModelError,
    identifiers::ModelId,
    message::Message,
    response::{LanguageModelResponse, StreamDelta},
};

/// Mirror of [`ResponseFormat`] without the per-variant payload.
///
/// Used in [`ChatModelInfo::supported_response_formats`] to advertise
/// which formats a model accepts; callers filter their model picker
/// against this. The corresponding [`ResponseFormat`] still carries
/// the schema / name / strict fields when invoking generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResponseFormatKind {
    Text,
    JsonObject,
    JsonSchema,
}

impl ResponseFormatKind {
    #[must_use]
    pub const fn from_format(format: &ResponseFormat) -> Self {
        match format {
            ResponseFormat::Text => Self::Text,
            ResponseFormat::JsonObject => Self::JsonObject,
            ResponseFormat::JsonSchema { .. } => Self::JsonSchema,
        }
    }
}

/// Per-call inputs for [`LanguageModelProvider::generate`] and
/// [`LanguageModelProvider::generate_stream`].
///
/// Bundling avoids transposition between fields. Because callers may construct
/// this value with a struct literal, adding fields is a semver-sensitive change.
pub struct GenerateRequest<'a> {
    /// Model identifier (provider-specific name, e.g.
    /// `"claude-sonnet-4-20250514"`). The [`ModelId`] newtype prevents
    /// passing it where an [`ApiKey`](crate::ApiKey) is expected at
    /// construction time.
    pub model: &'a ModelId,
    /// Conversation messages — the provider extracts the system
    /// prompt and serializes per its own conventions.
    pub messages: &'a [Message],
    /// Generation knobs (temperature, max_tokens, response_format, etc.).
    pub config: &'a LanguageModelConfig,
}

/// Catalog metadata for one model offered by a provider.
///
/// Populated via the hybrid dynamic-fetch + local-augment pattern:
/// provider-supplied IDs are merged with a static capability table in
/// the impl. IDs returned by upstream that have no local entry land
/// with conservative defaults (`context_window: None`,
/// `supports_*: false`, empty `media_support`) and emit a warning per
/// cache fill.
///
/// `Eq` is intentionally not derived — `ReasoningCapability` carries an
/// `Option<RangeInclusive<f64>>` for `top_p` clamping and `f64` is not
/// `Eq`. Callers needing equality use `PartialEq`.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatModelInfo {
    pub id: ModelId,
    /// Human-readable name when the upstream provides one (Anthropic
    /// returns `display_name`, OpenAI doesn't).
    pub display_name: Option<String>,
    /// Maximum context length in tokens. `None` when neither the
    /// upstream catalog nor the local table knows.
    pub context_window: Option<u32>,
    /// Whether `generate_stream` is expected to work for this model.
    pub supports_streaming: bool,
    /// Response formats the model accepts. Empty means unknown — treat
    /// as "try and see". Callers selecting a model should filter on
    /// this rather than discovering at generate time.
    pub supported_response_formats: Vec<ResponseFormatKind>,
    /// Media modalities and source kinds the model accepts. Missing
    /// keys = modality unsupported; empty map = text-only model. See
    /// [`crate::ModelCapabilities`] for the runtime-checked counterpart
    /// surfaced by [`LanguageModelProvider::capabilities`].
    pub media_support: BTreeMap<MediaKind, MediaSupport>,
    /// Reasoning / extended-thinking surface this model exposes.
    /// `None` ⇒ no reasoning support; any non-`Off`
    /// [`crate::ReasoningConfig`] supplied for this model will fail
    /// validation upstream of the provider call.
    pub reasoning: Option<ReasoningCapability>,
}

/// A chat / completion provider — one impl per backend.
///
/// Constructed once at boot with credentials baked into the impl;
/// the application router routes requests to the right impl
/// based on the provider name parsed from a `provider/model`
/// model_id.
#[async_trait]
pub trait LanguageModelProvider: Send + Sync {
    /// Stable provider key — used by the application router for
    /// `model_id.split_once('/')` routing. Examples: `"anthropic"`,
    /// `"openai"`, `"ollama"`, `"bedrock"`. Must be unique across the
    /// providers registered for one modality.
    fn name(&self) -> &'static str;

    /// Catalog of models this provider serves.
    ///
    /// Remote impls fetch upstream and merge with a local capability
    /// table; results are cached (TTL ~1h) so repeat calls don't hit
    /// the network. Local impls (e.g. a dummy provider for tests)
    /// return their static set directly.
    async fn list_models(&self) -> Result<Vec<ChatModelInfo>, LanguageModelError>;

    /// Generate a complete response.
    async fn generate(
        &self,
        request: GenerateRequest<'_>,
    ) -> Result<LanguageModelResponse, LanguageModelError>;

    /// Generate a streamed response — token deltas as a `Stream`.
    async fn generate_stream(
        &self,
        request: GenerateRequest<'_>,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamDelta, LanguageModelError>> + Send>>,
        LanguageModelError,
    >;

    /// Sync capability lookup against the provider's static
    /// `MODEL_CAPABILITIES` table. Returns `None` when the model is
    /// not in the table (treat as conservative-deny for media content;
    /// pure text calls bypass this lookup).
    ///
    /// Synchronous because every provider's table is in-process — no
    /// network round trip required. The async [`Self::list_models`]
    /// surface remains for live catalog discovery.
    fn capabilities(&self, model: &ModelId) -> Option<ModelCapabilities>;

    /// Validate that `request` only carries content parts the model
    /// actually accepts. Default impl walks `request.messages`, looks
    /// up the model in the provider's capability table, and returns
    /// the first [`CapabilityError`] variant that applies — or `Ok(())`
    /// for pure-text requests against text-only models.
    ///
    /// Providers should call this from `generate` /
    /// `generate_stream` before any network work. Override only if
    /// extra provider-specific checks need to run before/after the
    /// default scan.
    fn validate_request(&self, request: &GenerateRequest<'_>) -> Result<(), CapabilityError> {
        // Fast path: scan once for non-text parts. Pure-text requests
        // skip the entire capability lookup so text-only models with
        // empty `media_support` stay zero-cost.
        let has_media = request
            .messages
            .iter()
            .any(|m| m.content.iter().any(|p| p.media_kind().is_some()));
        if !has_media {
            return Ok(());
        }

        let caps = self.capabilities(request.model);
        let model_label = request.model.as_str();

        for msg in request.messages {
            for part in &msg.content {
                let Some(kind) = part.media_kind() else {
                    continue;
                };
                let Some(source) = part.media_source() else {
                    continue;
                };
                let Some(caps_ref) = caps.as_ref() else {
                    return Err(CapabilityError::UnknownModel {
                        model: model_label.to_owned(),
                        kind,
                    });
                };
                caps_ref.validate(kind, source)?;
            }
        }
        Ok(())
    }
}
