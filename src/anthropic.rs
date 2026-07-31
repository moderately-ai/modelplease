// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Anthropic Messages API implementation of [`LanguageModelProvider`].
//!
//! Wire-format helpers (request building, response parsing, SSE handling, content-part
//! translation, error mapping) live in [`crate::anthropic_wire`] and are shared with
//! [`crate::bedrock_mantle::BedrockMantleProvider`]'s Anthropic Messages surface.

use std::{
    collections::{BTreeMap, HashMap},
    pin::Pin,
    sync::{Arc, LazyLock},
    time::Duration,
};

use async_trait::async_trait;
use enumset::enum_set;
use futures::stream::{Stream, StreamExt};
use moka::future::Cache;
use serde::Deserialize;

use crate::{
    anthropic_wire::{
        AnthropicResponse, anthropic_reasoning_conflicts, apply_anthropic_beta_headers,
        build_request, convert_anthropic_sse_event, map_anthropic_error,
        model_supports_prompt_caching, parse_anthropic_response,
    },
    capabilities::{
        MediaKind, MediaSupport, ModelCapabilities, ReasoningCapability, ReasoningMode,
    },
    config::{CacheTtl, LanguageModelConfig, PromptCaching, ReasoningEffort, ResponseFormat},
    error::LanguageModelError,
    identifiers::{ApiKey, ModelId},
    media::SourceKind,
    message::Message,
    provider::{ChatModelInfo, GenerateRequest, LanguageModelProvider, ResponseFormatKind},
    response::{LanguageModelResponse, StreamDelta},
    retry::{RetryConfig, with_retry},
    sse::parse_sse_stream,
};

/// Anthropic Messages API language model.
///
/// Calls `POST {base_url}/v1/messages` with the Anthropic Messages API contract.
///
/// # Examples
///
/// ```no_run
/// use std::sync::Arc;
/// use modelplease::{
///     AnthropicConfig, AnthropicDeps, AnthropicLanguageModel, ApiKey, RetryConfig,
/// };
///
/// # fn build() -> Result<AnthropicLanguageModel, modelplease::ApiKeyError> {
/// let lm = AnthropicLanguageModel::new(
///     AnthropicDeps { client: Arc::new(reqwest::Client::new()) },
///     AnthropicConfig {
///         api_key: ApiKey::parse("sk-ant-example")?,
///         base_url: AnthropicConfig::DEFAULT_BASE_URL.to_owned(),
///         retry_config: RetryConfig::default(),
///     },
/// );
/// # Ok(lm)
/// # }
/// ```
pub struct AnthropicLanguageModel {
    client: Arc<reqwest::Client>,
    api_key: String,
    base_url: String,
    retry_config: RetryConfig,
    /// Single-entry TTL cache for [`LanguageModelProvider::list_models`].
    /// `time_to_live = 1h`, `max_capacity = 1`. moka's `try_get_with` coalesces concurrent misses
    /// so a thundering herd produces one upstream call.
    list_models_cache: Cache<(), Vec<ChatModelInfo>>,
}

/// Configuration for [`AnthropicLanguageModel`].
///
/// One provider instance serves every model in Anthropic's catalog — the model identifier arrives
/// in [`GenerateRequest`] per call. `api_key` is the only required field; `base_url` and
/// `retry_config` have safe defaults.
pub struct AnthropicConfig {
    pub api_key: ApiKey,
    pub base_url: String,
    pub retry_config: RetryConfig,
}

impl AnthropicConfig {
    pub const DEFAULT_BASE_URL: &'static str = "https://api.anthropic.com";
}

/// Injected dependencies for [`AnthropicLanguageModel`]. The HTTP client is constructed once at
/// the application composition root and shared across HTTP-based providers.
pub struct AnthropicDeps {
    pub client: Arc<reqwest::Client>,
}

impl AnthropicLanguageModel {
    /// Create a new Anthropic language model.
    #[must_use]
    pub fn new(deps: AnthropicDeps, config: AnthropicConfig) -> Self {
        Self {
            client: deps.client,
            api_key: config.api_key.into_string(),
            base_url: config.base_url,
            retry_config: config.retry_config,
            list_models_cache: Cache::builder()
                .time_to_live(Duration::from_secs(3600))
                .max_capacity(1)
                .build(),
        }
    }

    /// Send the request and parse the response.
    async fn execute(
        &self,
        model: &str,
        messages: &[Message],
        config: &LanguageModelConfig,
    ) -> Result<LanguageModelResponse, LanguageModelError> {
        if matches!(config.response_format, ResponseFormat::JsonObject) {
            return Err(LanguageModelError::provider(
                "Anthropic does not support JsonObject response format; \
                 use JsonSchema or tool use for structured output",
            ));
        }

        let (mut request_body, needs_files_beta) = build_request(model, messages, config);

        if let ResponseFormat::JsonSchema { schema, .. } = &config.response_format {
            request_body.output_format = Some(serde_json::json!({
                "type": "json_schema",
                "schema": schema,
            }));
        }

        let url = format!("{}/v1/messages", self.base_url);

        let mut req = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json");

        req = apply_anthropic_beta_headers(
            req,
            matches!(config.response_format, ResponseFormat::JsonSchema { .. }),
            needs_files_beta,
            matches!(config.prompt_caching, PromptCaching::Auto)
                && model_supports_prompt_caching(model)
                && matches!(config.cache_ttl, CacheTtl::OneHour),
        );

        let response = req
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LanguageModelError::provider(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let headers = response.headers().clone();
            let body =
                crate::http_error::read_error_body_or_warn(response, "anthropic", status.as_u16())
                    .await;
            return Err(map_anthropic_error(status.as_u16(), &body, &headers));
        }

        let api_response: AnthropicResponse = response
            .json()
            .await
            .map_err(|e| LanguageModelError::provider(format!("failed to parse response: {e}")))?;
        parse_anthropic_response(api_response)
    }

    /// Shared streaming helper used by [`LanguageModelProvider::generate_stream`].
    async fn stream(
        &self,
        model: &str,
        messages: &[Message],
        config: &LanguageModelConfig,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamDelta, LanguageModelError>> + Send>>,
        LanguageModelError,
    > {
        if matches!(config.response_format, ResponseFormat::JsonObject) {
            return Err(LanguageModelError::provider(
                "Anthropic does not support JsonObject response format; \
                 use JsonSchema or tool use for structured output",
            ));
        }

        let (mut request_body, needs_files_beta) = build_request(model, messages, config);
        request_body.stream = Some(true);

        if let ResponseFormat::JsonSchema { schema, .. } = &config.response_format {
            request_body.output_format = Some(serde_json::json!({
                "type": "json_schema",
                "schema": schema,
            }));
        }

        let url = format!("{}/v1/messages", self.base_url);

        let mut req = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json");

        req = apply_anthropic_beta_headers(
            req,
            matches!(config.response_format, ResponseFormat::JsonSchema { .. }),
            needs_files_beta,
            matches!(config.prompt_caching, PromptCaching::Auto)
                && model_supports_prompt_caching(model)
                && matches!(config.cache_ttl, CacheTtl::OneHour),
        );

        let response = req
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LanguageModelError::provider(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let headers = response.headers().clone();
            let body =
                crate::http_error::read_error_body_or_warn(response, "anthropic", status.as_u16())
                    .await;
            return Err(map_anthropic_error(status.as_u16(), &body, &headers));
        }

        let byte_stream = response.bytes_stream();
        let sse_stream = parse_sse_stream(byte_stream);

        let delta_stream = sse_stream
            .filter_map(|event_result| async move { convert_anthropic_sse_event(event_result) });

        Ok(Box::pin(delta_stream))
    }

    /// Fetch and merge the upstream model catalog.
    ///
    /// Hits Anthropic's `GET /v1/models` for the live ID list, then walks each ID through the
    /// local [`MODEL_CAPABILITIES`] table to populate context-window / capability fields. IDs
    /// missing from the local table emit a single `tracing::warn!` per cache fill.
    async fn fetch_and_merge_models(&self) -> Result<Vec<ChatModelInfo>, LanguageModelError> {
        let url = format!("{}/v1/models", self.base_url);
        let response = self
            .client
            .get(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .send()
            .await
            .map_err(|e| LanguageModelError::provider(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let headers = response.headers().clone();
            let body =
                crate::http_error::read_error_body_or_warn(response, "anthropic", status.as_u16())
                    .await;
            return Err(map_anthropic_error(status.as_u16(), &body, &headers));
        }

        let api_response: AnthropicListModelsResponse = response
            .json()
            .await
            .map_err(|e| LanguageModelError::provider(format!("failed to parse models: {e}")))?;

        let mut out = Vec::with_capacity(api_response.data.len());
        for entry in api_response.data {
            let caps = MODEL_CAPABILITIES.get(entry.id.as_str()).cloned();
            if caps.is_none() {
                tracing::warn!(
                    provider = "anthropic",
                    model_id = %entry.id,
                    "model returned by upstream but no local capability metadata; \
                     ChatModelInfo will have minimal fields. \
                     Update the local capability table when ready."
                );
            }
            let mut formats = vec![ResponseFormatKind::Text];
            if caps.as_ref().is_some_and(|c| c.supports_json_schema) {
                formats.push(ResponseFormatKind::JsonSchema);
            }
            out.push(ChatModelInfo {
                id: ModelId::new(entry.id),
                display_name: entry.display_name,
                context_window: caps.as_ref().map(|c| c.context_window),
                supports_streaming: caps.as_ref().is_some_and(|c| c.supports_streaming),
                supported_response_formats: formats,
                media_support: caps
                    .as_ref()
                    .map(anthropic_media_support)
                    .unwrap_or_default(),
                reasoning: caps.as_ref().and_then(|c| c.reasoning.clone()),
            });
        }
        Ok(out)
    }
}

#[async_trait]
impl LanguageModelProvider for AnthropicLanguageModel {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    fn capabilities(&self, model: &ModelId) -> Option<ModelCapabilities> {
        let caps = MODEL_CAPABILITIES.get(model.as_str())?;
        Some(ModelCapabilities {
            model_id: model.as_str().to_owned(),
            media_support: anthropic_media_support(caps),
            reasoning: caps.reasoning.clone(),
            // Latency-optimized inference is a Bedrock-only tier.
            latency_optimized_supported: false,
            // Native Anthropic supports the 1-hour extended-cache-ttl beta across the Claude
            // models it serves.
            extended_cache_ttl_supported: true,
        })
    }

    #[tracing::instrument(skip(self), fields(provider = "anthropic", model_count = tracing::field::Empty), err(Display))]
    async fn list_models(&self) -> Result<Vec<ChatModelInfo>, LanguageModelError> {
        let result = self
            .list_models_cache
            .try_get_with((), self.fetch_and_merge_models())
            .await
            .map_err(|arc_err| (*arc_err).clone());
        if let Ok(ref v) = result {
            tracing::Span::current().record("model_count", v.len());
        }
        result
    }

    #[tracing::instrument(
        skip(self, request),
        fields(
            provider = "anthropic",
            model = %request.model,
            messages = request.messages.len(),
            prompt_tokens = tracing::field::Empty,
            completion_tokens = tracing::field::Empty,
            total_tokens = tracing::field::Empty,
            cache_creation_input_tokens = tracing::field::Empty,
            cache_read_input_tokens = tracing::field::Empty,
        ),
        err(Display),
    )]
    async fn generate(
        &self,
        request: GenerateRequest<'_>,
    ) -> Result<LanguageModelResponse, LanguageModelError> {
        self.validate_request(&request)?;
        let model = request.model.as_str();
        let response = with_retry(&self.retry_config, || {
            self.execute(model, request.messages, request.config)
        })
        .await?;
        let span = tracing::Span::current();
        if let Some(usage) = &response.usage {
            span.record("prompt_tokens", usage.input_tokens);
            span.record("completion_tokens", usage.output_tokens);
            span.record("total_tokens", usage.input_tokens + usage.output_tokens);
            span.record(
                "cache_creation_input_tokens",
                usage.cache_creation_input_tokens,
            );
            span.record("cache_read_input_tokens", usage.cache_read_input_tokens);
        }
        Ok(response)
    }

    #[tracing::instrument(
        skip(self, request),
        fields(
            provider = "anthropic",
            model = %request.model,
            messages = request.messages.len(),
            first_token_ms = tracing::field::Empty,
            prompt_tokens = tracing::field::Empty,
            completion_tokens = tracing::field::Empty,
            total_tokens = tracing::field::Empty,
            cache_creation_input_tokens = tracing::field::Empty,
            cache_read_input_tokens = tracing::field::Empty,
        ),
        err(Display),
    )]
    async fn generate_stream(
        &self,
        request: GenerateRequest<'_>,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamDelta, LanguageModelError>> + Send>>,
        LanguageModelError,
    > {
        let started_at = std::time::Instant::now();
        self.validate_request(&request)?;
        let inner = self
            .stream(request.model.as_str(), request.messages, request.config)
            .await?;
        let wrapped =
            crate::streaming_timing::instrument_stream(tracing::Span::current(), started_at, inner);
        Ok(Box::pin(wrapped))
    }
}

/// Static capability table seeded with well-known direct-Anthropic model IDs.
///
/// Hybrid dynamic-fetch + local-augment pattern: the upstream catalog gives us the live ID list,
/// we annotate each entry with capability flags from this table. IDs not in the table fire a
/// one-time warning per cache fill.
#[derive(Debug, Clone)]
struct ChatModelCapabilities {
    context_window: u32,
    supports_streaming: bool,
    supports_json_schema: bool,
    /// Reasoning surface — `None` ⇒ no reasoning support on this model.
    reasoning: Option<ReasoningCapability>,
}

/// `ReasoningCapability` for Claude Opus 4.7. Adaptive-only (manual mode returns 400 per the
/// per-model card). Effort set is the full Anthropic adaptive vocabulary — `xhigh` is documented
/// as Opus 4.7-only.
const fn anthropic_opus_4_7_reasoning() -> ReasoningCapability {
    ReasoningCapability {
        supported_modes: enum_set!(ReasoningMode::Adaptive),
        supported_efforts: enum_set!(
            ReasoningEffort::Low
                | ReasoningEffort::Medium
                | ReasoningEffort::High
                | ReasoningEffort::XHigh
                | ReasoningEffort::Max
        ),
        manual_budget_range: None,
        conflicts: anthropic_reasoning_conflicts(),
        sampling_params_removed: true,
    }
}

/// `ReasoningCapability` for Claude Sonnet 5. Adaptive-only — manual `budget_tokens` is rejected
/// with a 400 (Sonnet 4.6's transitional manual surface is gone), matching the Opus 4.7/4.8 shape.
/// Sonnet 5 is the first Sonnet-tier model to accept the `xhigh` effort level. Kept separate from
/// `anthropic_opus_4_7_reasoning` so the two models' surfaces can diverge independently.
const fn anthropic_sonnet_5_reasoning() -> ReasoningCapability {
    ReasoningCapability {
        supported_modes: enum_set!(ReasoningMode::Adaptive),
        supported_efforts: enum_set!(
            ReasoningEffort::Low
                | ReasoningEffort::Medium
                | ReasoningEffort::High
                | ReasoningEffort::XHigh
                | ReasoningEffort::Max
        ),
        manual_budget_range: None,
        conflicts: anthropic_reasoning_conflicts(),
        sampling_params_removed: true,
    }
}

/// `ReasoningCapability` for Claude Sonnet 4.6. Adaptive + manual both accepted (manual is
/// deprecated but functional). Effort set {Low, Medium, High, Max} per the adaptive-thinking
/// docs (no `xhigh`). Manual budget bounded by Sonnet 4.6's 64K max output ceiling.
const fn anthropic_sonnet_4_6_reasoning() -> ReasoningCapability {
    ReasoningCapability {
        supported_modes: enum_set!(ReasoningMode::Adaptive | ReasoningMode::Manual),
        supported_efforts: enum_set!(
            ReasoningEffort::Low
                | ReasoningEffort::Medium
                | ReasoningEffort::High
                | ReasoningEffort::Max
        ),
        // Anthropic docs: budget_tokens must be < max_tokens and >= 1024. Sonnet 4.6's max output
        // is 64K, so the absolute upper bound is 64K - 1 token; clamped to 63K leaves headroom.
        manual_budget_range: Some(1024..=63_000),
        conflicts: anthropic_reasoning_conflicts(),
        sampling_params_removed: false,
    }
}

/// `ReasoningCapability` for older Claude 4 manual-only models (Sonnet 4, Opus 4) and Haiku 4.5.
/// Adaptive not supported. Effort set is the three pre-Max levels typically used with manual
/// budget mapping; the caller translates these to a per-provider budget constant
/// before reaching the wire.
const fn anthropic_manual_only_reasoning(max_output_tokens: u32) -> ReasoningCapability {
    let upper = max_output_tokens.saturating_sub(1024);
    ReasoningCapability {
        supported_modes: enum_set!(ReasoningMode::Manual),
        supported_efforts: enum_set!(
            ReasoningEffort::Low | ReasoningEffort::Medium | ReasoningEffort::High
        ),
        manual_budget_range: Some(1024..=upper),
        conflicts: anthropic_reasoning_conflicts(),
        sampling_params_removed: false,
    }
}

static MODEL_CAPABILITIES: LazyLock<HashMap<&'static str, ChatModelCapabilities>> =
    LazyLock::new(|| {
        let mut m = HashMap::new();
        // Claude Sonnet 4 (`claude-sonnet-4-20250514`) and Opus 4 (`claude-opus-4-20250514`) —
        // manual-only thinking surface, 64K max output. 200k context window.
        for id in ["claude-sonnet-4-20250514", "claude-opus-4-20250514"] {
            m.insert(
                id,
                ChatModelCapabilities {
                    context_window: 200_000,
                    supports_streaming: true,
                    supports_json_schema: true,
                    reasoning: Some(anthropic_manual_only_reasoning(64_000)),
                },
            );
        }
        // Claude Sonnet 4.6 — adaptive + manual both supported.
        m.insert(
            "claude-sonnet-4-6",
            ChatModelCapabilities {
                context_window: 200_000,
                supports_streaming: true,
                supports_json_schema: true,
                reasoning: Some(anthropic_sonnet_4_6_reasoning()),
            },
        );
        // Claude Sonnet 5 — adaptive only; manual `budget_tokens` rejected with 400.
        m.insert(
            "claude-sonnet-5",
            ChatModelCapabilities {
                context_window: 200_000,
                supports_streaming: true,
                supports_json_schema: true,
                reasoning: Some(anthropic_sonnet_5_reasoning()),
            },
        );
        // Claude Opus 4.7 — adaptive only.
        m.insert(
            "claude-opus-4-7",
            ChatModelCapabilities {
                context_window: 200_000,
                supports_streaming: true,
                supports_json_schema: true,
                reasoning: Some(anthropic_opus_4_7_reasoning()),
            },
        );
        // Claude Haiku 4.5 — manual-only per Anthropic docs (adaptive not documented for Haiku
        // class). 64K max output.
        m.insert(
            "claude-haiku-4-5-20251001",
            ChatModelCapabilities {
                context_window: 200_000,
                supports_streaming: true,
                supports_json_schema: true,
                reasoning: Some(anthropic_manual_only_reasoning(64_000)),
            },
        );
        m
    });

/// Derive the media-support table for a Claude 4.x model. All Claude 4.x surface the same shape
/// today (image + PDF documents, three source kinds); the `_caps` parameter is reserved for
/// future per-model carve outs without changing call sites.
fn anthropic_media_support(_caps: &ChatModelCapabilities) -> BTreeMap<MediaKind, MediaSupport> {
    let mut m = BTreeMap::new();
    let three_sources =
        enum_set!(SourceKind::Url | SourceKind::InlineBytes | SourceKind::ProviderFile);
    m.insert(
        MediaKind::Image,
        MediaSupport {
            sources: three_sources,
            formats: &["png", "jpeg", "gif", "webp"],
            // Anthropic publishes a 5 MB per-image cap via the API (10 MB on claude.ai). Enforce
            // the API cap locally.
            max_bytes: Some(5 * 1024 * 1024),
            max_count_per_message: None,
        },
    );
    m.insert(
        MediaKind::Document,
        MediaSupport {
            sources: three_sources,
            // The Messages API document block accepts PDF only (plain text is a separate source
            // type we don't model yet).
            formats: &["pdf"],
            max_bytes: Some(32 * 1024 * 1024),
            max_count_per_message: None,
        },
    );
    m
}

// --- list_models serde structs ---

#[derive(Deserialize)]
struct AnthropicListModelsResponse {
    data: Vec<AnthropicListModelEntry>,
}

#[derive(Deserialize)]
struct AnthropicListModelEntry {
    id: String,
    #[serde(default)]
    display_name: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CapabilityError, ContentPart, MediaSource, MediaType, message::Role};

    fn inline_png(data: &[u8]) -> MediaSource {
        MediaSource::InlineBytes {
            mime: MediaType::parse("image/png").unwrap(),
            data: data.to_vec(),
        }
    }

    #[test]
    fn capabilities_returns_image_and_document_for_claude_4() {
        let lm = AnthropicLanguageModel::new(
            AnthropicDeps {
                client: Arc::new(reqwest::Client::new()),
            },
            AnthropicConfig {
                api_key: ApiKey::parse("test").unwrap(),
                base_url: AnthropicConfig::DEFAULT_BASE_URL.to_owned(),
                retry_config: RetryConfig::default(),
            },
        );
        let caps = lm.capabilities(&ModelId::new("claude-sonnet-4-6")).unwrap();
        assert!(caps.media_support.contains_key(&MediaKind::Image));
        assert!(caps.media_support.contains_key(&MediaKind::Document));
        let image = caps.media_support.get(&MediaKind::Image).unwrap();
        assert!(image.sources.contains(SourceKind::Url));
        assert!(image.sources.contains(SourceKind::InlineBytes));
        assert!(image.sources.contains(SourceKind::ProviderFile));
    }

    #[test]
    fn capabilities_returns_none_for_unknown_model() {
        let lm = AnthropicLanguageModel::new(
            AnthropicDeps {
                client: Arc::new(reqwest::Client::new()),
            },
            AnthropicConfig {
                api_key: ApiKey::parse("test").unwrap(),
                base_url: AnthropicConfig::DEFAULT_BASE_URL.to_owned(),
                retry_config: RetryConfig::default(),
            },
        );
        assert!(lm.capabilities(&ModelId::new("nope")).is_none());
    }

    #[test]
    fn validate_request_rejects_audio_for_anthropic() {
        let lm = AnthropicLanguageModel::new(
            AnthropicDeps {
                client: Arc::new(reqwest::Client::new()),
            },
            AnthropicConfig {
                api_key: ApiKey::parse("test").unwrap(),
                base_url: AnthropicConfig::DEFAULT_BASE_URL.to_owned(),
                retry_config: RetryConfig::default(),
            },
        );
        let model = ModelId::new("claude-sonnet-4-6");
        let messages = vec![Message::with_parts(
            Role::User,
            vec![ContentPart::audio(inline_png(b"x"))],
        )];
        let req = GenerateRequest {
            model: &model,
            messages: &messages,
            config: &LanguageModelConfig::default(),
        };
        let err = lm.validate_request(&req).unwrap_err();
        match err {
            CapabilityError::ModalityUnsupported { kind, .. } => {
                assert_eq!(kind, MediaKind::Audio);
            }
            other => panic!("expected ModalityUnsupported, got {other:?}"),
        }
    }
}
