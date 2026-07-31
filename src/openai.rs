// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! OpenAI Chat Completions API implementation of [`LanguageModelProvider`].
//!
//! Also works with any OpenAI-compatible endpoint (vLLM, Azure OpenAI, etc.) via the
//! `base_url` configuration. Ollama is its own provider impl with adjusted defaults; MLX uses
//! this provider directly.
//!
//! Wire-format helpers (request building, response parsing, SSE handling, content-part
//! translation, error mapping) live in [`crate::openai_wire`] and are shared with
//! [`crate::bedrock_mantle::BedrockMantleProvider`]'s Chat Completions surface.

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
    capabilities::{
        MediaKind, MediaSupport, ModelCapabilities, ReasoningCapability, ReasoningMode,
        ReasoningParamConflicts,
    },
    config::ReasoningEffort,
    error::LanguageModelError,
    identifiers::{ApiKey, ModelId},
    media::SourceKind,
    openai_wire::{
        OpenAiResponse, OpenAiStreamOptions, build_request, convert_openai_sse_event,
        map_openai_error, parse_openai_response,
    },
    provider::{ChatModelInfo, GenerateRequest, LanguageModelProvider, ResponseFormatKind},
    response::{LanguageModelResponse, StreamDelta},
    retry::{RetryConfig, with_retry},
    sse::parse_sse_stream,
};

/// OpenAI Chat Completions API language model.
///
/// Calls `POST {base_url}/chat/completions`. Works with OpenAI directly or any OpenAI-compatible
/// endpoint via [`OpenAiConfig::base_url`].
///
/// # Compatible Providers
///
/// | Provider | Base URL | Start command |
/// |----------|----------|---------------|
/// | OpenAI | `https://api.openai.com/v1` (default) | — |
/// | MLX (text) | `http://localhost:8080/v1` | `mlx_lm.server --model <model>` |
/// | MLX (vision) | `http://localhost:8080/v1` | `mlx_vlm.server --model <model>` |
/// | vLLM | `http://<host>/v1` | `vllm serve <model>` |
///
/// The Ollama daemon has its own provider impl ([`crate::OllamaLanguageModel`]) with
/// reasoning-effort defaults that differ from OpenAI's.
pub struct OpenAiLanguageModel {
    client: Arc<reqwest::Client>,
    api_key: String,
    base_url: String,
    retry_config: RetryConfig,
    /// Single-entry TTL cache for [`LanguageModelProvider::list_models`].
    /// `time_to_live = 1h`, `max_capacity = 1`. moka's `try_get_with`
    /// coalesces concurrent misses.
    list_models_cache: Cache<(), Vec<ChatModelInfo>>,
}

/// Configuration for [`OpenAiLanguageModel`].
///
/// One provider instance serves every model the upstream supports — the model identifier arrives
/// in [`GenerateRequest`] per call. Override `base_url` for OpenAI-compatible providers (MLX,
/// vLLM, Azure).
pub struct OpenAiConfig {
    pub api_key: ApiKey,
    pub base_url: String,
    pub retry_config: RetryConfig,
}

impl OpenAiConfig {
    pub const DEFAULT_BASE_URL: &'static str = "https://api.openai.com/v1";
}

/// Injected dependencies for [`OpenAiLanguageModel`].
///
/// The HTTP client is constructed once at the application composition root
/// and cloned into every HTTP-based provider so the
/// macOS reqwest system-proxy trap fires at most once per process and the connection pool is
/// shared.
pub struct OpenAiDeps {
    pub client: Arc<reqwest::Client>,
}

impl OpenAiLanguageModel {
    /// Create a new OpenAI language model.
    #[must_use]
    pub fn new(deps: OpenAiDeps, config: OpenAiConfig) -> Self {
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
        messages: &[crate::message::Message],
        config: &crate::config::LanguageModelConfig,
    ) -> Result<LanguageModelResponse, LanguageModelError> {
        let request_body = build_request(
            model,
            messages,
            config,
            openai_uses_completion_tokens(model),
        );
        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LanguageModelError::provider(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let headers = response.headers().clone();
            let body =
                crate::http_error::read_error_body_or_warn(response, "openai", status.as_u16())
                    .await;
            return Err(map_openai_error(status.as_u16(), &body, &headers));
        }

        let api_response: OpenAiResponse = response
            .json()
            .await
            .map_err(|e| LanguageModelError::provider(format!("failed to parse response: {e}")))?;
        parse_openai_response(api_response)
    }

    /// Shared streaming helper for [`LanguageModelProvider::generate_stream`].
    async fn stream(
        &self,
        model: &str,
        messages: &[crate::message::Message],
        config: &crate::config::LanguageModelConfig,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamDelta, LanguageModelError>> + Send>>,
        LanguageModelError,
    > {
        let mut request_body = build_request(
            model,
            messages,
            config,
            openai_uses_completion_tokens(model),
        );
        request_body.stream = Some(true);
        // Without this flag OpenAI streams every chunk with `usage: null` — the final pre-`[DONE]`
        // chunk only carries usage when the client opts in. See StreamDelta::usage docs for
        // per-provider semantics.
        request_body.stream_options = Some(OpenAiStreamOptions {
            include_usage: true,
        });
        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LanguageModelError::provider(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let headers = response.headers().clone();
            let body =
                crate::http_error::read_error_body_or_warn(response, "openai", status.as_u16())
                    .await;
            return Err(map_openai_error(status.as_u16(), &body, &headers));
        }

        let byte_stream = response.bytes_stream();
        let sse_stream = parse_sse_stream(byte_stream);
        let delta_stream = sse_stream
            .filter_map(|event_result| async move { convert_openai_sse_event(event_result) });
        Ok(Box::pin(delta_stream))
    }

    /// Fetch and merge the upstream model catalog for [`LanguageModelProvider::list_models`].
    ///
    /// Calls `GET {base_url}/models` (auth: `Authorization: Bearer {api_key}`). OpenAI's
    /// `/v1/models` returns IDs only — no context_window or capability flags — so per-ID metadata
    /// comes from the local [`MODEL_CAPABILITIES`] table. IDs missing from the local table fire a
    /// single `tracing::warn!` per cache fill.
    async fn fetch_and_merge_models(&self) -> Result<Vec<ChatModelInfo>, LanguageModelError> {
        let url = format!("{}/models", self.base_url);
        let response = self
            .client
            .get(&url)
            .header("authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .map_err(|e| LanguageModelError::provider(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let headers = response.headers().clone();
            let body =
                crate::http_error::read_error_body_or_warn(response, "openai", status.as_u16())
                    .await;
            return Err(map_openai_error(status.as_u16(), &body, &headers));
        }

        let api_response: OpenAiListModelsResponse = response
            .json()
            .await
            .map_err(|e| LanguageModelError::provider(format!("failed to parse models: {e}")))?;

        let mut out = Vec::with_capacity(api_response.data.len());
        for entry in api_response.data {
            let caps = MODEL_CAPABILITIES.get(entry.id.as_str()).cloned();
            if caps.is_none() {
                tracing::warn!(
                    provider = "openai",
                    model_id = %entry.id,
                    "model returned by upstream but no local capability metadata; \
                     ChatModelInfo will have minimal fields. \
                     Update the local capability table when ready."
                );
            }
            // OpenAI accepts JsonObject everywhere it accepts JsonSchema, so the schema flag
            // implies both. Older models without JsonSchema typically still support JsonObject;
            // keep that conservative until we add a separate flag.
            let mut formats = vec![ResponseFormatKind::Text];
            if caps.as_ref().is_some_and(|c| c.supports_json_schema) {
                formats.push(ResponseFormatKind::JsonObject);
                formats.push(ResponseFormatKind::JsonSchema);
            }
            out.push(ChatModelInfo {
                id: ModelId::new(entry.id),
                display_name: None,
                context_window: caps.as_ref().map(|c| c.context_window),
                supports_streaming: caps.as_ref().is_some_and(|c| c.supports_streaming),
                supported_response_formats: formats,
                media_support: caps.as_ref().map(openai_media_support).unwrap_or_default(),
                reasoning: caps.as_ref().and_then(|c| c.reasoning.clone()),
            });
        }
        Ok(out)
    }
}

#[async_trait]
impl LanguageModelProvider for OpenAiLanguageModel {
    fn name(&self) -> &'static str {
        "openai"
    }

    fn capabilities(&self, model: &ModelId) -> Option<ModelCapabilities> {
        let caps = MODEL_CAPABILITIES.get(model.as_str())?;
        Some(ModelCapabilities {
            model_id: model.as_str().to_owned(),
            media_support: openai_media_support(caps),
            reasoning: caps.reasoning.clone(),
            // Latency-optimized inference is a Bedrock-only tier.
            latency_optimized_supported: false,
            // Our `cache_ttl` knob targets Bedrock/Anthropic; OpenAI caching is automatic, so the
            // extended-TTL tier doesn't apply.
            extended_cache_ttl_supported: false,
        })
    }

    #[tracing::instrument(skip(self), fields(provider = "openai", model_count = tracing::field::Empty), err(Display))]
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
            provider = "openai",
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
            provider = "openai",
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

/// Derive the media-support table for an OpenAI model.
///
/// Every entry in [`MODEL_CAPABILITIES`] today is a vision model (gpt-4o family, gpt-5 family,
/// o-series reasoning), so image + document support is the default. Audio-only models
/// (`gpt-audio*`, `gpt-4o-audio-preview`) are not yet seeded in the capability table — they need
/// a separate `MediaKind::Audio` branch keyed off the model id when they are.
///
/// Source-kind notes:
/// - `Image` accepts `Url` (HTTPS) and `InlineBytes` (transmitted as a `data:` URI on the wire).
///   Chat Completions has no `file_id` primitive for images today — Files API images flow through
///   the Assistants API, not surfaced here.
/// - `Document` accepts `InlineBytes` and `ProviderFile` (Files API `file_id` via the `file`
///   content part).
fn openai_media_support(_caps: &ChatModelCapabilities) -> BTreeMap<MediaKind, MediaSupport> {
    let mut m = BTreeMap::new();
    m.insert(
        MediaKind::Image,
        MediaSupport {
            sources: enum_set!(SourceKind::Url | SourceKind::InlineBytes),
            formats: &["png", "jpeg", "webp", "gif"],
            // 20 MB upper bound per the OpenAI vision docs; enforce locally so callers don't waste
            // a round-trip on oversize base64 payloads.
            max_bytes: Some(20 * 1024 * 1024),
            max_count_per_message: None,
        },
    );
    m.insert(
        MediaKind::Document,
        MediaSupport {
            sources: enum_set!(SourceKind::InlineBytes | SourceKind::ProviderFile),
            // OpenAI's `file` content part accepts a broad set; PDFs are the well-tested format.
            // Subsequent passes can expand this.
            formats: &["pdf"],
            max_bytes: Some(50 * 1024 * 1024),
            max_count_per_message: None,
        },
    );
    m
}

/// Static capability table for OpenAI models.
///
/// `/v1/models` returns IDs only — context_window and capability flags come from this table.
/// New IDs returned upstream that aren't here fire a one-time warning per cache fill, prompting an
/// update.
#[derive(Debug, Clone)]
struct ChatModelCapabilities {
    context_window: u32,
    supports_streaming: bool,
    supports_json_schema: bool,
    /// `Some(_)` when the model accepts the `reasoning_effort` field. OpenAI doesn't expose a
    /// manual-budget knob, so every populated entry sets `supported_modes` to just `Adaptive`.
    reasoning: Option<ReasoningCapability>,
}

/// Sampling conflicts that apply to every OpenAI reasoning model.
///
/// OpenAI reasoning models (o-series + gpt-5 family) hard-reject any `temperature` value other
/// than the default `1.0` while `reasoning_effort` is set to a non-`none` level (live-tested
/// against the API). `top_p` has no documented restriction.
const fn openai_reasoning_conflicts() -> ReasoningParamConflicts {
    ReasoningParamConflicts {
        temperature_forbidden: true,
        top_k_forbidden: false,
        top_p_allowed_range: None,
    }
}

/// `ReasoningCapability` for the gpt-5 family — supports the full 5-value enum
/// (`none|low|medium|high|xhigh`) per OpenAI's published model spec for GPT-5.5 / GPT-5.4 /
/// GPT-5.4-mini.
const fn openai_gpt5_reasoning() -> ReasoningCapability {
    ReasoningCapability {
        supported_modes: enum_set!(ReasoningMode::Adaptive),
        supported_efforts: enum_set!(
            ReasoningEffort::None
                | ReasoningEffort::Low
                | ReasoningEffort::Medium
                | ReasoningEffort::High
                | ReasoningEffort::XHigh
        ),
        manual_budget_range: None,
        conflicts: openai_reasoning_conflicts(),
        sampling_params_removed: false,
    }
}

/// `ReasoningCapability` for the o-series (o1/o3/o4-mini family). Per OpenAI docs, accepts
/// `reasoning_effort: "low"|"medium"|"high"` — no `none` and no `xhigh` on these legacy reasoning
/// models.
const fn openai_o_series_reasoning() -> ReasoningCapability {
    ReasoningCapability {
        supported_modes: enum_set!(ReasoningMode::Adaptive),
        supported_efforts: enum_set!(
            ReasoningEffort::Low | ReasoningEffort::Medium | ReasoningEffort::High
        ),
        manual_budget_range: None,
        conflicts: openai_reasoning_conflicts(),
        sampling_params_removed: false,
    }
}

static MODEL_CAPABILITIES: LazyLock<HashMap<&'static str, ChatModelCapabilities>> =
    LazyLock::new(|| {
        let mut m = HashMap::new();
        // GPT-4o family — 128k context, full streaming + structured outputs. Non-reasoning models:
        // `reasoning_effort` is rejected by the API.
        for id in [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
        ] {
            m.insert(
                id,
                ChatModelCapabilities {
                    context_window: 128_000,
                    supports_streaming: true,
                    supports_json_schema: true,
                    reasoning: None,
                },
            );
        }
        // o-series reasoning models — 128k context, streaming, JsonSchema. Effort enum:
        // {low, medium, high} — the surface published with o1 and unchanged through o4-mini per
        // OpenAI's reasoning guide.
        for id in ["o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini"] {
            m.insert(
                id,
                ChatModelCapabilities {
                    context_window: 128_000,
                    supports_streaming: true,
                    supports_json_schema: true,
                    reasoning: Some(openai_o_series_reasoning()),
                },
            );
        }
        // Legacy GPT-5 IDs preserved from prior workspace state — no reasoning surface declared.
        for id in ["gpt-5", "gpt-5-2025-08-07"] {
            m.insert(
                id,
                ChatModelCapabilities {
                    context_window: 256_000,
                    supports_streaming: true,
                    supports_json_schema: true,
                    reasoning: None,
                },
            );
        }
        // GPT-5.5 / GPT-5.4: per the OpenAI model spec, 1M context window, 128K max output,
        // accepts the full 5-value `reasoning_effort` enum.
        for id in ["gpt-5.5", "gpt-5.4"] {
            m.insert(
                id,
                ChatModelCapabilities {
                    context_window: 1_000_000,
                    supports_streaming: true,
                    supports_json_schema: true,
                    reasoning: Some(openai_gpt5_reasoning()),
                },
            );
        }
        // GPT-5.4-mini: same effort enum, 400K context per the spec.
        m.insert(
            "gpt-5.4-mini",
            ChatModelCapabilities {
                context_window: 400_000,
                supports_streaming: true,
                supports_json_schema: true,
                reasoning: Some(openai_gpt5_reasoning()),
            },
        );
        m
    });

/// Whether the given OpenAI model uses `max_completion_tokens` rather than the legacy
/// `max_tokens`. Driven by the local capability table: any model whose
/// `ChatModelCapabilities.reasoning` is `Some(_)` is on the new field. Unknown models fall back to
/// `max_tokens` for backwards-compat — surfacing a clear API error is preferable to a silent
/// override on an unrecognised model.
fn openai_uses_completion_tokens(model: &str) -> bool {
    MODEL_CAPABILITIES
        .get(model)
        .is_some_and(|c| c.reasoning.is_some())
}

// --- list_models serde structs ---

#[derive(Deserialize)]
struct OpenAiListModelsResponse {
    data: Vec<OpenAiListModelEntry>,
}

#[derive(Deserialize)]
struct OpenAiListModelEntry {
    id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_known_model_returns_image_support() {
        let lm = OpenAiLanguageModel::new(
            OpenAiDeps {
                client: Arc::new(reqwest::Client::new()),
            },
            OpenAiConfig {
                api_key: ApiKey::parse("test").unwrap(),
                base_url: OpenAiConfig::DEFAULT_BASE_URL.to_owned(),
                retry_config: RetryConfig::default(),
            },
        );
        let caps = lm.capabilities(&ModelId::new("gpt-4o")).unwrap();
        let image = caps.media_support.get(&MediaKind::Image).unwrap();
        assert!(image.sources.contains(SourceKind::Url));
        assert!(image.sources.contains(SourceKind::InlineBytes));
        // Chat-completions doesn't surface ProviderFile for images.
        assert!(!image.sources.contains(SourceKind::ProviderFile));
    }

    #[test]
    fn custom_base_url() {
        let lm = OpenAiLanguageModel::new(
            OpenAiDeps {
                client: Arc::new(reqwest::Client::new()),
            },
            OpenAiConfig {
                base_url: "https://my-vllm.example.com/v1".into(),
                api_key: ApiKey::parse("key").unwrap(),
                retry_config: RetryConfig::default(),
            },
        );
        assert_eq!(lm.base_url, "https://my-vllm.example.com/v1");
    }

    #[test]
    fn gpt5_family_uses_completion_tokens() {
        // gpt-5.x is in the cap table with a reasoning entry → uses_completion_tokens=true.
        assert!(openai_uses_completion_tokens("gpt-5.5"));
        assert!(openai_uses_completion_tokens("gpt-5.4"));
        assert!(openai_uses_completion_tokens("gpt-5.4-mini"));
        assert!(openai_uses_completion_tokens("o3-mini"));
        // gpt-4o is non-reasoning → legacy max_tokens.
        assert!(!openai_uses_completion_tokens("gpt-4o"));
        // Unknown model falls back to legacy max_tokens.
        assert!(!openai_uses_completion_tokens("some-future-model"));
    }
}
