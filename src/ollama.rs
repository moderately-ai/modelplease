// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ollama language model implementation.
//!
//! Uses the OpenAI-compatible API but handles Ollama-specific quirks:
//! - `reasoning_effort` is stripped when `response_format` is set (they don't compose in Ollama)
//! - No API key required
//! - Default base URL is `http://localhost:11434/v1`
//!
//! # Examples
//!
//! ```rust
//! use std::sync::Arc;
//! use modelplease::{OllamaConfig, OllamaDeps, OllamaLanguageModel};
//!
//! let deps = || OllamaDeps { client: Arc::new(reqwest::Client::new()) };
//! let lm = OllamaLanguageModel::new(deps(), OllamaConfig::default());
//!
//! // Custom Ollama host
//! let lm = OllamaLanguageModel::new(deps(), OllamaConfig {
//!     base_url: "http://my-gpu-server:11434/v1".into(),
//!     ..OllamaConfig::default()
//! });
//!
//! // The model identifier arrives in `GenerateRequest` per call,
//! // not at construction time.
//! ```

use std::{collections::BTreeMap, pin::Pin, sync::Arc, time::Duration};

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
    config::{LanguageModelConfig, ReasoningConfig, ReasoningEffort, ResponseFormat},
    error::LanguageModelError,
    identifiers::ModelId,
    media::SourceKind,
    message::Message,
    openai_wire::{
        OpenAiRequest, OpenAiStreamOptions, build_openai_messages, convert_openai_sse_event,
        map_openai_error, parse_openai_response,
    },
    provider::{ChatModelInfo, GenerateRequest, LanguageModelProvider, ResponseFormatKind},
    response::{LanguageModelResponse, StreamDelta},
    retry::{RetryConfig, with_retry},
    sse::parse_sse_stream,
};

/// Ollama language model.
///
/// Calls `POST {base_url}/chat/completions` using the OpenAI-compatible API
/// that Ollama exposes. Handles Ollama-specific behavior:
///
/// - No API key required (Ollama is local)
/// - `reasoning_effort` is automatically stripped when `response_format` is set, because Ollama
///   silently ignores `response_format` when `reasoning_effort` is present
/// - Default base URL is `http://localhost:11434/v1`
pub struct OllamaLanguageModel {
    client: Arc<reqwest::Client>,
    base_url: String,
    retry_config: RetryConfig,
    /// Single-entry TTL cache for [`LanguageModelProvider::list_models`].
    /// `time_to_live = 1h`, `max_capacity = 1`.
    list_models_cache: Cache<(), Vec<ChatModelInfo>>,
}

/// Configuration for [`OllamaLanguageModel`].
///
/// Ollama is local — no API key required. `base_url` defaults to
/// `http://localhost:11434/v1`; `retry_config` defaults via
/// [`RetryConfig::default`]. The model identifier arrives in
/// [`GenerateRequest`] per call.
pub struct OllamaConfig {
    pub base_url: String,
    pub retry_config: RetryConfig,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: Self::DEFAULT_BASE_URL.to_owned(),
            retry_config: RetryConfig::default(),
        }
    }
}

impl OllamaConfig {
    pub const DEFAULT_BASE_URL: &'static str = "http://localhost:11434/v1";
}

/// Injected dependencies for [`OllamaLanguageModel`]. The HTTP client
/// is constructed once at the application composition root and shared
/// across HTTP-based providers.
pub struct OllamaDeps {
    pub client: Arc<reqwest::Client>,
}

impl OllamaLanguageModel {
    /// Create a new Ollama language model.
    ///
    /// No API key is needed — Ollama runs locally.
    #[must_use]
    pub fn new(deps: OllamaDeps, config: OllamaConfig) -> Self {
        Self {
            client: deps.client,
            base_url: config.base_url,
            retry_config: config.retry_config,
            list_models_cache: Cache::builder()
                .time_to_live(Duration::from_secs(3600))
                .max_capacity(1)
                .build(),
        }
    }

    /// Build the request body, handling the `reasoning_effort` + `response_format` conflict.
    #[tracing::instrument(skip_all, fields(model, msg_count = messages.len()), level = "trace")]
    fn build_request(
        model: &str,
        messages: &[Message],
        config: &LanguageModelConfig,
    ) -> OpenAiRequest {
        let api_messages = build_openai_messages(messages);

        // Ollama silently ignores response_format when reasoning_effort is set.
        // Strip reasoning_effort when a structured format is requested.
        let reasoning_effort = match &config.response_format {
            ResponseFormat::Text => ollama_reasoning_effort_wire(config.reasoning.as_ref()),
            _ => None,
        };

        let response_format = match &config.response_format {
            ResponseFormat::Text => None,
            ResponseFormat::JsonObject => Some(serde_json::json!({"type": "json_object"})),
            ResponseFormat::JsonSchema {
                name,
                schema,
                strict,
            } => Some(serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "schema": schema,
                    "strict": strict,
                }
            })),
        };

        OpenAiRequest {
            model: model.to_owned(),
            messages: api_messages,
            temperature: config.temperature,
            // Ollama's OpenAI-compat layer accepts the legacy
            // `max_tokens` field across the board — the
            // `max_completion_tokens` knob is OpenAI-cloud-only and not
            // wired into Ollama's local serving stack.
            max_tokens: config.max_tokens,
            max_completion_tokens: None,
            top_p: config.top_p,
            stop: if config.stop.is_empty() {
                None
            } else {
                Some(config.stop.clone())
            },
            reasoning_effort,
            stream: None,
            stream_options: None,
            response_format,
        }
    }

    /// Send the request and parse the response.
    async fn execute(
        &self,
        model: &str,
        messages: &[Message],
        config: &LanguageModelConfig,
    ) -> Result<LanguageModelResponse, LanguageModelError> {
        let request_body = Self::build_request(model, messages, config);
        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LanguageModelError::provider(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let headers = response.headers().clone();
            let body =
                crate::http_error::read_error_body_or_warn(response, "ollama", status.as_u16())
                    .await;
            return Err(map_openai_error(status.as_u16(), &body, &headers));
        }

        let api_response = response
            .json()
            .await
            .map_err(|e| LanguageModelError::provider(format!("failed to parse response: {e}")))?;
        parse_openai_response(api_response)
    }
}

impl OllamaLanguageModel {
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
        let mut request_body = Self::build_request(model, messages, config);
        request_body.stream = Some(true);
        // Match OpenAI: opt into a final usage chunk so StreamDelta::usage
        // populates on the last pre-`[DONE]` event rather than always None.
        request_body.stream_options = Some(OpenAiStreamOptions {
            include_usage: true,
        });
        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LanguageModelError::provider(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let headers = response.headers().clone();
            let body =
                crate::http_error::read_error_body_or_warn(response, "ollama", status.as_u16())
                    .await;
            return Err(map_openai_error(status.as_u16(), &body, &headers));
        }

        let byte_stream = response.bytes_stream();
        let sse_stream = parse_sse_stream(byte_stream);

        // Ollama's `/v1` proxy emits the same SSE shape as OpenAI's Chat
        // Completions API — reuse the OpenAI event mapper.
        let delta_stream = sse_stream
            .filter_map(|event_result| async move { convert_openai_sse_event(event_result) });

        Ok(Box::pin(delta_stream))
    }

    /// Compute Ollama's native API root from `base_url`.
    ///
    /// `base_url` points at the OpenAI-compatible shim (typically
    /// `http://localhost:11434/v1`); Ollama-native endpoints live one
    /// level up (`http://localhost:11434/api/...`). Strips a trailing
    /// `/v1` if present.
    fn ollama_api_root(&self) -> &str {
        self.base_url.strip_suffix("/v1").unwrap_or(&self.base_url)
    }

    /// Fetch the locally-installed model list from Ollama's native
    /// `/api/tags` endpoint.
    ///
    /// `/api/tags` returns whatever the operator has pulled — there's
    /// no canonical upstream catalog to compare against, so no
    /// warn-on-miss table. Ollama supports streaming + JSON-schema for
    /// any model that accepts it (configurable per-model via the
    /// modelfile), so the static-table approach used by Anthropic /
    /// OpenAI doesn't fit; we report `supports_streaming: true` and
    /// `supports_json_schema: true` uniformly. `context_window` would
    /// require an N+1 sweep of `/api/show` and isn't surfaced today.
    async fn fetch_and_merge_models(&self) -> Result<Vec<ChatModelInfo>, LanguageModelError> {
        let url = format!("{}/api/tags", self.ollama_api_root());
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| LanguageModelError::provider(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let headers = response.headers().clone();
            let body =
                crate::http_error::read_error_body_or_warn(response, "ollama", status.as_u16())
                    .await;
            return Err(map_openai_error(status.as_u16(), &body, &headers));
        }

        let api_response: OllamaTagsResponse = response
            .json()
            .await
            .map_err(|e| LanguageModelError::provider(format!("failed to parse tags: {e}")))?;

        Ok(api_response
            .models
            .into_iter()
            .map(|m| {
                let media_support = ollama_media_support_for(m.name.as_str());
                let reasoning = ollama_reasoning_for(m.name.as_str());
                ChatModelInfo {
                    id: ModelId::new(m.name),
                    display_name: None,
                    context_window: None,
                    supports_streaming: true,
                    supported_response_formats: vec![
                        ResponseFormatKind::Text,
                        ResponseFormatKind::JsonObject,
                        ResponseFormatKind::JsonSchema,
                    ],
                    media_support,
                    reasoning,
                }
            })
            .collect())
    }
}

#[async_trait]
impl LanguageModelProvider for OllamaLanguageModel {
    fn name(&self) -> &'static str {
        "ollama"
    }

    fn capabilities(&self, model: &ModelId) -> Option<ModelCapabilities> {
        // Ollama does not publish per-model capability metadata via
        // `/api/tags`; we lean on the model name prefix to classify
        // vision-capable variants and thinking variants. An unknown
        // model returns Some(empty) so the default `validate_request`
        // impl produces `ModalityUnsupported` rather than `UnknownModel`
        // (which would be misleading — we *know* the model, just not
        // its modality).
        Some(ModelCapabilities {
            model_id: model.as_str().to_owned(),
            media_support: ollama_media_support_for(model.as_str()),
            reasoning: ollama_reasoning_for(model.as_str()),
            // Latency-optimized inference is a Bedrock-only tier.
            latency_optimized_supported: false,
            // Extended-TTL prompt caching is a Bedrock/Anthropic tier.
            extended_cache_ttl_supported: false,
        })
    }

    #[tracing::instrument(skip(self), fields(provider = "ollama", model_count = tracing::field::Empty), err(Display))]
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
            provider = "ollama",
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
            provider = "ollama",
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

// --- /api/tags response serde structs ---

#[derive(Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaTag>,
}

#[derive(Deserialize)]
struct OllamaTag {
    name: String,
}

/// Derive the media-support table for an Ollama model name.
///
/// Ollama's `/api/tags` exposes only a list of names — no per-model
/// capability metadata — so the classification is by name prefix.
/// Vision-capable families (llava, bakllava, llama3.2-vision, moondream,
/// qwen-vl, Gemma 3 / Gemma 4 multimodal, etc.) accept `InlineBytes`
/// images only; Url is excluded because the Ollama daemon doesn't fetch
/// external URLs. Everything else returns an empty map (text-only).
///
/// Future improvement: query `/api/show` to read the model's GGUF
/// metadata directly (`<family>.vision.*` / `<family>.audio.*` keys)
/// instead of maintaining a static prefix allow-list. That requires
/// caching at `list_models` time since [`Self::capabilities`] is sync;
/// tracked separately.
fn ollama_media_support_for(name: &str) -> BTreeMap<MediaKind, MediaSupport> {
    let mut m = BTreeMap::new();
    // Match a name like "llava:7b-q4" by checking the leading token
    // before the colon (Ollama tags use `<name>:<size-or-quant>`).
    let head = name.split(':').next().unwrap_or(name).to_ascii_lowercase();
    let is_vision = head.starts_with("llava")
        || head.starts_with("bakllava")
        || head.starts_with("moondream")
        || head.contains("vision")
        || head.starts_with("llama3.2-vision")
        || head.starts_with("llama4")
        || head.starts_with("qwen-vl")
        || head.starts_with("qwen2-vl")
        || head.starts_with("qwen2.5-vl")
        || head.starts_with("qwen3-vl")
        || head.starts_with("minicpm-v")
        // Gemma 3 (4B / 12B / 27B) and Gemma 4 are natively multimodal —
        // GGUF metadata exposes `gemma{3,4}.vision.*` keys. The 1B
        // Gemma 3 / Gemma 4 variants are text-only; surface them
        // honestly by excluding the `:1b` tag.
        || (head.starts_with("gemma3") && !name.ends_with(":1b"))
        || (head.starts_with("gemma4") && !name.ends_with(":1b"));
    if is_vision {
        m.insert(
            MediaKind::Image,
            MediaSupport {
                sources: enum_set!(SourceKind::InlineBytes),
                formats: &["png", "jpeg"],
                max_bytes: Some(20 * 1024 * 1024),
                max_count_per_message: None,
            },
        );
    }
    m
}

/// Classify whether the given Ollama model name is a thinking variant,
/// and if so what `ReasoningConfig` it accepts.
///
/// Ollama's OpenAI-compat layer accepts `reasoning_effort` with
/// `{none, low, medium, high}` (4-value enum — no xhigh/max) per the
/// OpenAI-compatibility docs reviewed during planning. The known
/// thinking-capable families today are Qwen3 (excluding instruct/vl
/// variants), DeepSeek-R1, and OpenAI's gpt-oss family. Non-thinking
/// models return `None` and any non-`Off` reasoning config fails
/// validation upstream of the wire.
fn ollama_reasoning_for(name: &str) -> Option<ReasoningCapability> {
    let head = name.split(':').next().unwrap_or(name).to_ascii_lowercase();
    let is_thinking = head.starts_with("deepseek-r1")
        || head.starts_with("gpt-oss")
        // Qwen3 reasoning models — exclude vl (vision) and -instruct
        // variants which Alibaba ships without the thinking head.
        || (head.starts_with("qwen3")
            && !head.contains("-vl")
            && !head.contains("-instruct"));
    if !is_thinking {
        return None;
    }
    Some(ReasoningCapability {
        supported_modes: enum_set!(ReasoningMode::Adaptive),
        supported_efforts: enum_set!(
            ReasoningEffort::None
                | ReasoningEffort::Low
                | ReasoningEffort::Medium
                | ReasoningEffort::High
        ),
        manual_budget_range: None,
        conflicts: ReasoningParamConflicts::default(),
        sampling_params_removed: false,
    })
}

/// Resolve the on-the-wire `reasoning_effort` string for Ollama.
///
/// Same shape as the OpenAI helper but restricted to the
/// {none|low|medium|high} subset; manual mode is rejected upstream by
/// the capability validation layer (Ollama doesn't expose a budget knob).
fn ollama_reasoning_effort_wire(reasoning: Option<&ReasoningConfig>) -> Option<&'static str> {
    match reasoning {
        None => None,
        Some(ReasoningConfig::Off) => Some("none"),
        Some(ReasoningConfig::Adaptive { effort }) => Some(effort.as_str()),
        Some(ReasoningConfig::Manual { .. }) => {
            debug_assert!(
                false,
                "Ollama does not support manual-budget reasoning mode; \
                 capability validation should have rejected this upstream"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_deps() -> OllamaDeps {
        OllamaDeps {
            client: Arc::new(reqwest::Client::new()),
        }
    }

    #[test]
    fn default_base_url() {
        let lm = OllamaLanguageModel::new(test_deps(), OllamaConfig::default());
        assert_eq!(lm.base_url, "http://localhost:11434/v1");
    }

    #[test]
    fn custom_base_url() {
        let lm = OllamaLanguageModel::new(
            test_deps(),
            OllamaConfig {
                base_url: "http://gpu-server:11434/v1".into(),
                ..OllamaConfig::default()
            },
        );
        assert_eq!(lm.base_url, "http://gpu-server:11434/v1");
    }

    #[test]
    fn no_api_key_in_constructor() {
        // Just verifying the API — no api_key parameter
        let _lm = OllamaLanguageModel::new(test_deps(), OllamaConfig::default());
    }

    #[test]
    fn reasoning_effort_stripped_for_json_object() {
        let config = LanguageModelConfig {
            reasoning: Some(ReasoningConfig::Adaptive {
                effort: ReasoningEffort::High,
            }),
            response_format: ResponseFormat::JsonObject,
            ..Default::default()
        };
        let req =
            OllamaLanguageModel::build_request("test-model", &[Message::user("test")], &config);
        assert!(req.reasoning_effort.is_none());
        assert!(req.response_format.is_some());
    }

    #[test]
    fn reasoning_effort_stripped_for_json_schema() {
        let config = LanguageModelConfig {
            reasoning: Some(ReasoningConfig::Adaptive {
                effort: ReasoningEffort::High,
            }),
            response_format: ResponseFormat::JsonSchema {
                name: "test".into(),
                schema: serde_json::json!({"type": "object"}),
                strict: true,
            },
            ..Default::default()
        };
        let req =
            OllamaLanguageModel::build_request("test-model", &[Message::user("test")], &config);
        assert!(req.reasoning_effort.is_none());
        assert!(req.response_format.is_some());
    }

    #[test]
    fn reasoning_effort_preserved_for_text() {
        let config = LanguageModelConfig {
            reasoning: Some(ReasoningConfig::Adaptive {
                effort: ReasoningEffort::High,
            }),
            ..Default::default()
        };
        let req =
            OllamaLanguageModel::build_request("test-model", &[Message::user("test")], &config);
        assert_eq!(req.reasoning_effort, Some("high"));
        assert!(req.response_format.is_none());
    }

    #[test]
    fn reasoning_off_maps_to_none_wire_value() {
        let config = LanguageModelConfig {
            reasoning: Some(ReasoningConfig::Off),
            ..Default::default()
        };
        let req =
            OllamaLanguageModel::build_request("test-model", &[Message::user("test")], &config);
        assert_eq!(req.reasoning_effort, Some("none"));
    }

    #[test]
    fn ollama_reasoning_for_classifies_thinking_models() {
        assert!(ollama_reasoning_for("qwen3:8b").is_some());
        assert!(ollama_reasoning_for("deepseek-r1:7b").is_some());
        assert!(ollama_reasoning_for("gpt-oss-20b").is_some());
        // Vision and instruct variants are not thinking models.
        assert!(ollama_reasoning_for("qwen3-vl:8b").is_none());
        assert!(ollama_reasoning_for("qwen3-instruct:8b").is_none());
        // Non-Qwen3/DeepSeek/gpt-oss families return None.
        assert!(ollama_reasoning_for("llava:7b").is_none());
        assert!(ollama_reasoning_for("llama3.2:8b").is_none());
    }

    #[test]
    fn no_auth_header_in_request() {
        // Ollama doesn't need auth — verify no api_key field exists
        let config = LanguageModelConfig::default();
        let req =
            OllamaLanguageModel::build_request("test-model", &[Message::user("test")], &config);
        // The request struct has no api_key field — auth is handled at HTTP level
        let json = serde_json::to_value(&req).unwrap();
        assert!(!json.as_object().unwrap().contains_key("api_key"));
    }

    #[test]
    fn build_request_stream_options_serialize_when_set() {
        // Ollama's `/v1` proxy honours OpenAI's stream_options.include_usage
        // — assert the field shape since OllamaLanguageModel reuses the
        // OpenAI request DTO.
        let config = LanguageModelConfig::default();
        let mut req =
            OllamaLanguageModel::build_request("test-model", &[Message::user("hi")], &config);
        req.stream = Some(true);
        req.stream_options = Some(OpenAiStreamOptions {
            include_usage: true,
        });

        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["stream"], true);
        assert_eq!(json["stream_options"]["include_usage"], true);
    }

    // ----- Media capability classification by name -----

    #[test]
    fn ollama_classifies_llava_as_vision_capable() {
        let m = ollama_media_support_for("llava:7b");
        let image = m.get(&crate::MediaKind::Image).unwrap();
        assert!(image.sources.contains(SourceKind::InlineBytes));
        // Ollama daemon can't fetch URLs; the capability table excludes
        // SourceKind::Url so validate_request rejects URL-sourced images
        // before they reach the wire.
        assert!(!image.sources.contains(SourceKind::Url));
    }

    #[test]
    fn ollama_classifies_llama_3_2_vision_as_vision_capable() {
        assert!(
            ollama_media_support_for("llama3.2-vision:11b").contains_key(&crate::MediaKind::Image)
        );
    }

    #[test]
    fn ollama_text_only_model_has_empty_support() {
        assert!(ollama_media_support_for("qwen3:7b").is_empty());
        assert!(ollama_media_support_for("phi3:mini").is_empty());
        assert!(ollama_media_support_for("mistral:7b").is_empty());
    }

    #[test]
    fn ollama_classifies_gemma3_4b_plus_as_vision_capable() {
        // Gemma 3 / Gemma 4 GGUF metadata exposes `<family>.vision.*`
        // keys for the 4B+ variants — surface them as image-capable so
        // multimodal callers don't get a ModalityUnsupported rejection.
        assert!(ollama_media_support_for("gemma3:4b").contains_key(&crate::MediaKind::Image));
        assert!(ollama_media_support_for("gemma3:27b").contains_key(&crate::MediaKind::Image));
        assert!(ollama_media_support_for("gemma4:latest").contains_key(&crate::MediaKind::Image));
    }

    #[test]
    fn ollama_classifies_gemma_1b_as_text_only() {
        // The 1B Gemma 3 / Gemma 4 variants ship without the vision
        // tower; honesty-first classification keeps validate_request
        // rejecting image inputs against them.
        assert!(ollama_media_support_for("gemma3:1b").is_empty());
        assert!(ollama_media_support_for("gemma4:1b").is_empty());
    }

    #[test]
    fn ollama_capabilities_returns_some_empty_for_unknown_model() {
        let lm = OllamaLanguageModel::new(test_deps(), OllamaConfig::default());
        // Some(empty) means "known provider, model accepts no media".
        let caps = lm.capabilities(&ModelId::new("phi3:mini")).unwrap();
        assert!(caps.media_support.is_empty());
    }

    #[test]
    fn ollama_validate_request_rejects_image_url() {
        use crate::{CapabilityError, ContentPart, HttpsUrl, MediaSource, Role};
        let lm = OllamaLanguageModel::new(test_deps(), OllamaConfig::default());
        let model = ModelId::new("llava:7b");
        let messages = vec![Message::with_parts(
            Role::User,
            vec![ContentPart::image(MediaSource::Url {
                url: HttpsUrl::parse("https://example.com/x.png").unwrap(),
            })],
        )];
        let req = GenerateRequest {
            model: &model,
            messages: &messages,
            config: &LanguageModelConfig::default(),
        };
        let err = lm.validate_request(&req).unwrap_err();
        assert!(matches!(
            err,
            CapabilityError::SourceKindUnsupported {
                attempted: SourceKind::Url,
                ..
            }
        ));
    }
}
