// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! OpenAI Chat Completions wire layer — DTOs, request builders, response parsers, SSE handling,
//! error mapping. Shared between two providers that speak the same wire format:
//!
//! - [`crate::openai::OpenAiLanguageModel`] (OpenAI direct, plus Ollama / MLX / vLLM via
//!   `with_base_url`).
//! - [`crate::bedrock_mantle::BedrockMantleProvider`]'s Chat Completions surface
//!   (`{base}/v1/chat/completions` against `bedrock-mantle.{region}.api.aws`).
//!
//! Everything in this module is `pub(crate)`. Cross-crate consumers continue to use
//! `OpenAiLanguageModel` / `BedrockMantleProvider` as their entry points — the wire layer is an
//! internal sharing seam, not a public API.

use base64::Engine;
use serde::{Deserialize, Serialize};

#[cfg(any(feature = "openai", feature = "bedrock-mantle", test))]
use crate::config::{LanguageModelConfig, ReasoningConfig, ResponseFormat};
use crate::{
    error::LanguageModelError,
    media::MediaSource,
    message::{ContentPart, Message, Role},
    response::{LanguageModelResponse, StreamDelta, Usage},
    sse::SseEvent,
};

/// Build the OpenAI Chat Completions request body from messages + config.
///
/// `uses_completion_tokens` lets each caller pick the correct `max_tokens` field for the target
/// model — reasoning models (o-series, gpt-5 family, plus any future Mantle-hosted variants
/// declared in their respective cap tables) reject the legacy `max_tokens` field and require
/// `max_completion_tokens` instead. The caller consults its own per-model table and passes the
/// resolved boolean here; this function stays oblivious to the catalog.
#[tracing::instrument(skip_all, fields(model, msg_count = messages.len()), level = "trace")]
#[cfg(any(feature = "openai", feature = "bedrock-mantle", test))]
pub(crate) fn build_request(
    model: &str,
    messages: &[Message],
    config: &LanguageModelConfig,
    uses_completion_tokens: bool,
) -> OpenAiRequest {
    let api_messages = build_openai_messages(messages);
    let (max_tokens, max_completion_tokens) = if uses_completion_tokens {
        (None, config.max_tokens)
    } else {
        (config.max_tokens, None)
    };
    OpenAiRequest {
        model: model.to_owned(),
        messages: api_messages,
        temperature: config.temperature,
        max_tokens,
        max_completion_tokens,
        top_p: config.top_p,
        stop: if config.stop.is_empty() {
            None
        } else {
            Some(config.stop.clone())
        },
        reasoning_effort: openai_reasoning_effort_wire(config.reasoning.as_ref()),
        stream: None,
        stream_options: None,
        response_format: match &config.response_format {
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
        },
    }
}

/// Pull the assistant message out of an OpenAI Chat Completions response and project it into our
/// internal [`LanguageModelResponse`]. Missing or empty content maps to
/// [`LanguageModelError::EmptyResponse`].
pub(crate) fn parse_openai_response(
    api_response: OpenAiResponse,
) -> Result<LanguageModelResponse, LanguageModelError> {
    let content = api_response
        .choices
        .first()
        .and_then(|c| c.message.content.clone())
        .ok_or(LanguageModelError::EmptyResponse)?;
    if content.is_empty() {
        return Err(LanguageModelError::EmptyResponse);
    }
    let stop_reason = api_response
        .choices
        .first()
        .and_then(|c| c.finish_reason.as_deref())
        .map(openai_stop_reason);
    Ok(LanguageModelResponse {
        content,
        thinking: None,
        usage: api_response.usage.map(|u| openai_usage(&u)),
        model: Some(api_response.model),
        stop_reason,
    })
}

fn openai_stop_reason(raw: &str) -> crate::StopReason {
    match raw {
        "stop" => crate::StopReason::EndTurn,
        "length" => crate::StopReason::MaxTokens,
        "tool_calls" | "function_call" => crate::StopReason::ToolUse,
        "content_filter" => crate::StopReason::ContentFilter,
        other => crate::StopReason::Other(other.to_owned()),
    }
}

/// Map OpenAI usage into our [`Usage`]. OpenAI's `prompt_tokens` *includes* the cached tokens,
/// whereas our `Usage` invariant (shared with Anthropic/Bedrock) is that `input_tokens` is the
/// *uncached* remainder and the cache counts are disjoint. Subtract `cached_tokens` so the
/// invariant holds and the prompt-total accounting (`input + cache_read + cache_creation`)
/// doesn't double-count. OpenAI has no separate cache-write notion, so `cache_creation` stays `0`.
fn openai_usage(u: &OpenAiUsage) -> Usage {
    let cached = u
        .prompt_tokens_details
        .as_ref()
        .map_or(0, |d| d.cached_tokens);
    Usage {
        input_tokens: u.prompt_tokens.saturating_sub(cached),
        output_tokens: u.completion_tokens,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: cached,
    }
}

/// Resolve the on-the-wire `reasoning_effort` string for a given [`ReasoningConfig`].
///
/// - `None` (caller didn't ask) ⇒ omit the field; upstream uses its default behaviour.
/// - `Some(Off)` ⇒ explicit `"none"` — OpenAI accepts `none` as a value that genuinely disables
///   reasoning on a reasoning-capable model.
/// - `Some(Adaptive { effort })` ⇒ the enum's wire value.
/// - `Some(Manual { .. })` ⇒ unreachable in practice (capability validation rejects `Manual` against
///   any OpenAI-shape model before we get here). Defensive fallback: omit the field.
#[cfg(any(feature = "openai", feature = "bedrock-mantle", test))]
fn openai_reasoning_effort_wire(reasoning: Option<&ReasoningConfig>) -> Option<&'static str> {
    match reasoning {
        None => None,
        Some(ReasoningConfig::Off) => Some("none"),
        Some(ReasoningConfig::Adaptive { effort }) => Some(effort.as_str()),
        Some(ReasoningConfig::Manual { .. }) => {
            debug_assert!(
                false,
                "OpenAI-shape models do not support manual-budget reasoning mode; \
                 capability validation should have rejected this upstream"
            );
            None
        }
    }
}

/// Translate one OpenAI Chat Completions SSE event into our outgoing [`StreamDelta`].
/// Returns `None` for keep-alive deltas (empty content with no usage). Reused by every provider
/// that speaks OpenAI Chat Completions: OpenAI direct, Ollama, vLLM/MLX, Bedrock Mantle.
pub(crate) fn convert_openai_sse_event(
    event_result: Result<SseEvent, LanguageModelError>,
) -> Option<Result<StreamDelta, LanguageModelError>> {
    let event = match event_result {
        Err(e) => return Some(Err(e)),
        Ok(event) => event,
    };
    if event.data == "[DONE]" {
        return Some(Ok(StreamDelta {
            content: String::new(),
            thinking: None,
            usage: None,
            model: None,
            stop_reason: None,
            is_final: true,
        }));
    }
    let chunk: OpenAiStreamChunk = match serde_json::from_str(&event.data) {
        Ok(c) => c,
        Err(e) => {
            return Some(Err(LanguageModelError::provider(format!(
                "failed to parse stream chunk: {e}"
            ))));
        }
    };
    let first = chunk.choices.first();
    let content = first
        .and_then(|c| c.delta.content.clone())
        .unwrap_or_default();
    let stop_reason = first
        .and_then(|c| c.finish_reason.as_deref())
        .map(openai_stop_reason);
    let usage = chunk.usage.map(|u| openai_usage(&u));
    if content.is_empty() && usage.is_none() && stop_reason.is_none() {
        return None;
    }
    Some(Ok(StreamDelta {
        content,
        thinking: None,
        usage,
        model: Some(chunk.model),
        stop_reason,
        is_final: false,
    }))
}

/// Map HTTP status + error body + response headers to a `LanguageModelError`.
/// On 429 the `Retry-After` header (delta-seconds form) is forwarded into
/// `RateLimited::retry_after` so the retry loop can honour it.
#[cfg(any(feature = "openai", feature = "ollama", test))]
pub(crate) fn map_openai_error(
    status: u16,
    body: &str,
    headers: &reqwest::header::HeaderMap,
) -> LanguageModelError {
    let message = serde_json::from_str::<OpenAiErrorResponse>(body)
        .map_or_else(|_| body.to_owned(), |e| e.error.message);
    match status {
        429 => match parse_retry_after(headers) {
            Some(d) => LanguageModelError::rate_limited_after(message, d),
            None => LanguageModelError::rate_limited(message),
        },
        401 | 403 => LanguageModelError::authentication(message),
        _ => LanguageModelError::provider(format!("HTTP {status}: {message}")),
    }
}

/// Parse the `Retry-After` header as integer delta-seconds (RFC 7231 § 7.1.3). HTTP-date form is
/// intentionally not supported — OpenAI emits delta-seconds in practice, and an unparseable value
/// falls back to the existing exponential backoff in `with_retry`.
#[cfg(any(feature = "openai", feature = "ollama", test))]
fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<std::time::Duration> {
    headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(std::time::Duration::from_secs)
}

/// Translate one [`ContentPart`] into the OpenAI Chat Completions content-part shape.
///
/// `validate_request` runs before this is called and rejects parts the chosen model can't serve.
/// Branches cover every combination the capability table actually advertises (Text, Image with
/// Url / InlineBytes, Document with InlineBytes / ProviderFile, Audio with InlineBytes).
pub(crate) fn translate_part_for_openai(part: &ContentPart) -> Option<OpenAiContentPart> {
    match part {
        ContentPart::Text { text } => Some(OpenAiContentPart::Text { text: text.clone() }),
        ContentPart::Image { source } => match source {
            MediaSource::Url { url } => Some(OpenAiContentPart::ImageUrl {
                image_url: OpenAiImageUrl {
                    url: url.as_str().to_owned(),
                },
            }),
            MediaSource::InlineBytes { mime, data } => {
                let b64 = base64::engine::general_purpose::STANDARD.encode(data);
                Some(OpenAiContentPart::ImageUrl {
                    image_url: OpenAiImageUrl {
                        url: format!("data:{};base64,{b64}", mime.as_str()),
                    },
                })
            }
            other => {
                tracing::warn!(
                    provider = "openai-wire",
                    source_kind = ?other.kind(),
                    "validate_request should have rejected this image source kind",
                );
                None
            }
        },
        ContentPart::Document { source, name } => match source {
            MediaSource::InlineBytes { mime, data } => {
                let b64 = base64::engine::general_purpose::STANDARD.encode(data);
                Some(OpenAiContentPart::File {
                    file: OpenAiFile {
                        file_id: None,
                        file_data: Some(format!("data:{};base64,{b64}", mime.as_str())),
                        filename: name.clone(),
                    },
                })
            }
            MediaSource::ProviderFile { file_id } => Some(OpenAiContentPart::File {
                file: OpenAiFile {
                    file_id: Some(file_id.as_str().to_owned()),
                    file_data: None,
                    filename: name.clone(),
                },
            }),
            other => {
                tracing::warn!(
                    provider = "openai-wire",
                    source_kind = ?other.kind(),
                    "validate_request should have rejected this document source kind",
                );
                None
            }
        },
        ContentPart::Audio {
            source: MediaSource::InlineBytes { mime, data },
        } => {
            let b64 = base64::engine::general_purpose::STANDARD.encode(data);
            Some(OpenAiContentPart::InputAudio {
                input_audio: OpenAiAudioPayload {
                    data: b64,
                    format: mime.subtype().to_owned(),
                },
            })
        }
        ContentPart::Audio { .. } | ContentPart::Video { .. } => {
            tracing::warn!(
                provider = "openai-wire",
                kind = %part.media_kind().map_or("?", crate::capabilities::MediaKind::label),
                "validate_request should have rejected this modality / source combo",
            );
            None
        }
        // OpenAI caches automatically server-side — explicit cache markers are dropped before this
        // in `build_openai_messages` so they don't affect the request shape.
        ContentPart::CacheBreakpoint => None,
    }
}

/// Convert `Message` slice to OpenAI-format message structs.
pub(crate) fn build_openai_messages(messages: &[Message]) -> Vec<OpenAiMessage> {
    messages
        .iter()
        .map(|msg| {
            let role = match msg.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            // Cache markers are not part of OpenAI's wire shape; exclude them so a marker never
            // flips a plain text message into the multipart array form.
            let parts: Vec<&ContentPart> = msg
                .content
                .iter()
                .filter(|p| !p.is_cache_breakpoint())
                .collect();
            let has_non_text = parts.iter().any(|p| !matches!(p, ContentPart::Text { .. }));
            let content = if has_non_text || parts.len() > 1 {
                OpenAiContent::Parts(
                    parts
                        .into_iter()
                        .filter_map(translate_part_for_openai)
                        .collect(),
                )
            } else {
                OpenAiContent::String(msg.text())
            };
            OpenAiMessage {
                role: role.to_owned(),
                content,
            }
        })
        .collect()
}

// --- Request serde structs ---

#[derive(Serialize)]
pub(crate) struct OpenAiRequest {
    pub(crate) model: String,
    pub(crate) messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f64>,
    /// Used on non-reasoning chat models (gpt-4o family). Reasoning models reject this field;
    /// populate `max_completion_tokens` instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_tokens: Option<u32>,
    /// Used on OpenAI reasoning models (o-series, gpt-5 family) — they hard-reject `max_tokens`
    /// with a 400 and expect this field which covers both visible output and internal reasoning
    /// tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning_effort: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stream_options: Option<OpenAiStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) response_format: Option<serde_json::Value>,
}

/// Streaming behaviour options. Setting `include_usage = true` makes OpenAI (and compatible
/// servers) emit a final `usage` chunk just before `[DONE]`, otherwise `usage` is `None` for the
/// entire stream.
#[derive(Serialize)]
pub(crate) struct OpenAiStreamOptions {
    pub(crate) include_usage: bool,
}

#[derive(Serialize)]
pub(crate) struct OpenAiMessage {
    pub(crate) role: String,
    pub(crate) content: OpenAiContent,
}

#[derive(Serialize)]
#[serde(untagged)]
pub(crate) enum OpenAiContent {
    String(String),
    Parts(Vec<OpenAiContentPart>),
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum OpenAiContentPart {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: OpenAiImageUrl,
    },
    /// `OpenAI` file content part for documents (PDFs and similar). Carries either a Files API
    /// `file_id` or inline `file_data` (a `data:` URI base64 payload) plus an optional filename
    /// hint.
    File {
        file: OpenAiFile,
    },
    /// `OpenAI` audio input. Currently accepted on `gpt-audio*` / `gpt-4o-audio-preview` and on
    /// Mantle's Voxtral mini / small chat models.
    InputAudio {
        input_audio: OpenAiAudioPayload,
    },
}

#[derive(Serialize)]
pub(crate) struct OpenAiImageUrl {
    pub(crate) url: String,
}

#[derive(Serialize)]
pub(crate) struct OpenAiFile {
    /// Either `file_id` (Files API reference) or `file_data` (data: URI inline base64) — exactly
    /// one of the two is set per content part.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) file_data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) filename: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct OpenAiAudioPayload {
    /// Base64-encoded audio bytes (standard alphabet).
    pub(crate) data: String,
    /// Audio format string (`wav` or `mp3` are documented).
    pub(crate) format: String,
}

// --- Response serde structs ---

#[derive(Deserialize)]
pub(crate) struct OpenAiResponse {
    pub(crate) model: String,
    pub(crate) choices: Vec<OpenAiChoice>,
    pub(crate) usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
pub(crate) struct OpenAiChoice {
    pub(crate) message: OpenAiChoiceMessage,
    /// OpenAI's documented `finish_reason` values: `stop`, `length`, `tool_calls`,
    /// `content_filter`, `function_call` (deprecated). `null` only appears on in-progress stream
    /// snapshots, never on a completed `POST /chat/completions` response.
    #[serde(default)]
    pub(crate) finish_reason: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct OpenAiChoiceMessage {
    pub(crate) content: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct OpenAiUsage {
    pub(crate) prompt_tokens: u64,
    pub(crate) completion_tokens: u64,
    /// Cached-prompt breakdown. OpenAI caches eligible prompts automatically; `cached_tokens` is
    /// the portion of `prompt_tokens` served from cache (a subset, not additive).
    #[serde(default)]
    pub(crate) prompt_tokens_details: Option<OpenAiPromptTokensDetails>,
}

#[derive(Deserialize, Default)]
pub(crate) struct OpenAiPromptTokensDetails {
    #[serde(default)]
    pub(crate) cached_tokens: u64,
}

#[derive(Deserialize)]
#[cfg(any(feature = "openai", feature = "ollama", test))]
pub(crate) struct OpenAiErrorResponse {
    pub(crate) error: OpenAiErrorDetail,
}

#[derive(Deserialize)]
#[cfg(any(feature = "openai", feature = "ollama", test))]
pub(crate) struct OpenAiErrorDetail {
    pub(crate) message: String,
}

// --- Streaming response serde structs ---

#[derive(Deserialize)]
pub(crate) struct OpenAiStreamChunk {
    pub(crate) model: String,
    pub(crate) choices: Vec<OpenAiStreamChoice>,
    pub(crate) usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
pub(crate) struct OpenAiStreamChoice {
    pub(crate) delta: OpenAiStreamDelta,
    /// `finish_reason` shows up on the final content chunk of an OpenAI stream
    /// (and stays `null` on every intermediate chunk). Surfaced to consumers
    /// via `StreamDelta::stop_reason` so they don't have to infer truncation
    /// from output-token counts.
    #[serde(default)]
    pub(crate) finish_reason: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct OpenAiStreamDelta {
    pub(crate) content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HttpsUrl, MediaType, ProviderFileId, message::Role};

    fn sample_messages() -> Vec<Message> {
        vec![
            Message::system("You are helpful."),
            Message::user("Hello"),
            Message::assistant("Hi!"),
        ]
    }

    #[test]
    fn build_request_keeps_system_in_messages() {
        let config = LanguageModelConfig::default();
        let req = build_request("gpt-5", &sample_messages(), &config, false);
        assert_eq!(req.messages.len(), 3);
        assert_eq!(req.messages[0].role, "system");
        assert_eq!(req.messages[1].role, "user");
        assert_eq!(req.messages[2].role, "assistant");
    }

    #[test]
    fn build_request_text_only_uses_string_content() {
        let config = LanguageModelConfig::default();
        let req = build_request("gpt-5", &[Message::user("Hello")], &config, false);
        let json = serde_json::to_value(&req.messages[0]).unwrap();
        assert!(json["content"].is_string());
        assert_eq!(json["content"], "Hello");
    }

    #[test]
    fn build_request_multipart_uses_array_content() {
        let config = LanguageModelConfig::default();
        let messages = vec![Message::with_parts(
            Role::User,
            vec![
                ContentPart::text("Describe:"),
                ContentPart::image(MediaSource::Url {
                    url: HttpsUrl::parse("https://example.com/img.png").unwrap(),
                }),
            ],
        )];
        let req = build_request("gpt-5", &messages, &config, false);
        let json = serde_json::to_value(&req.messages[0]).unwrap();
        assert!(json["content"].is_array());
        let parts = json["content"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[1]["image_url"]["url"], "https://example.com/img.png");
    }

    #[test]
    fn build_request_omits_optional_fields() {
        let config = LanguageModelConfig::default();
        let req = build_request("gpt-5", &[Message::user("hi")], &config, false);
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("temperature").is_none());
        assert!(json.get("max_tokens").is_none());
        assert!(json.get("top_p").is_none());
        assert!(json.get("stop").is_none());
        assert!(json.get("stream_options").is_none());
    }

    #[test]
    fn stream_options_serialize_with_include_usage() {
        let opts = OpenAiStreamOptions {
            include_usage: true,
        };
        let json = serde_json::to_value(&opts).unwrap();
        assert_eq!(json, serde_json::json!({"include_usage": true}));
    }

    #[test]
    fn build_request_stream_options_serialize_when_set() {
        let config = LanguageModelConfig::default();
        let mut req = build_request("gpt-5", &[Message::user("hi")], &config, false);
        req.stream = Some(true);
        req.stream_options = Some(OpenAiStreamOptions {
            include_usage: true,
        });
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["stream"], true);
        assert_eq!(json["stream_options"]["include_usage"], true);
    }

    #[test]
    fn build_request_includes_config_fields() {
        let config = LanguageModelConfig {
            temperature: Some(0.7),
            max_tokens: Some(2048),
            top_p: Some(0.95),
            stop: vec!["END".into()],
            ..Default::default()
        };
        let req = build_request("gpt-5", &[Message::user("hi")], &config, false);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["temperature"], 0.7);
        assert_eq!(json["max_tokens"], 2048);
        assert_eq!(json["top_p"], 0.95);
        assert_eq!(json["stop"], serde_json::json!(["END"]));
    }

    #[test]
    fn build_request_routes_to_completion_tokens_when_caller_says_so() {
        let config = LanguageModelConfig {
            max_tokens: Some(1024),
            ..Default::default()
        };
        let req = build_request("gpt-5.4-mini", &[Message::user("hi")], &config, true);
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("max_tokens").is_none(), "legacy field omitted");
        assert_eq!(json["max_completion_tokens"], 1024);
    }

    #[test]
    fn build_request_routes_to_legacy_max_tokens_when_caller_says_so() {
        let config = LanguageModelConfig {
            max_tokens: Some(1024),
            ..Default::default()
        };
        let req = build_request("gpt-4o", &[Message::user("hi")], &config, false);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["max_tokens"], 1024);
        assert!(json.get("max_completion_tokens").is_none());
    }

    fn openai_response_from_json(json: serde_json::Value) -> OpenAiResponse {
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn parse_response() {
        let json = serde_json::json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1_714_000_000,
            "model": "gpt-5-2025-08-07",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "The answer is 4."},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let resp: OpenAiResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.model, "gpt-5-2025-08-07");
        assert_eq!(
            resp.choices[0].message.content.as_deref(),
            Some("The answer is 4.")
        );
        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
    }

    #[test]
    fn parse_response_empty_content() {
        let json = serde_json::json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1_714_000_000,
            "model": "gpt-5",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": null},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}
        });
        let resp: OpenAiResponse = serde_json::from_value(json).unwrap();
        assert!(resp.choices[0].message.content.is_none());
    }

    #[test]
    fn parse_openai_response_happy_path() {
        let api = openai_response_from_json(serde_json::json!({
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-5",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
        }));
        let result = parse_openai_response(api).unwrap();
        assert_eq!(result.content, "hi");
        assert!(result.thinking.is_none());
        assert_eq!(result.model.as_deref(), Some("gpt-5"));
        assert_eq!(result.usage.unwrap().input_tokens, 3);
        assert_eq!(result.usage.unwrap().output_tokens, 1);
    }

    #[test]
    fn parse_openai_response_null_content_returns_empty_response() {
        let api = openai_response_from_json(serde_json::json!({
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-5",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": null}, "finish_reason": "stop"}]
        }));
        let err = parse_openai_response(api).unwrap_err();
        assert!(matches!(err, LanguageModelError::EmptyResponse));
    }

    #[test]
    fn parse_openai_response_empty_string_content_returns_empty_response() {
        let api = openai_response_from_json(serde_json::json!({
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-5",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}]
        }));
        let err = parse_openai_response(api).unwrap_err();
        assert!(matches!(err, LanguageModelError::EmptyResponse));
    }

    #[test]
    fn parse_openai_response_no_choices_returns_empty_response() {
        let api = openai_response_from_json(serde_json::json!({
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-5",
            "choices": []
        }));
        let err = parse_openai_response(api).unwrap_err();
        assert!(matches!(err, LanguageModelError::EmptyResponse));
    }

    fn sse(data: &str) -> SseEvent {
        SseEvent {
            event_type: "message".to_owned(),
            data: data.to_owned(),
            id: String::new(),
        }
    }

    #[test]
    fn convert_openai_sse_event_done_marks_final() {
        let out = convert_openai_sse_event(Ok(sse("[DONE]")))
            .unwrap()
            .unwrap();
        assert!(out.is_final);
        assert!(out.usage.is_none());
        assert!(out.content.is_empty());
    }

    #[test]
    fn convert_openai_sse_event_content_chunk() {
        let chunk = serde_json::json!({
            "id": "x",
            "object": "chat.completion.chunk",
            "model": "gpt-5",
            "choices": [{"index": 0, "delta": {"content": "hello"}}],
            "usage": null
        });
        let out = convert_openai_sse_event(Ok(sse(&chunk.to_string())))
            .unwrap()
            .unwrap();
        assert_eq!(out.content, "hello");
        assert!(!out.is_final);
        assert_eq!(out.model.as_deref(), Some("gpt-5"));
        assert!(
            out.stop_reason.is_none(),
            "intermediate chunks (finish_reason: null) must not carry stop_reason",
        );
    }

    #[test]
    fn convert_openai_sse_event_final_chunk_with_finish_reason_length() {
        // The final content chunk Mantle emits before the usage-only chunk
        // carries finish_reason. We must surface it as StopReason::MaxTokens
        // so callers don't have to infer from output_tokens >= max_tokens.
        let chunk = serde_json::json!({
            "id": "x",
            "object": "chat.completion.chunk",
            "model": "gemma-3-4b-it",
            "choices": [{
                "index": 0,
                "delta": {"content": "tail"},
                "finish_reason": "length",
            }],
            "usage": null,
        });
        let out = convert_openai_sse_event(Ok(sse(&chunk.to_string())))
            .unwrap()
            .unwrap();
        assert_eq!(out.content, "tail");
        assert_eq!(out.stop_reason, Some(crate::StopReason::MaxTokens));
    }

    #[test]
    fn convert_openai_sse_event_finish_reason_only_chunk_yields_stop_reason() {
        // Some compatible servers emit a final chunk with finish_reason but
        // no content delta. Pre-fix this would be a keep-alive (dropped);
        // post-fix it must surface stop_reason to the caller.
        let chunk = serde_json::json!({
            "id": "x",
            "object": "chat.completion.chunk",
            "model": "gpt-5",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": null,
        });
        let out = convert_openai_sse_event(Ok(sse(&chunk.to_string())))
            .unwrap()
            .unwrap();
        assert!(out.content.is_empty());
        assert_eq!(out.stop_reason, Some(crate::StopReason::EndTurn));
    }

    #[test]
    fn convert_openai_sse_event_usage_chunk_when_include_usage_is_true() {
        let chunk = serde_json::json!({
            "id": "x",
            "object": "chat.completion.chunk",
            "model": "gpt-5",
            "choices": [{"index": 0, "delta": {}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33}
        });
        let out = convert_openai_sse_event(Ok(sse(&chunk.to_string())))
            .unwrap()
            .unwrap();
        let usage = out.usage.unwrap();
        assert_eq!(usage.input_tokens, 11);
        assert_eq!(usage.output_tokens, 22);
    }

    #[test]
    fn convert_openai_sse_event_keep_alive_skipped() {
        let chunk = serde_json::json!({
            "id": "x",
            "object": "chat.completion.chunk",
            "model": "gpt-5",
            "choices": [{"index": 0, "delta": {}}],
            "usage": null
        });
        let out = convert_openai_sse_event(Ok(sse(&chunk.to_string())));
        assert!(out.is_none());
    }

    #[test]
    fn convert_openai_sse_event_malformed_yields_error() {
        let out = convert_openai_sse_event(Ok(sse("not-json"))).unwrap();
        assert!(matches!(out, Err(LanguageModelError::Provider { .. })));
    }

    #[test]
    fn map_error_rate_limited() {
        let headers = reqwest::header::HeaderMap::new();
        let err = map_openai_error(
            429,
            r#"{"error":{"message":"Rate limit exceeded","type":"rate_limit_error","code":"rate_limit_exceeded"}}"#,
            &headers,
        );
        assert!(matches!(
            err,
            LanguageModelError::RateLimited {
                retry_after: None,
                ..
            }
        ));
    }

    #[test]
    fn map_error_rate_limited_honours_retry_after() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "5".parse().unwrap());
        let err = map_openai_error(429, "{}", &headers);
        match err {
            LanguageModelError::RateLimited {
                retry_after: Some(d),
                ..
            } => {
                assert_eq!(d, std::time::Duration::from_secs(5));
            }
            other => panic!("expected RateLimited with retry_after=Some(5s), got {other:?}"),
        }
    }

    #[test]
    fn map_error_auth() {
        let headers = reqwest::header::HeaderMap::new();
        let err = map_openai_error(
            401,
            r#"{"error":{"message":"Invalid API key","type":"invalid_request_error","code":null}}"#,
            &headers,
        );
        assert!(matches!(err, LanguageModelError::Authentication { .. }));
    }

    #[test]
    fn map_error_server() {
        let headers = reqwest::header::HeaderMap::new();
        let err = map_openai_error(500, "Internal error", &headers);
        assert!(matches!(err, LanguageModelError::Provider { .. }));
    }

    #[test]
    fn translate_image_url_emits_https_url() {
        let part = ContentPart::image(MediaSource::Url {
            url: HttpsUrl::parse("https://example.com/x.png").unwrap(),
        });
        let block = translate_part_for_openai(&part).unwrap();
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "image_url");
        assert_eq!(json["image_url"]["url"], "https://example.com/x.png");
    }

    #[test]
    fn translate_image_inline_bytes_emits_data_uri() {
        let part = ContentPart::image(MediaSource::InlineBytes {
            mime: MediaType::parse("image/png").unwrap(),
            data: b"hello".to_vec(),
        });
        let block = translate_part_for_openai(&part).unwrap();
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "image_url");
        assert_eq!(json["image_url"]["url"], "data:image/png;base64,aGVsbG8=");
    }

    #[test]
    fn translate_audio_inline_bytes_emits_input_audio_block() {
        let part = ContentPart::audio(MediaSource::InlineBytes {
            mime: MediaType::parse("audio/wav").unwrap(),
            data: b"RIFF".to_vec(),
        });
        let block = translate_part_for_openai(&part).unwrap();
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "input_audio");
        assert_eq!(json["input_audio"]["format"], "wav");
        assert_eq!(json["input_audio"]["data"], "UklGRg==");
    }

    #[test]
    fn translate_document_inline_bytes_emits_file_data_data_uri() {
        let part = ContentPart::document(
            MediaSource::InlineBytes {
                mime: MediaType::parse("application/pdf").unwrap(),
                data: b"%PDF".to_vec(),
            },
            Some("report.pdf".into()),
        );
        let block = translate_part_for_openai(&part).unwrap();
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "file");
        assert_eq!(json["file"]["filename"], "report.pdf");
        assert_eq!(
            json["file"]["file_data"],
            "data:application/pdf;base64,JVBERg=="
        );
        assert!(json["file"].get("file_id").is_none());
    }

    #[test]
    fn translate_document_provider_file_emits_file_id() {
        let part = ContentPart::document(
            MediaSource::ProviderFile {
                file_id: ProviderFileId::parse("file-abc").unwrap(),
            },
            None,
        );
        let block = translate_part_for_openai(&part).unwrap();
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "file");
        assert_eq!(json["file"]["file_id"], "file-abc");
        assert!(json["file"].get("file_data").is_none());
    }
}
