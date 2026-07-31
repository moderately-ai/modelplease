// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Responses API wire layer (adapter B) for [`super::BedrockMantleProvider`].
//!
//! Used exclusively by the Mantle Responses surface for `openai.gpt-5.*`. If/when a direct-OpenAI
//! Responses provider shows up, lift these into a top-level `responses_wire.rs` following the
//! `openai_wire`/`anthropic_wire` pattern.

use crate::{
    config::{LanguageModelConfig, ReasoningConfig, ResponseFormat},
    error::LanguageModelError,
    media::MediaSource,
    message::{ContentPart, Message, Role},
    response::{LanguageModelResponse, StopReason, StreamDelta, Usage},
    sse::SseEvent,
};

/// Build the Responses request body from messages + config.
///
/// System messages collapse into the top-level `instructions` field (analogous to Anthropic's
/// `system`). User/assistant messages become `input` items with content-part arrays. `store` is
/// always `false` — our [`LanguageModelProvider`] trait is stateless, so we send the full message
/// history each turn rather than carrying `previous_response_id`.
pub(super) fn build_responses_request(
    model: &str,
    messages: &[Message],
    config: &LanguageModelConfig,
    stream: bool,
) -> ResponsesRequest {
    // Collect system text into a single instructions string. Multiple system messages join with
    // newline separators — same convention as Anthropic's system blocks.
    let instructions_parts: Vec<String> = messages
        .iter()
        .filter(|m| matches!(m.role, Role::System))
        .flat_map(|m| {
            m.content
                .iter()
                .filter_map(|p| p.as_text().map(str::to_owned))
        })
        .collect();
    let instructions = if instructions_parts.is_empty() {
        None
    } else {
        Some(instructions_parts.join("\n"))
    };

    let input: Vec<ResponsesInputItem> = messages
        .iter()
        .filter(|m| !matches!(m.role, Role::System))
        .map(|msg| {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                #[expect(
                    clippy::unreachable,
                    reason = "outer .filter excludes System; impossible by construction"
                )]
                Role::System => unreachable!("filtered above"),
            };
            let content: Vec<ResponsesInputContent> = msg
                .content
                .iter()
                .filter_map(translate_part_for_responses)
                .collect();
            ResponsesInputItem {
                role: role.to_owned(),
                content,
            }
        })
        .collect();

    // Resolve reasoning + temperature. gpt-5 reasoning models reject `temperature` whenever
    // `reasoning.effort` is set — strip defensively here even though the capability validation layer
    // should have caught explicit conflicts upstream.
    let (reasoning, wire_temperature, wire_top_p) = match config.reasoning.as_ref() {
        Some(ReasoningConfig::Adaptive { effort }) => (
            Some(ResponsesReasoning {
                effort: effort.as_str(),
            }),
            None,
            config.top_p,
        ),
        Some(ReasoningConfig::Manual { .. }) => {
            debug_assert!(
                false,
                "Responses API has no manual-budget reasoning mode; capability validation \
                 should have rejected this upstream",
            );
            (None, config.temperature, config.top_p)
        }
        Some(ReasoningConfig::Off) | None => (None, config.temperature, config.top_p),
    };

    let text = match &config.response_format {
        ResponseFormat::Text => None,
        ResponseFormat::JsonObject => Some(ResponsesText {
            format: serde_json::json!({"type": "json_object"}),
        }),
        ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        } => Some(ResponsesText {
            format: serde_json::json!({
                "type": "json_schema",
                "name": name,
                "schema": schema,
                "strict": strict,
            }),
        }),
    };

    ResponsesRequest {
        model: model.to_owned(),
        input,
        instructions,
        store: false,
        reasoning,
        max_output_tokens: config.max_tokens,
        temperature: wire_temperature,
        top_p: wire_top_p,
        text,
        stream: if stream { Some(true) } else { None },
    }
}

/// Translate one [`ContentPart`] into the Responses input content shape.
fn translate_part_for_responses(part: &ContentPart) -> Option<ResponsesInputContent> {
    use base64::Engine;
    match part {
        ContentPart::Text { text } => Some(ResponsesInputContent::InputText { text: text.clone() }),
        ContentPart::Image { source } => match source {
            MediaSource::Url { url } => Some(ResponsesInputContent::InputImage {
                image_url: url.as_str().to_owned(),
            }),
            MediaSource::InlineBytes { mime, data } => {
                let b64 = base64::engine::general_purpose::STANDARD.encode(data);
                Some(ResponsesInputContent::InputImage {
                    image_url: format!("data:{};base64,{b64}", mime.as_str()),
                })
            }
            other => {
                tracing::warn!(
                    provider = "bedrock-mantle/responses",
                    source_kind = ?other.kind(),
                    "validate_request should have rejected this image source kind",
                );
                None
            }
        },
        ContentPart::Document { .. } | ContentPart::Audio { .. } | ContentPart::Video { .. } => {
            tracing::warn!(
                provider = "bedrock-mantle/responses",
                kind = %part.media_kind().map_or("?", crate::capabilities::MediaKind::label),
                "validate_request should have rejected this modality on the Responses surface; \
                 phase 6 supports text + image only",
            );
            None
        }
        // Responses doesn't expose explicit caching at this layer.
        ContentPart::CacheBreakpoint => None,
    }
}

/// Parse a Responses synchronous response envelope into our internal [`LanguageModelResponse`].
pub(super) fn parse_responses_response(
    api: ResponsesResponse,
) -> Result<LanguageModelResponse, LanguageModelError> {
    let mut text_parts: Vec<String> = Vec::new();
    let mut thinking_parts: Vec<String> = Vec::new();
    for item in &api.output {
        match item {
            ResponsesOutputItem::Message { content } => {
                for c in content {
                    if let ResponsesOutputContent::OutputText { text } = c {
                        text_parts.push(text.clone());
                    }
                }
            }
            ResponsesOutputItem::Reasoning { summary } => {
                if let Some(summary) = summary {
                    for s in summary {
                        if let Some(text) = s.get("text").and_then(serde_json::Value::as_str) {
                            thinking_parts.push(text.to_owned());
                        }
                    }
                }
            }
            ResponsesOutputItem::Unknown => {}
        }
    }
    if text_parts.is_empty() {
        return Err(LanguageModelError::EmptyResponse);
    }
    let content = text_parts.join("");
    let thinking = (!thinking_parts.is_empty()).then(|| thinking_parts.join(""));

    let stop_reason = responses_stop_reason(
        api.status.as_deref(),
        api.incomplete_details
            .as_ref()
            .and_then(|d| d.reason.as_deref()),
    );

    let usage = api.usage.as_ref().map(responses_usage_to_internal);

    Ok(LanguageModelResponse {
        content,
        thinking,
        usage,
        model: api.model,
        stop_reason,
    })
}

/// Map Responses `status` + `incomplete_details.reason` to our [`StopReason`].
///
/// - `status == "completed"` ⇒ `EndTurn`.
/// - `status == "incomplete"` + `incomplete_details.reason == "max_output_tokens"` ⇒ `MaxTokens`.
/// - Other `incomplete` reasons (e.g. `"content_filter"`) ride through as `Other(reason)` so
///   downstream consumers can distinguish them without losing information.
fn responses_stop_reason(
    status: Option<&str>,
    incomplete_reason: Option<&str>,
) -> Option<StopReason> {
    match (status, incomplete_reason) {
        (Some("completed"), _) => Some(StopReason::EndTurn),
        (Some("incomplete"), Some("max_output_tokens")) => Some(StopReason::MaxTokens),
        (Some("incomplete"), Some("content_filter")) => Some(StopReason::ContentFilter),
        (Some("incomplete"), Some(other)) => Some(StopReason::Other(other.to_owned())),
        (Some(other), _) if other != "completed" => Some(StopReason::Other(other.to_owned())),
        _ => None,
    }
}

/// Project a Responses usage block into our internal [`Usage`]. `input_tokens` on the wire
/// *includes* cached tokens; subtract so the invariant (`input` = uncached, cache counts disjoint)
/// holds — same convention as the Chat Completions wire mapping.
fn responses_usage_to_internal(u: &ResponsesUsage) -> Usage {
    let cached = u
        .input_tokens_details
        .as_ref()
        .map_or(0, |d| d.cached_tokens);
    Usage {
        input_tokens: u.input_tokens.saturating_sub(cached),
        output_tokens: u.output_tokens,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: cached,
    }
}

/// Project one Responses SSE event into our internal [`StreamDelta`].
///
/// Responses uses TYPED events (the SSE `event:` line carries the type, e.g.
/// `response.output_text.delta`) — not delta-on-choices like Chat Completions. We extract text
/// from `response.output_text.delta`, finalize on `response.completed`, and surface errors from
/// `response.failed`. Other event types (`response.created`, `response.in_progress`,
/// `response.output_item.added`, reasoning deltas, etc.) are silently skipped.
pub(super) fn convert_responses_sse_event(
    event_result: Result<SseEvent, LanguageModelError>,
) -> Option<Result<StreamDelta, LanguageModelError>> {
    let event = match event_result {
        Err(e) => return Some(Err(e)),
        Ok(e) => e,
    };
    match event.event_type.as_str() {
        "response.output_text.delta" => {
            let parsed: serde_json::Value = serde_json::from_str(&event.data).unwrap_or_default();
            let delta = parsed
                .get("delta")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            if delta.is_empty() {
                None
            } else {
                Some(Ok(StreamDelta {
                    content: delta.to_owned(),
                    thinking: None,
                    usage: None,
                    model: None,
                    stop_reason: None,
                    is_final: false,
                }))
            }
        }
        "response.completed" => {
            let parsed: serde_json::Value = serde_json::from_str(&event.data).unwrap_or_default();
            let response = parsed.get("response");
            let usage = response
                .and_then(|r| r.get("usage"))
                .and_then(|u| serde_json::from_value::<ResponsesUsage>(u.clone()).ok())
                .as_ref()
                .map(responses_usage_to_internal);
            let model = response
                .and_then(|r| r.get("model"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned);
            let status = response
                .and_then(|r| r.get("status"))
                .and_then(serde_json::Value::as_str);
            let incomplete_reason = response
                .and_then(|r| r.get("incomplete_details"))
                .and_then(|d| d.get("reason"))
                .and_then(serde_json::Value::as_str);
            let stop_reason = responses_stop_reason(status, incomplete_reason);
            Some(Ok(StreamDelta {
                content: String::new(),
                thinking: None,
                usage,
                model,
                stop_reason,
                is_final: true,
            }))
        }
        // `response.incomplete` is emitted when a stream terminates early but
        // the run is still recoverable (e.g. hit `max_output_tokens` mid-
        // generation, content_filter trip). The sync path treats these as
        // successful responses with the appropriate `StopReason`; the stream
        // path must do the same so a stream cut short by max_tokens stops
        // raising an error to callers that were previously fine with the
        // truncation. `response.failed` is the genuinely-fatal case and
        // keeps the error path.
        "response.incomplete" => {
            let parsed: serde_json::Value = serde_json::from_str(&event.data).unwrap_or_default();
            let response = parsed.get("response");
            let usage = response
                .and_then(|r| r.get("usage"))
                .and_then(|u| serde_json::from_value::<ResponsesUsage>(u.clone()).ok())
                .as_ref()
                .map(responses_usage_to_internal);
            let model = response
                .and_then(|r| r.get("model"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned);
            let incomplete_reason = response
                .and_then(|r| r.get("incomplete_details"))
                .and_then(|d| d.get("reason"))
                .and_then(serde_json::Value::as_str);
            // Reuse the sync mapper so the stream and non-stream paths
            // categorise an identical `incomplete_details.reason` the same
            // way (`max_output_tokens` ⇒ MaxTokens, `content_filter` ⇒
            // ContentFilter, anything else ⇒ Other(reason)). The status
            // here is the literal "incomplete" string the event carries.
            let stop_reason = responses_stop_reason(Some("incomplete"), incomplete_reason);
            Some(Ok(StreamDelta {
                content: String::new(),
                thinking: None,
                usage,
                model,
                stop_reason,
                is_final: true,
            }))
        }
        "response.failed" => {
            let parsed: serde_json::Value = serde_json::from_str(&event.data).unwrap_or_default();
            let message = parsed
                .get("response")
                .and_then(|r| r.get("error"))
                .and_then(|e| e.get("message"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown responses streaming error")
                .to_owned();
            Some(Err(LanguageModelError::provider(message)))
        }
        _ => None,
    }
}

// --- Responses request serde structs ---

#[derive(serde::Serialize)]
pub(super) struct ResponsesRequest {
    pub(super) model: String,
    pub(super) input: Vec<ResponsesInputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) instructions: Option<String>,
    /// Always `false` — our `LanguageModelProvider` trait is stateless, so we send full message
    /// history each turn rather than relying on Mantle's server-side `previous_response_id` store.
    pub(super) store: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) reasoning: Option<ResponsesReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<ResponsesText>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) stream: Option<bool>,
}

#[derive(serde::Serialize)]
pub(super) struct ResponsesInputItem {
    pub(super) role: String,
    content: Vec<ResponsesInputContent>,
}

#[derive(serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponsesInputContent {
    InputText { text: String },
    InputImage { image_url: String },
}

#[derive(serde::Serialize)]
pub(super) struct ResponsesReasoning {
    pub(super) effort: &'static str,
}

#[derive(serde::Serialize)]
struct ResponsesText {
    format: serde_json::Value,
}

// --- Responses response serde structs ---

#[derive(serde::Deserialize)]
pub(super) struct ResponsesResponse {
    output: Vec<ResponsesOutputItem>,
    #[serde(default)]
    usage: Option<ResponsesUsage>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    incomplete_details: Option<ResponsesIncompleteDetails>,
    #[serde(default)]
    model: Option<String>,
}

#[derive(serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponsesOutputItem {
    Message {
        content: Vec<ResponsesOutputContent>,
    },
    Reasoning {
        #[serde(default)]
        summary: Option<Vec<serde_json::Value>>,
    },
    #[serde(other)]
    Unknown,
}

#[derive(serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponsesOutputContent {
    OutputText {
        text: String,
    },
    #[serde(other)]
    Unknown,
}

#[derive(serde::Deserialize, Clone)]
struct ResponsesUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    input_tokens_details: Option<ResponsesInputTokensDetails>,
}

#[derive(serde::Deserialize, Default, Clone)]
struct ResponsesInputTokensDetails {
    #[serde(default)]
    cached_tokens: u64,
}

#[derive(serde::Deserialize)]
struct ResponsesIncompleteDetails {
    #[serde(default)]
    reason: Option<String>,
}
