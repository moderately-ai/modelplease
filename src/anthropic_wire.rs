// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Anthropic Messages API wire layer — DTOs, request builders, response parsers, SSE handling,
//! content-part translation, error mapping. Shared between two providers that speak the same
//! wire format:
//!
//! - [`crate::anthropic::AnthropicLanguageModel`] against `api.anthropic.com/v1/messages`.
//! - [`crate::bedrock_mantle::BedrockMantleProvider`]'s Messages surface
//!   (`{base}/anthropic/v1/messages` against `bedrock-mantle.{region}.api.aws`).
//!
//! Everything in this module is `pub(crate)`. Cross-crate consumers continue to use
//! `AnthropicLanguageModel` / `BedrockMantleProvider` as their entry points — the wire layer is
//! an internal sharing seam, not a public API.

use base64::Engine;
#[cfg(any(feature = "bedrock-mantle", test))]
use enumset::enum_set;
use serde::{Deserialize, Serialize};

#[cfg(any(feature = "bedrock-mantle", test))]
use crate::capabilities::{ReasoningCapability, ReasoningMode};
#[cfg(any(feature = "bedrock-mantle", test))]
use crate::config::ReasoningEffort;
use crate::{
    capabilities::ReasoningParamConflicts,
    config::{CacheTtl, LanguageModelConfig, PromptCaching, ReasoningConfig},
    error::LanguageModelError,
    media::MediaSource,
    message::{ContentPart, Message, Role},
    response::{LanguageModelResponse, StreamDelta, Usage},
    sse::SseEvent,
};

/// Default `max_tokens` when not specified in config (Anthropic requires this field).
pub(crate) const DEFAULT_MAX_TOKENS: u32 = 8192;

/// Build the Anthropic Messages request body from messages + config.
///
/// Returns `(body, needs_files_beta)` — when `needs_files_beta` is `true`, the request references
/// provider Files API ids and the caller must include the `anthropic-beta: files-api-2025-04-14`
/// header (Mantle ignores Anthropic beta headers, so its caller drops the flag).
///
/// Cache control is gated by `config.prompt_caching` AND the model id's family (see
/// [`model_supports_prompt_caching`]). The Mantle Messages adapter forces
/// `prompt_caching = Off` on the config it passes here because Mantle rejects `cache_control`.
#[tracing::instrument(skip_all, fields(model, msg_count = messages.len()), level = "trace")]
pub(crate) fn build_request(
    model: &str,
    messages: &[Message],
    config: &LanguageModelConfig,
) -> (AnthropicRequest, bool) {
    let mut system_blocks: Vec<AnthropicTextBlock> = Vec::new();
    let mut api_messages = Vec::new();
    let mut needs_files_beta = false;
    let cache_enabled = matches!(config.prompt_caching, PromptCaching::Auto)
        && model_supports_prompt_caching(model);

    for msg in messages {
        match msg.role {
            Role::System => {
                for part in &msg.content {
                    match part {
                        ContentPart::Text { text } => system_blocks.push(AnthropicTextBlock {
                            block_type: "text".to_owned(),
                            text: text.clone(),
                            cache_control: None,
                        }),
                        // Cache the system prefix up to this marker by tagging the preceding block.
                        ContentPart::CacheBreakpoint if cache_enabled => {
                            if let Some(last) = system_blocks.last_mut() {
                                last.cache_control =
                                    Some(AnthropicCacheControl::from_ttl(config.cache_ttl));
                            }
                        }
                        _ => {}
                    }
                }
            }
            Role::User | Role::Assistant => {
                let role = match msg.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    #[expect(
                        clippy::unreachable,
                        reason = "outer match arm is `User | Assistant`; \
                                  System is impossible here by construction"
                    )]
                    Role::System => unreachable!("outer match guard excludes System"),
                };
                let mut content: Vec<AnthropicContentBlockRequest> = Vec::new();
                for part in &msg.content {
                    if part.is_cache_breakpoint() {
                        if cache_enabled && let Some(last) = content.last_mut() {
                            last.set_cache_control(AnthropicCacheControl::from_ttl(
                                config.cache_ttl,
                            ));
                        }
                        continue;
                    }
                    if let Some(block) = translate_part_for_anthropic(part, &mut needs_files_beta) {
                        content.push(block);
                    }
                }
                api_messages.push(AnthropicMessage {
                    role: role.to_owned(),
                    content,
                });
            }
        }
    }

    // Resolve `thinking` + `output_config` from `config.reasoning`. When thinking is on, Anthropic
    // hard-rejects `temperature` / `top_k` and clamps `top_p` to [0.95, 1]. We strip forbidden
    // fields here defensively — the capability validation layer should have failed loud on a user
    // who explicitly set them, but unset / defaulted values must not leak through.
    let (thinking, output_config) = anthropic_thinking_from_config(config.reasoning.as_ref());
    let (wire_temperature, wire_top_p) = if thinking.is_some() {
        let top_p = config.top_p.filter(|p| (0.95..=1.0).contains(p));
        (None, top_p)
    } else {
        (config.temperature, config.top_p)
    };
    let request = AnthropicRequest {
        model: model.to_owned(),
        max_tokens: config.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        messages: api_messages,
        system: if system_blocks.is_empty() {
            None
        } else {
            Some(system_blocks)
        },
        temperature: wire_temperature,
        top_p: wire_top_p,
        stop_sequences: if config.stop.is_empty() {
            None
        } else {
            Some(config.stop.clone())
        },
        stream: None,
        output_format: None,
        thinking,
        output_config,
    };
    (request, needs_files_beta)
}

/// Walk the content blocks once: collect text (the visible response) and thinking traces
/// separately. Multiple text blocks join into the final content; thinking blocks join into the
/// optional thinking trace; redacted and unknown blocks are skipped. Empty text → `EmptyResponse`.
pub(crate) fn parse_anthropic_response(
    api_response: AnthropicResponse,
) -> Result<LanguageModelResponse, LanguageModelError> {
    let mut text_parts: Vec<String> = Vec::new();
    let mut thinking_parts: Vec<String> = Vec::new();
    for block in &api_response.content {
        match block {
            AnthropicContentBlockResponse::Text { text } => text_parts.push(text.clone()),
            AnthropicContentBlockResponse::Thinking { thinking } => {
                thinking_parts.push(thinking.clone());
            }
            AnthropicContentBlockResponse::RedactedThinking {}
            | AnthropicContentBlockResponse::Unknown => {}
        }
    }
    if text_parts.is_empty() {
        return Err(LanguageModelError::EmptyResponse);
    }
    let content = text_parts.join("");
    let thinking = (!thinking_parts.is_empty()).then(|| thinking_parts.join(""));
    let stop_reason = api_response
        .stop_reason
        .as_deref()
        .map(anthropic_stop_reason);
    Ok(LanguageModelResponse {
        content,
        thinking,
        usage: Some(Usage {
            input_tokens: api_response.usage.input_tokens,
            output_tokens: api_response.usage.output_tokens,
            cache_creation_input_tokens: api_response.usage.cache_creation_input_tokens,
            cache_read_input_tokens: api_response.usage.cache_read_input_tokens,
        }),
        model: Some(api_response.model),
        stop_reason,
    })
}

/// Translate one [`ContentPart`] to the Anthropic wire shape.
///
/// `validate_request` has the front-line responsibility for rejecting unsupported parts
/// (modality not on the table, source kind outside the model's allow-list); this helper is the
/// in-builder backstop and silently drops anything that slips through with a `tracing::warn!`
/// trace so capability-table bugs are observable.
pub(crate) fn translate_part_for_anthropic(
    part: &ContentPart,
    needs_files_beta: &mut bool,
) -> Option<AnthropicContentBlockRequest> {
    match part {
        ContentPart::Text { text } => Some(AnthropicContentBlockRequest::Text {
            text: text.clone(),
            cache_control: None,
        }),
        ContentPart::Image { source } => media_source_to_anthropic(source, needs_files_beta)
            .map(|s| AnthropicContentBlockRequest::Image { source: s }),
        ContentPart::Document { source, name } => {
            media_source_to_anthropic(source, needs_files_beta).map(|s| {
                AnthropicContentBlockRequest::Document {
                    source: s,
                    title: name.clone(),
                }
            })
        }
        ContentPart::Audio { .. } | ContentPart::Video { .. } => {
            tracing::warn!(
                provider = "anthropic-wire",
                kind = %part.media_kind().map_or("?", crate::capabilities::MediaKind::label),
                "validate_request should have rejected this modality before \
                 reaching build_request — capability-table bug suspected",
            );
            None
        }
        // Cache markers are handled in `build_request`, which attaches `cache_control` to the
        // preceding block; they produce no block.
        ContentPart::CacheBreakpoint => None,
    }
}

/// Map a [`MediaSource`] to the corresponding `AnthropicSource`. Returns `None` for sources
/// Anthropic doesn't natively speak (S3 today) — `validate_request` rejects those before this fn
/// runs.
fn media_source_to_anthropic(
    source: &MediaSource,
    needs_files_beta: &mut bool,
) -> Option<AnthropicSource> {
    match source {
        MediaSource::Url { url } => Some(AnthropicSource::Url {
            url: url.as_str().to_owned(),
        }),
        MediaSource::InlineBytes { mime, data } => Some(AnthropicSource::Base64 {
            media_type: mime.as_str().to_owned(),
            data: base64::engine::general_purpose::STANDARD.encode(data),
        }),
        MediaSource::ProviderFile { file_id } => {
            *needs_files_beta = true;
            Some(AnthropicSource::File {
                file_id: file_id.as_str().to_owned(),
            })
        }
        MediaSource::S3 { .. } => {
            tracing::warn!(
                provider = "anthropic-wire",
                "validate_request should have rejected S3 source before \
                 reaching build_request — capability-table bug suspected",
            );
            None
        }
    }
}

/// Compose the `anthropic-beta` header value when the request uses structured-outputs, Files API,
/// and/or the extended (1-hour) cache TTL. Anthropic accepts a single comma-separated value
/// listing every beta the request opts into. Used by the direct Anthropic provider; Mantle's
/// Messages adapter does NOT call this (Mantle handles version selection on the server side).
#[cfg(any(feature = "anthropic", test))]
pub(crate) fn apply_anthropic_beta_headers(
    req: reqwest::RequestBuilder,
    structured_outputs: bool,
    files_api: bool,
    extended_cache_ttl: bool,
) -> reqwest::RequestBuilder {
    let mut betas: Vec<&'static str> = Vec::new();
    if structured_outputs {
        betas.push("structured-outputs-2025-11-13");
    }
    if files_api {
        betas.push("files-api-2025-04-14");
    }
    if extended_cache_ttl {
        betas.push("extended-cache-ttl-2025-04-11");
    }
    if betas.is_empty() {
        req
    } else {
        req.header("anthropic-beta", betas.join(","))
    }
}

/// Anthropic-family `ReasoningParamConflicts`. Per the Anthropic extended-thinking docs:
/// `temperature` and `top_k` are hard-rejected whenever `thinking` is set, and `top_p` is only
/// honoured inside `[0.95, 1]`. Identical for direct Anthropic, Bedrock-Claude (Converse), and
/// Claude on Mantle's Messages surface.
pub(crate) const fn anthropic_reasoning_conflicts() -> ReasoningParamConflicts {
    ReasoningParamConflicts {
        temperature_forbidden: true,
        top_k_forbidden: true,
        top_p_allowed_range: Some(0.95..=1.0),
    }
}

/// `ReasoningCapability` for Claude **Haiku 4.5** on Mantle. Manual-only thinking
/// (`budget_tokens`) — adaptive thinking and the `effort` parameter 400 on Haiku 4.5 — mirroring
/// the native ([`crate::anthropic`]) and Bedrock-Converse Haiku surface. Manual budget is bounded
/// by Haiku 4.5's 64K max output. Sampling params are accepted with thinking off, so
/// `sampling_params_removed` is false (the reasoning-on `conflicts` still apply when thinking is
/// on).
#[cfg(feature = "bedrock-mantle")]
pub(crate) const fn mantle_anthropic_haiku_reasoning() -> ReasoningCapability {
    ReasoningCapability {
        supported_modes: enum_set!(ReasoningMode::Manual),
        supported_efforts: enum_set!(
            ReasoningEffort::Low | ReasoningEffort::Medium | ReasoningEffort::High
        ),
        manual_budget_range: Some(1024..=(64_000 - 1024)),
        conflicts: anthropic_reasoning_conflicts(),
        sampling_params_removed: false,
    }
}

/// `ReasoningCapability` for the 4.7+ Claude generation on Mantle (Opus 4.7 / Opus 4.8 / Mythos
/// Preview). Adaptive-only — manual `budget_tokens` is rejected with a 400, matching these models'
/// native surface — across the full effort enum including `xhigh`/`max`. Sampling params
/// (`temperature`/`top_p`/`top_k`) are removed entirely: rejected in every request regardless of
/// reasoning state, so `sampling_params_removed` is true. Direct-Anthropic models use per-model
/// constructors in [`crate::anthropic`] because they carry further per-model carve-outs.
#[cfg(feature = "bedrock-mantle")]
pub(crate) const fn mantle_anthropic_adaptive_reasoning() -> ReasoningCapability {
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

/// Build the wire `thinking` block + `output_config` from the resolved [`ReasoningConfig`].
///
/// - `None` (caller didn't ask) or `Some(Off)` ⇒ omit both fields; Anthropic treats absence as
///   thinking disabled.
/// - `Some(Adaptive { effort })` ⇒ `thinking: {type: "adaptive"}` plus `output_config: {effort:
///   ...}` per the Messages API contract — the effort field lives at the request top level, NOT
///   inside `thinking`.
/// - `Some(Manual { budget_tokens })` ⇒ `thinking: {type: "enabled", budget_tokens}`; no
///   `output_config`.
///
/// Per-model gating (e.g. Opus 4.7 rejecting Manual) lives upstream in the capability validation
/// layer — this function is a pure mechanical translation and trusts whatever survived validation.
const fn anthropic_thinking_from_config(
    reasoning: Option<&ReasoningConfig>,
) -> (Option<AnthropicThinking>, Option<AnthropicOutputConfig>) {
    match reasoning {
        None | Some(ReasoningConfig::Off) => (None, None),
        Some(ReasoningConfig::Adaptive { effort }) => (
            Some(AnthropicThinking::Adaptive),
            Some(AnthropicOutputConfig {
                effort: effort.as_str(),
            }),
        ),
        Some(ReasoningConfig::Manual { budget_tokens }) => (
            Some(AnthropicThinking::Enabled {
                budget_tokens: *budget_tokens,
            }),
            None,
        ),
    }
}

fn anthropic_stop_reason(raw: &str) -> crate::StopReason {
    match raw {
        "end_turn" => crate::StopReason::EndTurn,
        "max_tokens" => crate::StopReason::MaxTokens,
        "stop_sequence" => crate::StopReason::StopSequence,
        "tool_use" => crate::StopReason::ToolUse,
        other => crate::StopReason::Other(other.to_owned()),
    }
}

/// Whether the model supports `cache_control`. Anthropic's catalog is Claude-only and every
/// current Claude model supports prompt caching on the direct surface; gating on the family keeps
/// one place to exclude a hypothetical non-caching addition.
pub(crate) fn model_supports_prompt_caching(model: &str) -> bool {
    model.contains("claude")
}

/// Read a streaming `usage` JSON object into [`Usage`], tolerating missing keys (each defaults to
/// `0`). `message_start` carries the prompt-side fields (`input_tokens`,
/// `cache_creation_input_tokens`, `cache_read_input_tokens`); `message_delta` carries
/// `output_tokens`.
///
/// Empirically (June 2026 sweep against AWS Bedrock Mantle / Haiku 4.5), the Mantle Anthropic
/// Messages surface includes the full prompt-side counters (`input_tokens`,
/// `cache_creation_input_tokens`, `cache_read_input_tokens`) on `message_delta` events
/// — not only `output_tokens`. Last-usage-wins is therefore safe for consumers reading the final
/// delta. If a future Anthropic-shape upstream sends only output_tokens on `message_delta`, this
/// function's `unwrap_or(0)` will zero out the prompt-side fields and the consumer would need to
/// merge with the earlier `message_start` usage instead of overwriting.
fn parse_anthropic_stream_usage(usage: &serde_json::Value) -> Usage {
    let field = |key: &str| {
        usage
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
    };
    Usage {
        input_tokens: field("input_tokens"),
        output_tokens: field("output_tokens"),
        cache_creation_input_tokens: field("cache_creation_input_tokens"),
        cache_read_input_tokens: field("cache_read_input_tokens"),
    }
}

/// Translate one Anthropic SSE event into an outgoing `StreamDelta`. Returns `None` for events
/// the caller should silently skip (`content_block_start`, `content_block_stop`, `ping`, and
/// content deltas whose text is empty). `message_start` emits a usage-only delta carrying the
/// prompt-side (cache) token counts.
pub(crate) fn convert_anthropic_sse_event(
    event_result: Result<SseEvent, LanguageModelError>,
) -> Option<Result<StreamDelta, LanguageModelError>> {
    let event = match event_result {
        Err(e) => return Some(Err(e)),
        Ok(event) => event,
    };
    match event.event_type.as_str() {
        "content_block_delta" => {
            let parsed: serde_json::Value = match serde_json::from_str(&event.data) {
                Ok(v) => v,
                Err(e) => {
                    return Some(Err(LanguageModelError::provider(format!(
                        "failed to parse stream chunk: {e}"
                    ))));
                }
            };
            let delta = &parsed["delta"];
            if delta.get("type").and_then(|t| t.as_str()) == Some("text_delta") {
                let text = delta["text"].as_str().unwrap_or_default().to_owned();
                if !text.is_empty() {
                    return Some(Ok(StreamDelta {
                        content: text,
                        thinking: None,
                        usage: None,
                        model: None,
                        stop_reason: None,
                        is_final: false,
                    }));
                }
            }
            None
        }
        "message_start" => {
            let parsed: serde_json::Value = serde_json::from_str(&event.data).unwrap_or_default();
            let usage = parsed
                .get("message")
                .and_then(|m| m.get("usage"))
                .map(parse_anthropic_stream_usage);
            usage.map(|u| {
                Ok(StreamDelta {
                    content: String::new(),
                    thinking: None,
                    usage: Some(u),
                    model: None,
                    stop_reason: None,
                    is_final: false,
                })
            })
        }
        "message_delta" => {
            let parsed: serde_json::Value = serde_json::from_str(&event.data).unwrap_or_default();
            let usage = parsed.get("usage").map(parse_anthropic_stream_usage);
            let stop_reason = parsed
                .get("delta")
                .and_then(|d| d.get("stop_reason"))
                .and_then(serde_json::Value::as_str)
                .map(anthropic_stop_reason);
            Some(Ok(StreamDelta {
                content: String::new(),
                thinking: None,
                usage,
                model: None,
                stop_reason,
                is_final: false,
            }))
        }
        "message_stop" => Some(Ok(StreamDelta {
            content: String::new(),
            thinking: None,
            usage: None,
            model: None,
            stop_reason: None,
            is_final: true,
        })),
        "error" => {
            let parsed: serde_json::Value = serde_json::from_str(&event.data).unwrap_or_default();
            let message = parsed
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("unknown streaming error")
                .to_owned();
            Some(Err(LanguageModelError::provider(message)))
        }
        _ => None,
    }
}

/// Map HTTP status + error body + response headers to a `LanguageModelError`. On 429 the
/// `Retry-After` header (delta-seconds form) is forwarded into `RateLimited::retry_after` so the
/// retry loop can honour it.
pub(crate) fn map_anthropic_error(
    status: u16,
    body: &str,
    headers: &reqwest::header::HeaderMap,
) -> LanguageModelError {
    let message = serde_json::from_str::<AnthropicErrorResponse>(body)
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
/// intentionally not supported — Anthropic emits delta-seconds in practice, and an unparseable
/// value falls back to the existing exponential backoff in `with_retry`.
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

// --- Request serde structs ---

#[derive(Serialize)]
pub(crate) struct AnthropicRequest {
    pub(crate) model: String,
    pub(crate) max_tokens: u32,
    pub(crate) messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) system: Option<Vec<AnthropicTextBlock>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) output_format: Option<serde_json::Value>,
    /// Extended-thinking configuration. Present whenever the request turns reasoning on; omitted
    /// otherwise so the wire stays minimal on plain text calls. Two variants serialize at the
    /// same JSON path distinguished by `type`: `"enabled"` (manual budget) and `"adaptive"`
    /// (qualitative effort lives separately on `output_config`, not inside `thinking`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) thinking: Option<AnthropicThinking>,
    /// Adaptive-thinking effort guidance — separate top-level field per the Messages API
    /// contract. Only populated when `thinking` is `Adaptive` AND an explicit effort was
    /// requested; absence falls back to Anthropic's documented `high` default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) output_config: Option<AnthropicOutputConfig>,
}

/// Anthropic Messages API `thinking` block.
///
/// `Enabled { budget_tokens }` is the legacy manual mode — supported on every current Claude
/// model EXCEPT Claude Opus 4.7 (which rejects it with a 400) and is deprecated on Sonnet 4.6 /
/// Opus 4.6 but still functional. `Adaptive` is the future-direction mode — required on Opus
/// 4.7, recommended on Sonnet 4.6 / Opus 4.6 / Mythos Preview. Effort lives on the sibling
/// [`AnthropicOutputConfig`], not inside this enum (per the Messages API wire contract).
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum AnthropicThinking {
    Enabled { budget_tokens: u32 },
    Adaptive,
}

/// `output_config` top-level field carrying the adaptive-thinking effort hint. Currently the only
/// field we emit; structured as a dedicated DTO so future output-format knobs (display, etc.)
/// can extend additively without changing call sites.
#[derive(Serialize)]
pub(crate) struct AnthropicOutputConfig {
    pub(crate) effort: &'static str,
}

#[derive(Serialize)]
pub(crate) struct AnthropicMessage {
    pub(crate) role: String,
    pub(crate) content: Vec<AnthropicContentBlockRequest>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum AnthropicContentBlockRequest {
    Text {
        text: String,
        /// Marks this block as a prompt-cache breakpoint when set — Anthropic caches the prefix
        /// up to and including it.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    Image {
        source: AnthropicSource,
    },
    /// PDF document block. `title` is the filename hint Anthropic surfaces alongside the document
    /// for prompt context — skipped when not supplied so the wire payload stays minimal.
    Document {
        source: AnthropicSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
}

/// Anthropic's image/document `source` object. Tag values match the Messages API contract: `url`,
/// `base64`, `file`.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum AnthropicSource {
    Url { url: String },
    Base64 { media_type: String, data: String },
    File { file_id: String },
}

#[derive(Serialize)]
pub(crate) struct AnthropicTextBlock {
    #[serde(rename = "type")]
    pub(crate) block_type: String,
    pub(crate) text: String,
    /// Prompt-cache breakpoint for this system block when set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_control: Option<AnthropicCacheControl>,
}

/// Anthropic `cache_control` marker. The default 5-min ephemeral form is GA and needs no
/// `anthropic-beta` header; the 1-hour form sets `ttl: "1h"` and requires the
/// `extended-cache-ttl-2025-04-11` beta header (applied in [`apply_anthropic_beta_headers`]).
#[derive(Serialize, Clone)]
pub(crate) struct AnthropicCacheControl {
    #[serde(rename = "type")]
    pub(crate) cache_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) ttl: Option<String>,
}

impl AnthropicCacheControl {
    /// Build from our [`CacheTtl`]: `FiveMin` omits `ttl` (default ephemeral); `OneHour` sets
    /// `ttl: "1h"`.
    fn from_ttl(ttl: CacheTtl) -> Self {
        let ttl = match ttl {
            CacheTtl::FiveMin => None,
            CacheTtl::OneHour => Some("1h".to_owned()),
        };
        Self {
            cache_type: "ephemeral".to_owned(),
            ttl,
        }
    }
}

impl AnthropicContentBlockRequest {
    /// Attach a cache breakpoint to this block. Only text blocks carry `cache_control` in our
    /// request shape; the adapter places the user-message breakpoint after a text field, so a
    /// marker following a non-text block is a no-op (no caching there, never an error).
    fn set_cache_control(&mut self, control: AnthropicCacheControl) {
        if let Self::Text { cache_control, .. } = self {
            *cache_control = Some(control);
        }
    }
}

// --- Response serde structs ---

#[derive(Deserialize)]
pub(crate) struct AnthropicResponse {
    pub(crate) model: String,
    pub(crate) content: Vec<AnthropicContentBlockResponse>,
    pub(crate) usage: AnthropicUsage,
    /// Anthropic's terminal reason for stopping. Documented values: `end_turn`, `max_tokens`,
    /// `stop_sequence`, `tool_use`. Captured so downstream (predict) can distinguish truncation
    /// from natural completion.
    #[serde(default)]
    pub(crate) stop_reason: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum AnthropicContentBlockResponse {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
    },
    RedactedThinking {},
    #[serde(other)]
    Unknown,
}

#[derive(Deserialize)]
pub(crate) struct AnthropicUsage {
    pub(crate) input_tokens: u64,
    pub(crate) output_tokens: u64,
    /// Tokens written to the cache on a miss. Absent (→ `0`) on responses where no
    /// `cache_control` breakpoint was sent.
    #[serde(default)]
    pub(crate) cache_creation_input_tokens: u64,
    /// Tokens served from the cache on a hit. Absent (→ `0`) on a cold call.
    #[serde(default)]
    pub(crate) cache_read_input_tokens: u64,
}

#[derive(Deserialize)]
pub(crate) struct AnthropicErrorResponse {
    pub(crate) error: AnthropicErrorDetail,
}

#[derive(Deserialize)]
pub(crate) struct AnthropicErrorDetail {
    pub(crate) message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HttpsUrl, MediaType, ProviderFileId};

    fn sample_messages() -> Vec<Message> {
        vec![
            Message::system("You are helpful."),
            Message::user("Hello"),
            Message::assistant("Hi there!"),
            Message::user("What is 2+2?"),
        ]
    }

    #[test]
    fn build_request_separates_system() {
        let config = LanguageModelConfig::default();
        let (req, _) = build_request("claude-sonnet-4-6", &sample_messages(), &config);
        assert!(req.system.is_some());
        assert_eq!(req.system.as_ref().unwrap().len(), 1);
        assert_eq!(req.system.as_ref().unwrap()[0].text, "You are helpful.");
        assert_eq!(req.messages.len(), 3);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.messages[1].role, "assistant");
        assert_eq!(req.messages[2].role, "user");
    }

    #[test]
    fn build_request_no_system() {
        let config = LanguageModelConfig::default();
        let messages = vec![Message::user("Hello")];
        let (req, _) = build_request("model", &messages, &config);
        assert!(req.system.is_none());
    }

    #[test]
    fn build_request_default_max_tokens() {
        let config = LanguageModelConfig::default();
        let (req, _) = build_request("model", &[Message::user("hi")], &config);
        assert_eq!(req.max_tokens, DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn build_request_custom_max_tokens() {
        let config = LanguageModelConfig {
            max_tokens: Some(1024),
            ..Default::default()
        };
        let (req, _) = build_request("model", &[Message::user("hi")], &config);
        assert_eq!(req.max_tokens, 1024);
    }

    #[test]
    fn build_request_maps_config() {
        let config = LanguageModelConfig {
            temperature: Some(0.5),
            top_p: Some(0.9),
            stop: vec!["STOP".into()],
            ..Default::default()
        };
        let (req, _) = build_request("model", &[Message::user("hi")], &config);
        assert_eq!(req.temperature, Some(0.5));
        assert_eq!(req.top_p, Some(0.9));
        assert_eq!(req.stop_sequences, Some(vec!["STOP".into()]));
        assert!(
            req.thinking.is_none(),
            "no reasoning set ⇒ no thinking field"
        );
    }

    #[test]
    fn build_request_adaptive_thinking_strips_temperature() {
        let config = LanguageModelConfig {
            temperature: Some(0.5),
            top_p: Some(0.97),
            reasoning: Some(ReasoningConfig::Adaptive {
                effort: ReasoningEffort::High,
            }),
            ..Default::default()
        };
        let (req, _) = build_request("model", &[Message::user("hi")], &config);
        assert!(
            req.temperature.is_none(),
            "temperature stripped when thinking is on"
        );
        assert_eq!(
            req.top_p,
            Some(0.97),
            "top_p preserved when inside [0.95, 1]"
        );
        let wire = serde_json::to_value(&req).unwrap();
        assert_eq!(wire["thinking"]["type"], "adaptive");
        assert!(wire["thinking"].get("effort").is_none());
        assert_eq!(wire["output_config"]["effort"], "high");
    }

    #[test]
    fn build_request_manual_thinking_emits_budget_tokens() {
        let config = LanguageModelConfig {
            reasoning: Some(ReasoningConfig::Manual {
                budget_tokens: 4096,
            }),
            ..Default::default()
        };
        let (req, _) = build_request("model", &[Message::user("hi")], &config);
        let wire = serde_json::to_value(&req).unwrap();
        assert_eq!(wire["thinking"]["type"], "enabled");
        assert_eq!(wire["thinking"]["budget_tokens"], 4096);
        assert!(wire.get("output_config").is_none());
    }

    #[test]
    fn build_request_off_reasoning_omits_thinking_field() {
        let config = LanguageModelConfig {
            reasoning: Some(ReasoningConfig::Off),
            temperature: Some(0.2),
            ..Default::default()
        };
        let (req, _) = build_request("model", &[Message::user("hi")], &config);
        assert!(req.thinking.is_none(), "Off ⇒ omit thinking entirely");
        assert_eq!(
            req.temperature,
            Some(0.2),
            "temperature preserved when thinking is off"
        );
    }

    #[test]
    fn build_request_thinking_drops_out_of_band_top_p() {
        let config = LanguageModelConfig {
            top_p: Some(0.5),
            reasoning: Some(ReasoningConfig::Adaptive {
                effort: ReasoningEffort::Medium,
            }),
            ..Default::default()
        };
        let (req, _) = build_request("model", &[Message::user("hi")], &config);
        assert!(req.top_p.is_none());
    }

    #[test]
    fn build_request_image_content() {
        let config = LanguageModelConfig::default();
        let messages = vec![Message::with_parts(
            Role::User,
            vec![
                ContentPart::text("Describe this:"),
                ContentPart::image(MediaSource::Url {
                    url: HttpsUrl::parse("https://example.com/img.png").unwrap(),
                }),
            ],
        )];
        let (req, _) = build_request("model", &messages, &config);
        assert_eq!(req.messages[0].content.len(), 2);
        let json = serde_json::to_value(&req.messages[0].content[1]).unwrap();
        assert_eq!(json["type"], "image");
        assert_eq!(json["source"]["type"], "url");
        assert_eq!(json["source"]["url"], "https://example.com/img.png");
    }

    #[test]
    fn request_serializes_correctly() {
        let config = LanguageModelConfig::default();
        let (req, _) = build_request("claude-sonnet-4-6", &[Message::user("Hello")], &config);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "claude-sonnet-4-6");
        assert!(json.get("system").is_none());
        assert!(json["messages"].is_array());
    }

    #[test]
    fn build_request_sets_cache_control_on_last_system_block() {
        let config = LanguageModelConfig::default();
        let messages = vec![
            Message::with_parts(
                Role::System,
                vec![
                    ContentPart::text("big static prompt"),
                    ContentPart::cache_breakpoint(),
                ],
            ),
            Message::user("hi"),
        ];
        let (req, _) = build_request("claude-haiku-4-5", &messages, &config);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["system"][0]["type"], "text");
        assert_eq!(json["system"][0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn build_request_omits_ttl_for_five_min_default() {
        let config = LanguageModelConfig::default();
        let messages = vec![Message::with_parts(
            Role::System,
            vec![
                ContentPart::text("big static prompt"),
                ContentPart::cache_breakpoint(),
            ],
        )];
        let (req, _) = build_request("claude-haiku-4-5", &messages, &config);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["system"][0]["cache_control"]["type"], "ephemeral");
        assert!(json["system"][0]["cache_control"].get("ttl").is_none());
    }

    #[test]
    fn build_request_sets_one_hour_ttl_on_cache_control() {
        let config = LanguageModelConfig {
            cache_ttl: CacheTtl::OneHour,
            ..Default::default()
        };
        let messages = vec![Message::with_parts(
            Role::System,
            vec![
                ContentPart::text("big static prompt"),
                ContentPart::cache_breakpoint(),
            ],
        )];
        let (req, _) = build_request("claude-haiku-4-5", &messages, &config);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(json["system"][0]["cache_control"]["ttl"], "1h");
    }

    #[test]
    fn build_request_no_cache_control_for_non_caching_model() {
        let config = LanguageModelConfig::default();
        let messages = vec![Message::with_parts(
            Role::System,
            vec![
                ContentPart::text("big static prompt"),
                ContentPart::cache_breakpoint(),
            ],
        )];
        let (req, _) = build_request("some-other-model", &messages, &config);
        let json = serde_json::to_value(&req).unwrap();
        assert!(json["system"][0].get("cache_control").is_none());
    }

    #[test]
    fn build_request_user_cache_control_on_preceding_block() {
        let config = LanguageModelConfig::default();
        let messages = vec![Message::with_parts(
            Role::User,
            vec![
                ContentPart::text("stable head"),
                ContentPart::cache_breakpoint(),
                ContentPart::text("dynamic tail"),
            ],
        )];
        let (req, _) = build_request("claude-haiku-4-5", &messages, &config);
        assert_eq!(req.messages[0].content.len(), 2);
        let head = serde_json::to_value(&req.messages[0].content[0]).unwrap();
        let tail = serde_json::to_value(&req.messages[0].content[1]).unwrap();
        assert_eq!(head["cache_control"]["type"], "ephemeral");
        assert!(tail.get("cache_control").is_none());
    }

    #[test]
    fn build_request_drops_cache_control_when_prompt_caching_off() {
        // Used by Mantle's Messages adapter — even on a "claude" model id, PromptCaching::Off
        // suppresses cache_control emission. Pins that contract.
        let config = LanguageModelConfig {
            prompt_caching: PromptCaching::Off,
            ..Default::default()
        };
        let messages = vec![Message::with_parts(
            Role::System,
            vec![
                ContentPart::text("big static prompt"),
                ContentPart::cache_breakpoint(),
            ],
        )];
        let (req, _) = build_request("claude-haiku-4-5", &messages, &config);
        let json = serde_json::to_value(&req).unwrap();
        assert!(json["system"][0].get("cache_control").is_none());
    }

    #[test]
    fn parse_response_maps_cache_tokens() {
        let json = serde_json::json!({
            "id": "msg_123",
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 7,
                "output_tokens": 2,
                "cache_creation_input_tokens": 1024,
                "cache_read_input_tokens": 2048
            }
        });
        let resp: AnthropicResponse = serde_json::from_value(json).unwrap();
        let usage = parse_anthropic_response(resp).unwrap().usage.unwrap();
        assert_eq!(usage.input_tokens, 7);
        assert_eq!(usage.cache_creation_input_tokens, 1024);
        assert_eq!(usage.cache_read_input_tokens, 2048);
    }

    #[test]
    fn parse_response_cache_tokens_default_zero_when_absent() {
        let json = serde_json::json!({
            "id": "msg_1",
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 7, "output_tokens": 2}
        });
        let resp: AnthropicResponse = serde_json::from_value(json).unwrap();
        let usage = parse_anthropic_response(resp).unwrap().usage.unwrap();
        assert_eq!(usage.cache_creation_input_tokens, 0);
        assert_eq!(usage.cache_read_input_tokens, 0);
    }

    #[test]
    fn parse_response() {
        let json = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [
                {"type": "text", "text": "The answer is 4."}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let resp: AnthropicResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.model, "claude-sonnet-4-6");
        assert_eq!(resp.usage.input_tokens, 10);
        let text = resp.content.iter().find_map(|b| {
            if let AnthropicContentBlockResponse::Text { text } = b {
                Some(text.clone())
            } else {
                None
            }
        });
        assert_eq!(text, Some("The answer is 4.".to_owned()));
    }

    #[test]
    fn parse_response_captures_thinking_separately_from_text() {
        let json = serde_json::json!({
            "id": "msg_123",
            "model": "claude-sonnet-4-6",
            "content": [
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "text", "text": "The answer is 4."}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let resp: AnthropicResponse = serde_json::from_value(json).unwrap();
        let text = resp.content.iter().find_map(|b| {
            if let AnthropicContentBlockResponse::Text { text } = b {
                Some(text.clone())
            } else {
                None
            }
        });
        let thinking = resp.content.iter().find_map(|b| {
            if let AnthropicContentBlockResponse::Thinking { thinking } = b {
                Some(thinking.clone())
            } else {
                None
            }
        });
        assert_eq!(text, Some("The answer is 4.".to_owned()));
        assert_eq!(thinking, Some("Let me think...".to_owned()));
    }

    fn anthropic_response_from_json(json: serde_json::Value) -> AnthropicResponse {
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn parse_anthropic_response_happy_path() {
        let api = anthropic_response_from_json(serde_json::json!({
            "id": "msg_1",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 7, "output_tokens": 2}
        }));
        let result = parse_anthropic_response(api).unwrap();
        assert_eq!(result.content, "hello");
        assert!(result.thinking.is_none());
        assert_eq!(result.model.as_deref(), Some("claude-sonnet-4-6"));
        assert_eq!(result.usage.unwrap().input_tokens, 7);
        assert_eq!(result.usage.unwrap().output_tokens, 2);
    }

    #[test]
    fn parse_anthropic_response_joins_multiple_text_blocks() {
        let api = anthropic_response_from_json(serde_json::json!({
            "id": "msg_2",
            "model": "claude-sonnet-4-6",
            "content": [
                {"type": "text", "text": "part one "},
                {"type": "text", "text": "part two"}
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }));
        let result = parse_anthropic_response(api).unwrap();
        assert_eq!(result.content, "part one part two");
    }

    #[test]
    fn parse_anthropic_response_combines_thinking_with_text() {
        let api = anthropic_response_from_json(serde_json::json!({
            "id": "msg_3",
            "model": "claude-sonnet-4-6",
            "content": [
                {"type": "thinking", "thinking": "first thought"},
                {"type": "thinking", "thinking": ", second thought"},
                {"type": "text", "text": "answer"}
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }));
        let result = parse_anthropic_response(api).unwrap();
        assert_eq!(result.content, "answer");
        assert_eq!(
            result.thinking.as_deref(),
            Some("first thought, second thought")
        );
    }

    #[test]
    fn parse_anthropic_response_skips_redacted_thinking_and_unknown() {
        let api = anthropic_response_from_json(serde_json::json!({
            "id": "msg_4",
            "model": "claude-sonnet-4-6",
            "content": [
                {"type": "redacted_thinking"},
                {"type": "tool_use", "id": "x", "name": "y"},
                {"type": "text", "text": "visible"}
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }));
        let result = parse_anthropic_response(api).unwrap();
        assert_eq!(result.content, "visible");
        assert!(
            result.thinking.is_none(),
            "redacted thinking should not surface"
        );
    }

    #[test]
    fn parse_anthropic_response_no_text_returns_empty_response() {
        let api = anthropic_response_from_json(serde_json::json!({
            "id": "msg_5",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "thinking", "thinking": "but no answer"}],
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }));
        let err = parse_anthropic_response(api).unwrap_err();
        assert!(matches!(err, LanguageModelError::EmptyResponse));
    }

    fn sse(event_type: &str, data: &str) -> SseEvent {
        SseEvent {
            event_type: event_type.to_owned(),
            data: data.to_owned(),
            id: String::new(),
        }
    }

    #[test]
    fn convert_anthropic_sse_event_text_delta() {
        let event = sse(
            "content_block_delta",
            r#"{"index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
        );
        let out = convert_anthropic_sse_event(Ok(event)).unwrap().unwrap();
        assert_eq!(out.content, "hi");
        assert!(!out.is_final);
        assert!(out.usage.is_none());
    }

    #[test]
    fn convert_anthropic_sse_event_message_delta_carries_usage() {
        let event = sse(
            "message_delta",
            r#"{"usage":{"input_tokens":12,"output_tokens":34}}"#,
        );
        let out = convert_anthropic_sse_event(Ok(event)).unwrap().unwrap();
        assert!(out.content.is_empty());
        assert!(!out.is_final);
        let usage = out.usage.unwrap();
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 34);
        assert!(
            out.stop_reason.is_none(),
            "message_delta without delta.stop_reason must leave stop_reason None",
        );
    }

    #[test]
    fn convert_anthropic_sse_event_message_delta_carries_stop_reason() {
        // Anthropic emits delta.stop_reason on the message_delta event that
        // accompanies the final usage tally — surface it as StopReason so
        // consumers don't have to infer truncation from output token counts.
        let event = sse(
            "message_delta",
            r#"{"delta":{"stop_reason":"max_tokens","stop_sequence":null},
                "usage":{"input_tokens":12,"output_tokens":34}}"#,
        );
        let out = convert_anthropic_sse_event(Ok(event)).unwrap().unwrap();
        assert_eq!(out.stop_reason, Some(crate::StopReason::MaxTokens));
        let usage = out.usage.unwrap();
        assert_eq!(usage.output_tokens, 34);
    }

    #[test]
    fn convert_anthropic_sse_event_message_delta_end_turn_stop_reason() {
        let event = sse(
            "message_delta",
            r#"{"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":50}}"#,
        );
        let out = convert_anthropic_sse_event(Ok(event)).unwrap().unwrap();
        assert_eq!(out.stop_reason, Some(crate::StopReason::EndTurn));
    }

    #[test]
    fn convert_anthropic_sse_event_message_stop_marks_final() {
        let event = sse("message_stop", "{}");
        let out = convert_anthropic_sse_event(Ok(event)).unwrap().unwrap();
        assert!(out.is_final);
        assert!(
            out.stop_reason.is_none(),
            "stop_reason is reported on message_delta on Anthropic — not message_stop",
        );
    }

    #[test]
    fn convert_anthropic_sse_event_error_event_yields_provider_error() {
        let event = sse(
            "error",
            r#"{"type":"error","error":{"type":"overloaded","message":"server is overloaded"}}"#,
        );
        let out = convert_anthropic_sse_event(Ok(event)).unwrap();
        match out {
            Err(LanguageModelError::Provider { message }) => {
                assert!(
                    message.contains("overloaded"),
                    "unexpected message: {message}"
                );
            }
            other => panic!("expected Provider error, got {other:?}"),
        }
    }

    #[test]
    fn convert_anthropic_sse_event_malformed_content_block_delta_yields_error() {
        let event = sse("content_block_delta", "not-json");
        let out = convert_anthropic_sse_event(Ok(event)).unwrap();
        assert!(matches!(out, Err(LanguageModelError::Provider { .. })));
    }

    #[test]
    fn convert_anthropic_sse_event_unknown_event_silently_skipped() {
        let event = sse("ping", "{}");
        assert!(convert_anthropic_sse_event(Ok(event)).is_none());
    }

    #[test]
    fn map_error_rate_limited() {
        let headers = reqwest::header::HeaderMap::new();
        let err = map_anthropic_error(
            429,
            r#"{"type":"error","error":{"type":"rate_limit_error","message":"Too many requests"}}"#,
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
        headers.insert(reqwest::header::RETRY_AFTER, "42".parse().unwrap());
        let err = map_anthropic_error(429, "{}", &headers);
        match err {
            LanguageModelError::RateLimited {
                retry_after: Some(d),
                ..
            } => {
                assert_eq!(d, std::time::Duration::from_secs(42));
            }
            other => panic!("expected RateLimited with retry_after=Some(42s), got {other:?}"),
        }
    }

    #[test]
    fn map_error_rate_limited_skips_malformed_retry_after() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::RETRY_AFTER,
            "Wed, 21 Oct 2026 07:28:00 GMT".parse().unwrap(),
        );
        let err = map_anthropic_error(429, "{}", &headers);
        assert!(matches!(
            err,
            LanguageModelError::RateLimited {
                retry_after: None,
                ..
            }
        ));
    }

    #[test]
    fn map_error_auth() {
        let headers = reqwest::header::HeaderMap::new();
        let err = map_anthropic_error(
            401,
            r#"{"type":"error","error":{"type":"authentication_error","message":"Invalid API key"}}"#,
            &headers,
        );
        assert!(matches!(err, LanguageModelError::Authentication { .. }));
    }

    #[test]
    fn map_error_server() {
        let headers = reqwest::header::HeaderMap::new();
        let err = map_anthropic_error(500, "Internal error", &headers);
        assert!(matches!(err, LanguageModelError::Provider { .. }));
    }

    fn inline_png(data: &[u8]) -> MediaSource {
        MediaSource::InlineBytes {
            mime: MediaType::parse("image/png").unwrap(),
            data: data.to_vec(),
        }
    }

    #[test]
    fn translate_image_url_emits_url_source() {
        let mut needs_beta = false;
        let part = ContentPart::image(MediaSource::Url {
            url: HttpsUrl::parse("https://example.com/x.png").unwrap(),
        });
        let block = translate_part_for_anthropic(&part, &mut needs_beta).unwrap();
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "image");
        assert_eq!(json["source"]["type"], "url");
        assert_eq!(json["source"]["url"], "https://example.com/x.png");
        assert!(!needs_beta);
    }

    #[test]
    fn translate_image_inline_bytes_emits_base64_source() {
        let mut needs_beta = false;
        let part = ContentPart::image(inline_png(b"hello"));
        let block = translate_part_for_anthropic(&part, &mut needs_beta).unwrap();
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "image");
        assert_eq!(json["source"]["type"], "base64");
        assert_eq!(json["source"]["media_type"], "image/png");
        assert_eq!(json["source"]["data"], "aGVsbG8=");
        assert!(!needs_beta);
    }

    #[test]
    fn translate_image_provider_file_sets_files_beta_flag() {
        let mut needs_beta = false;
        let part = ContentPart::image(MediaSource::ProviderFile {
            file_id: ProviderFileId::parse("file-abc").unwrap(),
        });
        let block = translate_part_for_anthropic(&part, &mut needs_beta).unwrap();
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["source"]["type"], "file");
        assert_eq!(json["source"]["file_id"], "file-abc");
        assert!(
            needs_beta,
            "ProviderFile source must flag the files-api beta header"
        );
    }

    #[test]
    fn translate_document_preserves_title() {
        let mut needs_beta = false;
        let part = ContentPart::document(
            MediaSource::InlineBytes {
                mime: MediaType::parse("application/pdf").unwrap(),
                data: b"%PDF".to_vec(),
            },
            Some("report.pdf".into()),
        );
        let block = translate_part_for_anthropic(&part, &mut needs_beta).unwrap();
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "document");
        assert_eq!(json["title"], "report.pdf");
        assert_eq!(json["source"]["type"], "base64");
        assert_eq!(json["source"]["media_type"], "application/pdf");
    }

    #[test]
    fn translate_audio_and_video_drop_with_warning() {
        let mut needs_beta = false;
        assert!(
            translate_part_for_anthropic(&ContentPart::audio(inline_png(b"x")), &mut needs_beta)
                .is_none()
        );
        assert!(
            translate_part_for_anthropic(&ContentPart::video(inline_png(b"x")), &mut needs_beta)
                .is_none()
        );
        assert!(!needs_beta);
    }

    #[test]
    fn build_request_threads_files_beta_when_provider_file_present() {
        let config = LanguageModelConfig::default();
        let messages = vec![Message::with_parts(
            Role::User,
            vec![
                ContentPart::text("Describe:"),
                ContentPart::image(MediaSource::ProviderFile {
                    file_id: ProviderFileId::parse("file-xyz").unwrap(),
                }),
            ],
        )];
        let (_, needs_beta) = build_request("claude-sonnet-4-6", &messages, &config);
        assert!(needs_beta);
    }

    fn beta_header(builder: reqwest::RequestBuilder) -> Option<String> {
        builder
            .build()
            .unwrap()
            .headers()
            .get("anthropic-beta")
            .map(|v| v.to_str().unwrap().to_owned())
    }

    #[test]
    fn apply_anthropic_beta_headers_omits_header_when_no_betas() {
        let client = reqwest::Client::new();
        let req =
            apply_anthropic_beta_headers(client.post("https://example.com"), false, false, false);
        assert!(beta_header(req).is_none());
    }

    #[test]
    fn apply_anthropic_beta_headers_combines_structured_outputs_and_files() {
        let client = reqwest::Client::new();
        let req =
            apply_anthropic_beta_headers(client.post("https://example.com"), true, true, false);
        assert_eq!(
            beta_header(req).as_deref(),
            Some("structured-outputs-2025-11-13,files-api-2025-04-14"),
        );
    }

    #[test]
    fn apply_anthropic_beta_headers_emits_extended_cache_ttl_only() {
        let client = reqwest::Client::new();
        let req =
            apply_anthropic_beta_headers(client.post("https://example.com"), false, false, true);
        assert_eq!(
            beta_header(req).as_deref(),
            Some("extended-cache-ttl-2025-04-11")
        );
    }

    #[test]
    fn apply_anthropic_beta_headers_combines_all_three_in_order() {
        let client = reqwest::Client::new();
        let req =
            apply_anthropic_beta_headers(client.post("https://example.com"), true, true, true);
        assert_eq!(
            beta_header(req).as_deref(),
            Some(
                "structured-outputs-2025-11-13,files-api-2025-04-14,extended-cache-ttl-2025-04-11"
            ),
        );
    }
}
