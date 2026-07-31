// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Language model response types.

/// Normalized reason the model stopped generating.
///
/// Maps the per-provider stop-reason strings into one shared enum so
/// downstream consumers can distinguish
/// "ran out of room" (`MaxTokens`) from "finished naturally" (`EndTurn`)
/// without knowing which provider produced the response. Without this,
/// a truncated completion might otherwise surface as a parse error with no
/// hint that the model ran out of `max_tokens`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// Model finished generating naturally. Anthropic `end_turn`,
    /// OpenAI `stop`, Bedrock `end_turn`.
    EndTurn,
    /// Generation hit the `max_tokens` budget and was truncated.
    /// Anthropic `max_tokens`, OpenAI `length`, Bedrock `max_tokens`.
    MaxTokens,
    /// A configured stop sequence appeared in the output. Anthropic
    /// `stop_sequence`, OpenAI `stop` with a sequence match, Bedrock
    /// `stop_sequence`.
    StopSequence,
    /// Model emitted a tool-use block as its terminal action. Anthropic
    /// `tool_use`, Bedrock `tool_use`. OpenAI uses `tool_calls`.
    ToolUse,
    /// Provider-side safety / content filter aborted generation.
    /// OpenAI `content_filter`, Bedrock `content_filtered` or
    /// `guardrail_intervened`.
    ContentFilter,
    /// An upstream-specific stop-reason value we don't normalize.
    /// Carries the raw string so logs and downstream parsing can still
    /// see what the provider sent.
    Other(String),
}

/// A response from a language model generation call.
#[derive(Debug, Clone, Default)]
pub struct LanguageModelResponse {
    /// The generated text content.
    pub content: String,
    /// Extended-thinking reasoning blocks, concatenated in order.
    ///
    /// Populated for Anthropic Claude 4.x models that emit "thinking"
    /// content blocks ahead of the final text. Empty when the provider
    /// doesn't produce thinking traces. Always billable even when empty
    /// — the token count for thinking lives in `usage.output_tokens`.
    pub thinking: Option<String>,
    /// Token usage statistics, if available.
    pub usage: Option<Usage>,
    /// The model that generated this response, if reported.
    pub model: Option<String>,
    /// Normalized reason generation stopped. `None` only when the
    /// provider didn't include one on the response (older OpenAI-compat
    /// servers, in-progress stream snapshots). Production providers
    /// always populate this; downstream code that needs to distinguish
    /// truncation from natural completion treats `None` as "unknown,
    /// fall through to existing parse logic".
    pub stop_reason: Option<StopReason>,
}

/// An incremental chunk from a streaming language model response.
#[derive(Debug, Clone)]
pub struct StreamDelta {
    /// Incremental text content (may be empty on non-content events).
    pub content: String,
    /// Extended-thinking / reasoning text emitted on this chunk. Streamed
    /// alongside (or before) `content` on providers that expose
    /// chain-of-thought as a distinct content block:
    ///
    /// - **Bedrock Converse:** populated from `ContentBlockDelta::Reasoning` events (gpt-oss,
    ///   DeepSeek R1, Magistral, MiniMax M2, Nemotron-super on the Runtime endpoint).
    ///
    /// `None` on the OpenAI Chat Completions surface (reasoning isn't
    /// streamed per OpenAI spec — it shows up only in
    /// `completion_tokens_details.reasoning_tokens` accounting), and on
    /// adapters that haven't been extended to surface reasoning yet
    /// (Anthropic thinking blocks, Mantle Responses reasoning events).
    pub thinking: Option<String>,
    /// Token usage statistics. Presence and timing depend on the provider:
    ///
    /// - **Anthropic:** populated on `message_delta` events (mid-stream, sometimes on multiple
    ///   events as token counts evolve). The final `message_stop` event carries `None`.
    /// - **OpenAI / Ollama:** populated only on the final pre-`[DONE]` chunk, because we set
    ///   `stream_options.include_usage = true` on the request. Without that flag the field would
    ///   be `None` for every chunk.
    /// - **Default `generate_stream` impl** (non-streaming providers wrapped into a one-item
    ///   stream): populated on the single delta which is also `is_final`.
    ///
    /// Consumers that need cost tracking should accumulate or take the
    /// last non-`None` value rather than expect `usage` on every chunk.
    pub usage: Option<Usage>,
    /// The model name — typically only present on the first chunk.
    pub model: Option<String>,
    /// Normalized reason generation stopped — populated on whichever
    /// chunk carries the wire-level stop signal:
    ///
    /// - **OpenAI Chat Completions:** the final content chunk with a non-null `finish_reason`.
    /// - **Anthropic Messages:** the `message_delta` event (which also carries final usage).
    /// - **Bedrock Converse:** the `MessageStop` event (precedes the terminating `Metadata` chunk
    ///   that carries usage).
    /// - **Mantle Responses:** the `response.completed` event (also carries usage).
    ///   `response.incomplete` events with recoverable reasons (`max_output_tokens`,
    ///   `content_filter`) surface their stop reason here too instead of being raised as errors.
    ///
    /// `None` on intermediate content deltas and on providers that
    /// don't surface this on streams. Downstream consumers comparing
    /// against `max_tokens` should prefer this field when present rather
    /// than inferring truncation from output-token counts.
    pub stop_reason: Option<StopReason>,
    /// Whether this is the final chunk in the stream.
    pub is_final: bool,
}

/// Token usage statistics for a generation call.
///
/// Under prompt caching the provider splits the prompt: `input_tokens`
/// counts only the *uncached* remainder, while the cached prefix is
/// reported separately in `cache_creation_input_tokens` (a cache write
/// on a miss) and `cache_read_input_tokens` (a cache hit). The full
/// prompt size is therefore `input_tokens + cache_creation_input_tokens
/// + cache_read_input_tokens` — summing only `input_tokens` undercounts
/// once caching engages. Providers that don't report caching leave both
/// cache fields `0`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Usage {
    /// Number of uncached tokens in the input/prompt.
    pub input_tokens: u64,
    /// Number of tokens in the generated output.
    pub output_tokens: u64,
    /// Tokens written to the prompt cache on a miss (Anthropic
    /// `cache_creation_input_tokens` / Bedrock `cacheWriteInputTokens`).
    /// `0` when caching was not engaged or not supported.
    pub cache_creation_input_tokens: u64,
    /// Tokens served from the prompt cache on a hit (Anthropic
    /// `cache_read_input_tokens` / Bedrock `cacheReadInputTokens`, OpenAI
    /// `prompt_tokens_details.cached_tokens`). `0` on a cold call.
    pub cache_read_input_tokens: u64,
}
