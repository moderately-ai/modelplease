// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use aws_sdk_bedrockruntime::{
    error::SdkError,
    operation::{converse::ConverseError, converse_stream::ConverseStreamError},
    types::{
        ContentBlock, ContentBlockDelta, ConverseStreamOutput, ReasoningContentBlock,
        ReasoningContentBlockDelta, TokenUsage,
    },
};

use crate::{
    error::LanguageModelError,
    response::{LanguageModelResponse, StreamDelta, Usage},
};

pub(super) fn bedrock_stop_reason(
    raw: &aws_sdk_bedrockruntime::types::StopReason,
) -> crate::StopReason {
    use aws_sdk_bedrockruntime::types::StopReason as Bedrock;
    match raw {
        Bedrock::EndTurn => crate::StopReason::EndTurn,
        Bedrock::MaxTokens => crate::StopReason::MaxTokens,
        Bedrock::StopSequence => crate::StopReason::StopSequence,
        Bedrock::ToolUse => crate::StopReason::ToolUse,
        Bedrock::ContentFiltered | Bedrock::GuardrailIntervened => crate::StopReason::ContentFilter,
        other => crate::StopReason::Other(other.as_str().to_owned()),
    }
}

/// Map Bedrock's `TokenUsage` into our [`Usage`], including the prompt
/// cache counters (`cacheWriteInputTokens` → creation, `cacheReadInputTokens`
/// → read). Both cache fields are optional on the wire and absent on
/// non-caching calls, so they saturate to `0`.
///
/// A `0` here doesn't necessarily mean the upstream isn't caching — Converse
/// may cache implicitly without surfacing the counter via the AWS SDK's
/// `TokenUsage` struct. The June 2026 sweep observed `cache_read_input_tokens
/// = 0` on every Converse call across 30 models even when the equivalent
/// Mantle call on the same model + prompt reported >2000 cached tokens.
/// Cross-check against [`crate::bedrock_mantle::BedrockMantleProvider`] when
/// reasoning about cost — the Mantle column is closer to the floor of actual
/// billing for repeat-prefix workloads.
pub(super) fn bedrock_usage(u: &TokenUsage) -> Usage {
    Usage {
        input_tokens: u64::try_from(u.input_tokens()).unwrap_or(0),
        output_tokens: u64::try_from(u.output_tokens()).unwrap_or(0),
        cache_creation_input_tokens: u
            .cache_write_input_tokens()
            .and_then(|v| u64::try_from(v).ok())
            .unwrap_or(0),
        cache_read_input_tokens: u
            .cache_read_input_tokens()
            .and_then(|v| u64::try_from(v).ok())
            .unwrap_or(0),
    }
}

pub(super) fn parse_converse_output(
    model: &str,
    output: Option<aws_sdk_bedrockruntime::types::ConverseOutput>,
    usage: Option<&TokenUsage>,
    stop_reason: Option<&aws_sdk_bedrockruntime::types::StopReason>,
) -> Result<LanguageModelResponse, LanguageModelError> {
    let Some(aws_sdk_bedrockruntime::types::ConverseOutput::Message(msg)) = output else {
        return Err(LanguageModelError::EmptyResponse);
    };
    let mut text_parts: Vec<String> = Vec::new();
    let mut reasoning_parts: Vec<String> = Vec::new();
    for block in msg.content() {
        match block {
            ContentBlock::Text(text) => text_parts.push(text.clone()),
            // Reasoning models (gpt-oss, DeepSeek R1, …) return chain-of-thought
            // in a ReasoningContent block; keep it out of `content` and surface
            // it as `thinking` rather than discarding it.
            ContentBlock::ReasoningContent(ReasoningContentBlock::ReasoningText(rt)) => {
                reasoning_parts.push(rt.text().to_owned());
            }
            _ => {}
        }
    }
    let had_text_block = !text_parts.is_empty();
    // gpt-oss can also inline its CoT in the text block, wrapped in
    // <reasoning>…</reasoning> before the answer (the documented InvokeModel
    // shape); split it out so `content` is just the answer.
    let (content, inline_reasoning) = split_inline_reasoning(&text_parts.join(""));
    if let Some(r) = inline_reasoning {
        reasoning_parts.push(r);
    }
    let thinking = (!reasoning_parts.is_empty()).then(|| reasoning_parts.join(""));

    if content.is_empty() {
        // The model produced reasoning but no answer. If it hit the token
        // ceiling mid-reasoning, say so actionably instead of a bare empty
        // response — this is the gpt-oss-on-a-tiny-budget failure mode.
        if thinking.is_some()
            && matches!(
                stop_reason,
                Some(aws_sdk_bedrockruntime::types::StopReason::MaxTokens)
            )
        {
            return Err(LanguageModelError::provider(format!(
                "model `{model}` exhausted max_tokens during reasoning before \
                 producing an answer; raise max_tokens or lower reasoning effort \
                 (reasoning: {{ mode: adaptive, effort: low }})"
            )));
        }
        // No content block at all (and not the reasoning-exhaustion case) is a
        // genuine empty response; an explicitly-empty text block keeps the
        // historical empty-content behaviour.
        if !had_text_block {
            return Err(LanguageModelError::EmptyResponse);
        }
    }
    Ok(LanguageModelResponse {
        content,
        thinking,
        usage: usage.map(bedrock_usage),
        model: Some(model.to_owned()),
        stop_reason: stop_reason.map(bedrock_stop_reason),
    })
}

/// Split a model's text output into `(answer, reasoning)` when it inlines
/// chain-of-thought as `<reasoning>…</reasoning>` ahead of the answer (the
/// shape AWS documents for OpenAI gpt-oss). With no `<reasoning>` marker the
/// text is returned verbatim (no trimming) and reasoning is `None`. An
/// unterminated `<reasoning>` (budget exhausted mid-thought) treats the
/// remainder as reasoning and yields an empty answer.
pub(super) fn split_inline_reasoning(text: &str) -> (String, Option<String>) {
    const OPEN: &str = "<reasoning>";
    const CLOSE: &str = "</reasoning>";
    let Some(open_idx) = text.find(OPEN) else {
        return (text.to_owned(), None);
    };
    let after_open = &text[open_idx + OPEN.len()..];
    let (reasoning, after_close) = after_open
        .find(CLOSE)
        .map_or((after_open, ""), |close_idx| {
            (
                &after_open[..close_idx],
                &after_open[close_idx + CLOSE.len()..],
            )
        });
    let answer = format!("{}{}", &text[..open_idx], after_close);
    (answer.trim().to_owned(), Some(reasoning.trim().to_owned()))
}

pub(super) fn convert_stream_event_to_delta(event: &ConverseStreamOutput) -> Option<StreamDelta> {
    match event {
        ConverseStreamOutput::ContentBlockDelta(d) => match d.delta() {
            Some(ContentBlockDelta::Text(text)) => Some(StreamDelta {
                content: text.clone(),
                thinking: None,
                usage: None,
                model: None,
                stop_reason: None,
                is_final: false,
            }),
            // Stream reasoning text alongside content so reasoning models
            // (gpt-oss, DeepSeek R1, Magistral, MiniMax M2, Nemotron-super)
            // produce observable output as soon as the CoT starts — without
            // this arm Converse appeared to "buffer" the entire reasoning
            // trace before any delta showed up. The sync `parse_converse_output`
            // already concatenates these into `LanguageModelResponse::thinking`;
            // this brings the streaming surface to parity.
            //
            // `RedactedContent` is encrypted by the provider for safety and
            // `Signature` is a verification token; neither carries human-
            // displayable text so we surface only `Text` here.
            Some(ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::Text(text))) => {
                Some(StreamDelta {
                    content: String::new(),
                    thinking: Some(text.clone()),
                    usage: None,
                    model: None,
                    stop_reason: None,
                    is_final: false,
                })
            }
            _ => None,
        },
        ConverseStreamOutput::MessageStop(s) => Some(StreamDelta {
            content: String::new(),
            thinking: None,
            usage: None,
            model: None,
            stop_reason: Some(bedrock_stop_reason(s.stop_reason())),
            // `Metadata` is the real stream terminator on Converse — it
            // arrives after `MessageStop` with the final usage payload, so
            // we keep `is_final: false` here and let the metadata arm flip
            // the terminator bit.
            is_final: false,
        }),
        ConverseStreamOutput::Metadata(m) => {
            let usage = m.usage().map(bedrock_usage);
            Some(StreamDelta {
                content: String::new(),
                thinking: None,
                usage,
                model: None,
                stop_reason: None,
                is_final: true,
            })
        }
        _ => None,
    }
}

/// Build the user-facing message for a Bedrock `ValidationException`.
///
/// A cross-region / global inference profile invoked from a source region
/// outside its geography surfaces as a plain `ValidationException`, not a
/// distinct error type. Add an actionable hint for that case so operators
/// don't read it as a malformed request; every other validation message
/// passes through verbatim.
fn validation_message(msg: &str) -> String {
    let lower = msg.to_ascii_lowercase();
    let region_profile_mismatch = lower.contains("inference profile")
        && (lower.contains("region") || lower.contains("supported"));
    if region_profile_mismatch {
        format!(
            "bedrock validation: {msg}. Hint: a cross-region inference profile \
             (e.g. a `us.` id) must be invoked from a source region inside its \
             geography, and latency-optimized inference requires such a profile — \
             check the application's AWS region against the profile's geography."
        )
    } else {
        format!("bedrock validation: {msg}")
    }
}

pub(super) fn map_converse_error(err: SdkError<ConverseError>) -> LanguageModelError {
    match err {
        SdkError::ServiceError(svc) => {
            match svc.into_err() {
                ConverseError::ThrottlingException(e) => LanguageModelError::rate_limited(
                    e.message().unwrap_or("bedrock throttled").to_owned(),
                ),
                ConverseError::AccessDeniedException(e) => LanguageModelError::authentication(
                    e.message().unwrap_or("bedrock access denied").to_owned(),
                ),
                ConverseError::ResourceNotFoundException(e) => LanguageModelError::provider(
                    format!("bedrock resource not found: {}", e.message().unwrap_or("")),
                ),
                ConverseError::ValidationException(e) => {
                    LanguageModelError::provider(validation_message(e.message().unwrap_or("")))
                }
                ConverseError::ModelTimeoutException(e) => LanguageModelError::provider(format!(
                    "bedrock model timeout: {}",
                    e.message().unwrap_or("")
                )),
                ConverseError::ModelNotReadyException(e) => LanguageModelError::provider(format!(
                    "bedrock model not ready: {}",
                    e.message().unwrap_or("")
                )),
                ConverseError::ServiceUnavailableException(e) => LanguageModelError::provider(
                    format!("bedrock service unavailable: {}", e.message().unwrap_or("")),
                ),
                ConverseError::InternalServerException(e) => LanguageModelError::provider(format!(
                    "bedrock internal: {}",
                    e.message().unwrap_or("")
                )),
                ConverseError::ModelErrorException(e) => LanguageModelError::provider(format!(
                    "bedrock model error: {}",
                    e.message().unwrap_or("")
                )),
                other => LanguageModelError::provider(format!("bedrock: {other:?}")),
            }
        }
        SdkError::TimeoutError(_) => LanguageModelError::provider("bedrock: timeout".to_owned()),
        SdkError::DispatchFailure(d) => {
            LanguageModelError::provider(format!("bedrock dispatch: {d:?}"))
        }
        SdkError::ResponseError(r) => {
            LanguageModelError::provider(format!("bedrock response: {r:?}"))
        }
        SdkError::ConstructionFailure(c) => {
            LanguageModelError::provider(format!("bedrock construction: {c:?}"))
        }
        other => LanguageModelError::provider(format!("bedrock: {other:?}")),
    }
}

pub(super) fn map_converse_stream_error(err: SdkError<ConverseStreamError>) -> LanguageModelError {
    match err {
        SdkError::ServiceError(svc) => match svc.into_err() {
            ConverseStreamError::ThrottlingException(e) => LanguageModelError::rate_limited(
                e.message().unwrap_or("bedrock throttled").to_owned(),
            ),
            ConverseStreamError::AccessDeniedException(e) => LanguageModelError::authentication(
                e.message().unwrap_or("bedrock access denied").to_owned(),
            ),
            ConverseStreamError::ResourceNotFoundException(e) => LanguageModelError::provider(
                format!("bedrock resource not found: {}", e.message().unwrap_or("")),
            ),
            ConverseStreamError::ValidationException(e) => {
                LanguageModelError::provider(validation_message(e.message().unwrap_or("")))
            }
            ConverseStreamError::ModelStreamErrorException(e) => LanguageModelError::provider(
                format!("bedrock stream error: {}", e.message().unwrap_or("")),
            ),
            ConverseStreamError::ModelTimeoutException(e) => LanguageModelError::provider(format!(
                "bedrock model timeout: {}",
                e.message().unwrap_or("")
            )),
            ConverseStreamError::ModelNotReadyException(e) => LanguageModelError::provider(
                format!("bedrock model not ready: {}", e.message().unwrap_or("")),
            ),
            ConverseStreamError::ServiceUnavailableException(e) => LanguageModelError::provider(
                format!("bedrock service unavailable: {}", e.message().unwrap_or("")),
            ),
            ConverseStreamError::InternalServerException(e) => LanguageModelError::provider(
                format!("bedrock internal: {}", e.message().unwrap_or("")),
            ),
            ConverseStreamError::ModelErrorException(e) => LanguageModelError::provider(format!(
                "bedrock model error: {}",
                e.message().unwrap_or("")
            )),
            other => LanguageModelError::provider(format!("bedrock: {other:?}")),
        },
        SdkError::TimeoutError(_) => LanguageModelError::provider("bedrock: timeout".to_owned()),
        SdkError::DispatchFailure(d) => {
            LanguageModelError::provider(format!("bedrock dispatch: {d:?}"))
        }
        SdkError::ResponseError(r) => {
            LanguageModelError::provider(format!("bedrock response: {r:?}"))
        }
        SdkError::ConstructionFailure(c) => {
            LanguageModelError::provider(format!("bedrock construction: {c:?}"))
        }
        other => LanguageModelError::provider(format!("bedrock: {other:?}")),
    }
}
