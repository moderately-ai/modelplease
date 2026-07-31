// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use super::{
    BedrockMantleAuth, BedrockMantleProvider, BedrockMantleProviderConfig,
    BedrockMantleProviderDeps, Route,
    responses_wire::{
        ResponsesResponse, build_responses_request, convert_responses_sse_event,
        parse_responses_response,
    },
};
use crate::{
    capabilities::{MediaKind, ReasoningMode},
    config::{LanguageModelConfig, ReasoningConfig, ReasoningEffort},
    error::LanguageModelError,
    identifiers::ModelId,
    media::SourceKind,
    message::Message,
    provider::LanguageModelProvider,
    response::StopReason,
    retry::RetryConfig,
    sse::SseEvent,
};

#[test]
fn route_dispatches_gpt5_to_responses() {
    assert_eq!(Route::from_model_id("openai.gpt-5.5"), Route::Responses);
    assert_eq!(Route::from_model_id("openai.gpt-5.4"), Route::Responses);
    assert_eq!(
        Route::from_model_id("openai.gpt-5.5-2026-04-23"),
        Route::Responses
    );
    assert_eq!(
        Route::from_model_id("openai.gpt-5.4-2026-03-05"),
        Route::Responses
    );
}

#[test]
fn route_dispatches_anthropic_to_messages() {
    assert_eq!(
        Route::from_model_id("anthropic.claude-haiku-4-5"),
        Route::Messages
    );
    assert_eq!(
        Route::from_model_id("anthropic.claude-opus-4-8"),
        Route::Messages
    );
    assert_eq!(
        Route::from_model_id("anthropic.claude-mythos-preview"),
        Route::Messages
    );
}

#[test]
fn route_dispatches_everything_else_to_chat_completions() {
    // Bedrock Mantle live catalog (2026-06-04) — every other family routes through
    // Chat Completions, including OpenAI's open-weight `gpt-oss-*` (which also accepts
    // Responses but we pick the Chat path for code reuse with the existing OpenAI provider).
    assert_eq!(
        Route::from_model_id("openai.gpt-oss-120b"),
        Route::ChatCompletions
    );
    assert_eq!(
        Route::from_model_id("openai.gpt-oss-20b"),
        Route::ChatCompletions
    );
    assert_eq!(
        Route::from_model_id("openai.gpt-oss-safeguard-120b"),
        Route::ChatCompletions
    );
    assert_eq!(
        Route::from_model_id("deepseek.v3.2"),
        Route::ChatCompletions
    );
    assert_eq!(
        Route::from_model_id("mistral.ministral-3-8b-instruct"),
        Route::ChatCompletions
    );
    assert_eq!(
        Route::from_model_id("qwen.qwen3-vl-235b-a22b-instruct"),
        Route::ChatCompletions
    );
    assert_eq!(
        Route::from_model_id("google.gemma-3-27b-it"),
        Route::ChatCompletions
    );
    assert_eq!(Route::from_model_id("zai.glm-5"), Route::ChatCompletions);
}

fn make_provider() -> BedrockMantleProvider {
    BedrockMantleProvider::new(
        BedrockMantleProviderDeps {
            client: Arc::new(reqwest::Client::new()),
            auth: BedrockMantleAuth::ApiKey("test-key".into()),
        },
        BedrockMantleProviderConfig {
            default_region: BedrockMantleProviderConfig::DEFAULT_REGION.into(),
            openai_gpt5_region: BedrockMantleProviderConfig::DEFAULT_OPENAI_GPT5_REGION.into(),
            anthropic_region: BedrockMantleProviderConfig::DEFAULT_ANTHROPIC_REGION.into(),
            retry_config: RetryConfig::default(),
        },
    )
}

#[test]
fn endpoint_base_uses_per_region_host() {
    assert_eq!(
        BedrockMantleProvider::endpoint_base("us-east-2"),
        "https://bedrock-mantle.us-east-2.api.aws"
    );
    assert_eq!(
        BedrockMantleProvider::endpoint_base("eu-west-1"),
        "https://bedrock-mantle.eu-west-1.api.aws"
    );
}

#[test]
fn region_for_uses_per_family_config() {
    let provider = make_provider();
    assert_eq!(provider.region_for(Route::ChatCompletions), "us-west-2");
    assert_eq!(provider.region_for(Route::Responses), "us-east-2");
    assert_eq!(provider.region_for(Route::Messages), "us-east-1");
}

#[test]
fn name_is_bedrock_mantle() {
    let provider = make_provider();
    assert_eq!(provider.name(), "bedrock-mantle");
}

#[test]
fn capabilities_returns_some_for_table_models() {
    let provider = make_provider();
    let caps = provider.capabilities(&ModelId::new("openai.gpt-5.5"));
    assert!(caps.is_some(), "gpt-5.5 should be in cap table");
    let caps = caps.unwrap();
    let reasoning = caps.reasoning.expect("gpt-5.5 declares reasoning");
    assert!(reasoning.supported_efforts.contains(ReasoningEffort::XHigh));
    // Latency-optimized is bedrock-runtime only, never Mantle.
    assert!(!caps.latency_optimized_supported);
    assert!(!caps.extended_cache_ttl_supported);
}

#[test]
fn capabilities_returns_none_for_unknown_model() {
    let provider = make_provider();
    assert!(
        provider
            .capabilities(&ModelId::new("not-a-real-mantle-model"))
            .is_none()
    );
}

#[test]
fn deepseek_v3_2_marked_non_reasoning() {
    // V3.2 is the non-reasoning DeepSeek variant — the table reflects that so callers can't
    // accidentally request reasoning effort against it.
    let provider = make_provider();
    let caps = provider
        .capabilities(&ModelId::new("deepseek.v3.2"))
        .unwrap();
    assert!(caps.reasoning.is_none());
}

#[test]
fn anthropic_on_mantle_reasoning_modes_track_generation() {
    let provider = make_provider();

    // Haiku 4.5 is manual-only on Mantle — adaptive thinking / the effort
    // parameter 400 on Haiku — and still accepts sampling with thinking off.
    let haiku = provider
        .capabilities(&ModelId::new("anthropic.claude-haiku-4-5"))
        .unwrap()
        .reasoning
        .expect("haiku declares reasoning");
    assert!(haiku.supported_modes.contains(ReasoningMode::Manual));
    assert!(!haiku.supported_modes.contains(ReasoningMode::Adaptive));
    assert!(!haiku.sampling_params_removed);
    // Reasoning-on conflicts still apply for the manual surface.
    assert!(haiku.conflicts.temperature_forbidden);
    assert!(haiku.conflicts.top_k_forbidden);

    // Opus 4.7/4.8 + Mythos are the 4.7+ generation: adaptive-only (manual 400s),
    // `xhigh` effort, and sampling params removed entirely.
    for id in [
        "anthropic.claude-opus-4-7",
        "anthropic.claude-opus-4-8",
        "anthropic.claude-mythos-preview",
    ] {
        let r = provider
            .capabilities(&ModelId::new(id))
            .unwrap()
            .reasoning
            .unwrap_or_else(|| panic!("{id} declares reasoning"));
        assert!(
            r.supported_modes.contains(ReasoningMode::Adaptive),
            "{id} adaptive"
        );
        assert!(
            !r.supported_modes.contains(ReasoningMode::Manual),
            "{id} no manual"
        );
        assert!(
            r.supported_efforts.contains(ReasoningEffort::XHigh),
            "{id} xhigh"
        );
        assert!(r.sampling_params_removed, "{id} sampling removed");
    }
}

#[test]
fn vision_models_declare_image_media_support() {
    let provider = make_provider();
    for id in [
        "qwen.qwen3-vl-235b-a22b-instruct",
        "writer.palmyra-vision-7b",
        "google.gemma-3-27b-it",
        "nvidia.nemotron-nano-12b-v2",
    ] {
        let caps = provider
            .capabilities(&ModelId::new(id))
            .unwrap_or_else(|| panic!("{id} missing from cap table"));
        let image = caps
            .media_support
            .get(&MediaKind::Image)
            .unwrap_or_else(|| panic!("{id} should declare Image media support"));
        assert!(
            image.sources.contains(SourceKind::Url),
            "{id} should accept image URLs"
        );
        assert!(
            image.sources.contains(SourceKind::InlineBytes),
            "{id} should accept inline image bytes",
        );
        assert!(image.formats.contains(&"png"));
    }
}

#[test]
fn voxtral_declares_audio_media_support() {
    let provider = make_provider();
    for id in [
        "mistral.voxtral-mini-3b-2507",
        "mistral.voxtral-small-24b-2507",
    ] {
        let caps = provider
            .capabilities(&ModelId::new(id))
            .unwrap_or_else(|| panic!("{id} missing from cap table"));
        let audio = caps
            .media_support
            .get(&MediaKind::Audio)
            .unwrap_or_else(|| panic!("{id} should declare Audio media support"));
        assert!(audio.sources.contains(SourceKind::InlineBytes));
        assert!(audio.formats.contains(&"wav"));
        assert!(audio.formats.contains(&"mp3"));
    }
}

// --- Responses adapter (Phase 6) ---

fn responses_sse(event_type: &str, data: &str) -> SseEvent {
    SseEvent {
        event_type: event_type.to_owned(),
        data: data.to_owned(),
        id: String::new(),
    }
}

#[test]
fn build_responses_request_extracts_system_into_instructions() {
    let config = LanguageModelConfig::default();
    let messages = vec![Message::system("You are concise."), Message::user("Hi")];
    let req = build_responses_request("openai.gpt-5.5", &messages, &config, false);
    assert_eq!(req.instructions.as_deref(), Some("You are concise."));
    // input array should not contain the system message.
    assert_eq!(req.input.len(), 1);
    assert_eq!(req.input[0].role, "user");
}

#[test]
fn build_responses_request_emits_reasoning_and_strips_temperature() {
    let config = LanguageModelConfig {
        temperature: Some(0.7),
        reasoning: Some(ReasoningConfig::Adaptive {
            effort: ReasoningEffort::High,
        }),
        ..Default::default()
    };
    let req = build_responses_request("openai.gpt-5.5", &[Message::user("hi")], &config, false);
    // Reasoning emitted with effort.
    assert_eq!(req.reasoning.as_ref().map(|r| r.effort), Some("high"));
    // Temperature stripped — gpt-5 rejects it whenever reasoning.effort is set.
    assert!(req.temperature.is_none());
}

#[test]
fn build_responses_request_no_reasoning_preserves_temperature() {
    let config = LanguageModelConfig {
        temperature: Some(0.7),
        ..Default::default()
    };
    let req = build_responses_request("openai.gpt-5.5", &[Message::user("hi")], &config, false);
    assert!(req.reasoning.is_none());
    assert_eq!(req.temperature, Some(0.7));
}

#[test]
fn build_responses_request_always_sets_store_false() {
    let config = LanguageModelConfig::default();
    let req = build_responses_request("openai.gpt-5.5", &[Message::user("hi")], &config, false);
    let wire = serde_json::to_value(&req).unwrap();
    assert_eq!(wire["store"], false);
}

#[test]
fn build_responses_request_sets_stream_when_requested() {
    let config = LanguageModelConfig::default();
    let req = build_responses_request("openai.gpt-5.5", &[Message::user("hi")], &config, true);
    assert_eq!(req.stream, Some(true));
}

#[test]
fn parse_responses_response_extracts_message_text() {
    let api = serde_json::from_value::<ResponsesResponse>(serde_json::json!({
        "model": "openai.gpt-5.5",
        "status": "completed",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "Paris"}]}
        ],
        "usage": {"input_tokens": 7, "output_tokens": 1}
    }))
    .unwrap();
    let result = parse_responses_response(api).unwrap();
    assert_eq!(result.content, "Paris");
    assert_eq!(result.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(result.usage.unwrap().input_tokens, 7);
}

#[test]
fn parse_responses_response_captures_reasoning_separately() {
    let api = serde_json::from_value::<ResponsesResponse>(serde_json::json!({
        "model": "openai.gpt-5.5",
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": [{"text": "thinking step 1"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "result"}]}
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1}
    }))
    .unwrap();
    let result = parse_responses_response(api).unwrap();
    assert_eq!(result.content, "result");
    assert_eq!(result.thinking.as_deref(), Some("thinking step 1"));
}

#[test]
fn parse_responses_response_incomplete_max_tokens_maps_to_max_tokens_stop() {
    let api = serde_json::from_value::<ResponsesResponse>(serde_json::json!({
        "model": "openai.gpt-5.5",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "truncated..."}]}
        ],
        "usage": {"input_tokens": 1, "output_tokens": 256}
    }))
    .unwrap();
    let result = parse_responses_response(api).unwrap();
    assert_eq!(result.stop_reason, Some(StopReason::MaxTokens));
}

#[test]
fn parse_responses_response_empty_content_yields_empty_response() {
    let api = serde_json::from_value::<ResponsesResponse>(serde_json::json!({
        "model": "openai.gpt-5.5",
        "status": "completed",
        "output": [{"type": "reasoning", "summary": [{"text": "thought only"}]}],
        "usage": {"input_tokens": 1, "output_tokens": 1}
    }))
    .unwrap();
    let err = parse_responses_response(api).unwrap_err();
    assert!(matches!(err, LanguageModelError::EmptyResponse));
}

#[test]
fn parse_responses_response_subtracts_cached_from_input_tokens() {
    let api = serde_json::from_value::<ResponsesResponse>(serde_json::json!({
        "model": "openai.gpt-5.5",
        "status": "completed",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "x"}]}
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 10,
            "input_tokens_details": {"cached_tokens": 60}
        }
    }))
    .unwrap();
    let result = parse_responses_response(api).unwrap();
    let usage = result.usage.unwrap();
    assert_eq!(usage.input_tokens, 40, "uncached prompt portion");
    assert_eq!(usage.cache_read_input_tokens, 60);
}

#[test]
fn convert_responses_sse_event_output_text_delta() {
    let event = responses_sse(
        "response.output_text.delta",
        r#"{"type":"response.output_text.delta","delta":"Hello"}"#,
    );
    let out = convert_responses_sse_event(Ok(event)).unwrap().unwrap();
    assert_eq!(out.content, "Hello");
    assert!(!out.is_final);
}

#[test]
fn convert_responses_sse_event_empty_delta_skipped() {
    let event = responses_sse(
        "response.output_text.delta",
        r#"{"type":"response.output_text.delta","delta":""}"#,
    );
    assert!(convert_responses_sse_event(Ok(event)).is_none());
}

#[test]
fn convert_responses_sse_event_completed_carries_usage_and_is_final() {
    let event = responses_sse(
        "response.completed",
        r#"{"type":"response.completed","response":{"model":"openai.gpt-5.5","status":"completed","usage":{"input_tokens":11,"output_tokens":22}}}"#,
    );
    let out = convert_responses_sse_event(Ok(event)).unwrap().unwrap();
    assert!(out.is_final);
    let usage = out.usage.unwrap();
    assert_eq!(usage.input_tokens, 11);
    assert_eq!(usage.output_tokens, 22);
    assert_eq!(out.model.as_deref(), Some("openai.gpt-5.5"));
    assert_eq!(out.stop_reason, Some(StopReason::EndTurn));
}

#[test]
fn convert_responses_sse_event_incomplete_max_tokens_is_successful_final() {
    // Pre-fix this returned an error, contradicting the sync path's
    // treatment of incomplete + max_output_tokens as a successful
    // response with StopReason::MaxTokens. Now the streaming surface
    // agrees: emit a usable final StreamDelta with the right stop_reason.
    let event = responses_sse(
        "response.incomplete",
        r#"{"type":"response.incomplete","response":{"model":"openai.gpt-5.5","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"usage":{"input_tokens":11,"output_tokens":22}}}"#,
    );
    let out = convert_responses_sse_event(Ok(event)).unwrap().unwrap();
    assert!(out.is_final, "max-tokens hit is still the terminator");
    assert_eq!(out.stop_reason, Some(StopReason::MaxTokens));
    let usage = out.usage.unwrap();
    assert_eq!(usage.output_tokens, 22);
}

#[test]
fn convert_responses_sse_event_incomplete_content_filter_is_successful_final() {
    // Content-filter trips are also surfaced (not errored) so callers
    // can decide whether to retry with different inputs.
    let event = responses_sse(
        "response.incomplete",
        r#"{"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"content_filter"}}}"#,
    );
    let out = convert_responses_sse_event(Ok(event)).unwrap().unwrap();
    assert!(out.is_final);
    assert_eq!(out.stop_reason, Some(StopReason::ContentFilter));
}

#[test]
fn convert_responses_sse_event_failed_still_yields_error() {
    // `response.failed` (as opposed to `response.incomplete`) remains
    // a genuine error — the run is unrecoverable.
    let event = responses_sse(
        "response.failed",
        r#"{"type":"response.failed","response":{"error":{"message":"safety filter triggered"}}}"#,
    );
    let out = convert_responses_sse_event(Ok(event)).unwrap();
    match out {
        Err(LanguageModelError::Provider { message }) => {
            assert!(message.contains("safety filter"));
        }
        other => panic!("expected Provider error, got {other:?}"),
    }
}

#[test]
fn convert_responses_sse_event_unknown_event_silently_skipped() {
    let event = responses_sse("response.in_progress", "{}");
    assert!(convert_responses_sse_event(Ok(event)).is_none());
}

#[test]
fn text_only_models_have_no_media_support() {
    // Sanity-check a few text-only models so a future paste-error doesn't accidentally turn
    // them into multimodal ones.
    let provider = make_provider();
    for id in [
        "deepseek.v3.2",
        "mistral.mistral-large-3-675b-instruct",
        "openai.gpt-oss-120b",
        "moonshotai.kimi-k2.5",
    ] {
        let caps = provider
            .capabilities(&ModelId::new(id))
            .unwrap_or_else(|| panic!("{id} missing from cap table"));
        assert!(caps.media_support.is_empty(), "{id} should be text-only");
    }
}
