// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use aws_sdk_bedrockruntime::types::{
    CacheTtl as BedrockCacheTtl, ContentBlock, ContentBlockDelta, ContentBlockDeltaEvent,
    ConversationRole, ConverseOutput as BedrockConverseOutput, ConverseStreamMetadataEvent,
    ConverseStreamOutput, DocumentFormat, ImageFormat, ImageSource, Message as BedrockMessage,
    MessageStartEvent, MessageStopEvent, OutputFormatStructure, OutputFormatType,
    PerformanceConfigLatency, ReasoningContentBlock, ReasoningContentBlockDelta,
    SystemContentBlock, VideoFormat, builders::TokenUsageBuilder,
};
use rustc_hash::FxHashMap;

use super::{
    capabilities::{ChatFeature, MODEL_CAPABILITIES, bedrock_media_support},
    request::{
        bedrock_additional_request_fields, build_bedrock_messages, build_inference_config,
        build_output_config, build_performance_config, cache_point_block,
        foundation_id_from_model_arn, strip_region_prefix, supports_prompt_caching_with,
        translate_part_for_bedrock,
    },
    response::{
        bedrock_usage, convert_stream_event_to_delta, parse_converse_output, split_inline_reasoning,
    },
};
use crate::{
    capabilities::ReasoningMode,
    config::{
        CacheTtl, LanguageModelConfig, LatencyMode, ReasoningConfig, ReasoningEffort,
        ResponseFormat,
    },
    error::LanguageModelError,
    media::MediaSource,
    message::{ContentPart, Message, Role},
};

#[test]
fn build_bedrock_messages_extracts_system() {
    let messages = vec![
        Message::system("be concise"),
        Message::user("hi"),
        Message::assistant("hello"),
    ];
    let (system, out) = build_bedrock_messages(&messages, false, CacheTtl::FiveMin).unwrap();
    assert_eq!(system.len(), 1);
    match &system[0] {
        SystemContentBlock::Text(t) => assert_eq!(t, "be concise"),
        _ => panic!("expected text system block"),
    }
    assert_eq!(out.len(), 2);
    assert!(matches!(out[0].role, ConversationRole::User));
    assert!(matches!(out[1].role, ConversationRole::Assistant));
}

#[test]
fn build_bedrock_messages_drops_image_in_system() {
    let messages = vec![Message::with_parts(
        Role::System,
        vec![
            ContentPart::text("be concise"),
            ContentPart::image(MediaSource::Url {
                url: crate::HttpsUrl::parse("https://example.com/img.png").unwrap(),
            }),
        ],
    )];
    let (system, out) = build_bedrock_messages(&messages, false, CacheTtl::FiveMin).unwrap();
    assert_eq!(system.len(), 1);
    assert!(out.is_empty());
}

#[test]
fn build_bedrock_messages_rejects_empty_user() {
    let messages = vec![Message::with_parts(
        Role::User,
        vec![ContentPart::image(MediaSource::Url {
            url: crate::HttpsUrl::parse("https://example.com/img.png").unwrap(),
        })],
    )];
    let result = build_bedrock_messages(&messages, false, CacheTtl::FiveMin);
    assert!(matches!(result, Err(LanguageModelError::Provider { .. })));
}

#[test]
fn supports_prompt_caching_gates_by_family() {
    // Claude + Nova (raw and region-prefixed) cache; everything else doesn't.
    // No ARN resolution involved, so an empty map exercises the
    // strip-prefix + family path the provider method delegates to.
    let none = FxHashMap::default();
    let supports = |id: &str| supports_prompt_caching_with(&none, id);
    assert!(supports("anthropic.claude-haiku-4-5-20251001-v1:0"));
    assert!(supports("us.anthropic.claude-sonnet-4-5-v1:0"));
    assert!(supports("global.anthropic.claude-haiku-4-5-20251001-v1:0"));
    assert!(supports("global.anthropic.claude-sonnet-4-5-v1:0"));
    assert!(supports("jp.anthropic.claude-haiku-4-5-20251001-v1:0"));
    assert!(supports("au.anthropic.claude-haiku-4-5-20251001-v1:0"));
    assert!(supports("anthropic.claude-sonnet-5"));
    assert!(supports("us.anthropic.claude-sonnet-5"));
    assert!(supports("amazon.nova-lite-v1:0"));
    // Claude 3.5 (note the `-3-5-`) caches; the original Claude 3 (2024)
    // models do NOT — they predate Bedrock prompt caching.
    assert!(supports("us.anthropic.claude-3-5-sonnet-20241022-v2:0"));
    assert!(!supports("us.anthropic.claude-3-haiku-20240307-v1:0"));
    assert!(!supports("anthropic.claude-3-sonnet-20240229-v1:0"));
    assert!(!supports("anthropic.claude-3-opus-20240229-v1:0"));
    assert!(!supports("meta.llama3-3-70b-instruct-v1:0"));
    assert!(!supports("openai.gpt-oss-120b-1:0"));
    assert!(!supports("moonshotai.kimi-k2.5"));
    assert!(!supports("zai.glm-5"));
    assert!(!supports("mistral.mistral-large-3-675b-instruct"));
    assert!(!supports("deepseek.r1-v1:0"));
    assert!(!supports("amazon.titan-text-express-v1"));
}

#[test]
fn cache_point_block_sets_ttl_only_for_one_hour() {
    // The 5-minute default sends no explicit ttl (Bedrock's implicit
    // default) so non-Anthropic models accept it; 1-hour sets the field.
    let five = cache_point_block(CacheTtl::FiveMin).unwrap();
    assert!(
        five.ttl().is_none(),
        "5-min cachePoint must not carry an explicit ttl"
    );
    let hour = cache_point_block(CacheTtl::OneHour).unwrap();
    assert_eq!(hour.ttl(), Some(&BedrockCacheTtl::OneHour));
}

#[test]
fn extended_cache_ttl_allow_list_is_claude_4_5_plus_only() {
    let extended = |id: &str| {
        MODEL_CAPABILITIES[id]
            .features
            .contains(ChatFeature::ExtendedCacheTtl)
    };
    assert!(extended("anthropic.claude-haiku-4-5-20251001-v1:0"));
    assert!(extended("anthropic.claude-sonnet-4-6"));
    assert!(extended("anthropic.claude-sonnet-5"));
    // Older Opus 4.1 and every Nova model do not support the 1-hour tier.
    assert!(!extended("anthropic.claude-opus-4-1-20250805-v1:0"));
    assert!(!extended("amazon.nova-pro-v1:0"));
}

#[test]
fn claude_sonnet_5_reasoning_is_adaptive_only() {
    // Sonnet 5 dropped Sonnet 4.6's transitional manual budget: adaptive is the
    // only mode (manual `budget_tokens` 400s on the wire), and it is the first
    // Sonnet-tier model to advertise the `xhigh` effort level.
    let reasoning = MODEL_CAPABILITIES["anthropic.claude-sonnet-5"]
        .reasoning
        .as_ref()
        .expect("sonnet 5 advertises a reasoning surface");
    assert!(reasoning.supported_modes.contains(ReasoningMode::Adaptive));
    assert!(!reasoning.supported_modes.contains(ReasoningMode::Manual));
    assert!(reasoning.manual_budget_range.is_none());
    assert!(reasoning.supported_efforts.contains(ReasoningEffort::XHigh));
    assert!(reasoning.supported_efforts.contains(ReasoningEffort::Max));
    // 4.7+ generation removed sampling params entirely — rejected even with
    // thinking off (the caller suppresses temperature / fails loud on it).
    assert!(reasoning.sampling_params_removed);
}

#[test]
fn claude_sampling_params_removed_tracks_4_7_plus_generation() {
    // Sonnet 5 / Opus 4.7 removed temperature/top_p/top_k entirely (400 in every
    // request); the 4.6 generation and Haiku still accept them with thinking off.
    let removed = |id: &str| {
        MODEL_CAPABILITIES[id]
            .reasoning
            .as_ref()
            .is_some_and(|r| r.sampling_params_removed)
    };
    assert!(removed("anthropic.claude-sonnet-5"));
    assert!(removed("anthropic.claude-opus-4-7"));
    assert!(!removed("anthropic.claude-sonnet-4-6"));
    assert!(!removed("anthropic.claude-opus-4-6-v1"));
    assert!(!removed("anthropic.claude-haiku-4-5-20251001-v1:0"));
}

#[test]
fn supports_prompt_caching_resolves_application_profile_arn() {
    // An application-profile ARN carries no family in its string, so the
    // decision rides on the resolved foundation id. A profile wrapping
    // Claude caches; one wrapping Llama doesn't.
    let claude_arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc";
    let llama_arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/def";
    let mut resolved = FxHashMap::default();
    resolved.insert(
        claude_arn.to_owned(),
        "anthropic.claude-sonnet-4-6".to_owned(),
    );
    resolved.insert(
        llama_arn.to_owned(),
        "meta.llama3-3-70b-instruct-v1:0".to_owned(),
    );
    assert!(supports_prompt_caching_with(&resolved, claude_arn));
    assert!(!supports_prompt_caching_with(&resolved, llama_arn));
    // A provisioned-model ARN is never in the map and matches no family,
    // so caching stays off — correct, since AWS forbids it on PT.
    let pt_arn = "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/xyz";
    assert!(!supports_prompt_caching_with(&resolved, pt_arn));
}

#[test]
fn foundation_id_from_model_arn_strips_arn_and_region_prefix() {
    // Application profile wrapping a foundation model directly.
    assert_eq!(
        foundation_id_from_model_arn(
            "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-6"
        ),
        "anthropic.claude-sonnet-4-6"
    );
    // Application profile wrapping a cross-region system profile — the
    // region prefix on the tail is stripped to reach the foundation id.
    assert_eq!(
        foundation_id_from_model_arn(
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.amazon.nova-pro-v1:0"
        ),
        "amazon.nova-pro-v1:0"
    );
}

#[test]
fn build_performance_config_only_set_when_optimized() {
    let mut config = LanguageModelConfig::default();
    assert!(build_performance_config(&config).is_none());
    config.latency = LatencyMode::Optimized;
    let perf = build_performance_config(&config).expect("optimized -> Some");
    assert_eq!(perf.latency(), &PerformanceConfigLatency::Optimized);
}

#[test]
fn latency_optimized_capability_flag_tracks_allow_list() {
    // Nova Pro + Llama 3.1 70B/405B are on the verified allow-list; Claude
    // and other families are not.
    let optimized = |id: &str| {
        MODEL_CAPABILITIES[id]
            .features
            .contains(ChatFeature::LatencyOptimized)
    };
    assert!(optimized("amazon.nova-pro-v1:0"));
    assert!(optimized("meta.llama3-1-70b-instruct-v1:0"));
    assert!(optimized("meta.llama3-1-405b-instruct-v1:0"));
    assert!(!optimized("anthropic.claude-haiku-4-5-20251001-v1:0"));
    assert!(!optimized("amazon.nova-lite-v1:0"));
    assert!(!optimized("meta.llama3-1-8b-instruct-v1:0"));
}

#[test]
fn build_bedrock_messages_inserts_system_cachepoint_when_enabled() {
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
    let (system, _out) = build_bedrock_messages(&messages, true, CacheTtl::FiveMin).unwrap();
    assert_eq!(system.len(), 2, "text block + cache point");
    assert!(matches!(system[0], SystemContentBlock::Text(_)));
    assert!(matches!(system[1], SystemContentBlock::CachePoint(_)));
}

#[test]
fn build_bedrock_messages_drops_cachepoint_when_disabled() {
    let messages = vec![Message::with_parts(
        Role::System,
        vec![
            ContentPart::text("big static prompt"),
            ContentPart::cache_breakpoint(),
        ],
    )];
    let (system, _out) = build_bedrock_messages(&messages, false, CacheTtl::FiveMin).unwrap();
    assert_eq!(
        system.len(),
        1,
        "marker dropped, only the text block remains"
    );
    assert!(matches!(system[0], SystemContentBlock::Text(_)));
}

#[test]
fn build_bedrock_messages_inserts_user_cachepoint_at_marker() {
    let messages = vec![Message::with_parts(
        Role::User,
        vec![
            ContentPart::text("stable head"),
            ContentPart::cache_breakpoint(),
            ContentPart::text("dynamic tail"),
        ],
    )];
    let (_system, out) = build_bedrock_messages(&messages, true, CacheTtl::FiveMin).unwrap();
    assert_eq!(out.len(), 1);
    let content = out[0].content();
    assert_eq!(content.len(), 3, "text, cache point, text");
    assert!(matches!(content[0], ContentBlock::Text(_)));
    assert!(matches!(content[1], ContentBlock::CachePoint(_)));
    assert!(matches!(content[2], ContentBlock::Text(_)));
}

#[test]
fn build_bedrock_messages_sets_requested_ttl_on_cachepoint() {
    let messages = vec![Message::with_parts(
        Role::System,
        vec![
            ContentPart::text("big static prompt"),
            ContentPart::cache_breakpoint(),
        ],
    )];
    let (system, _out) = build_bedrock_messages(&messages, true, CacheTtl::OneHour).unwrap();
    let SystemContentBlock::CachePoint(cp) = &system[1] else {
        panic!("expected a cache point block");
    };
    assert_eq!(cp.ttl(), Some(&BedrockCacheTtl::OneHour));
}

#[test]
fn bedrock_usage_maps_cache_tokens() {
    let usage = aws_sdk_bedrockruntime::types::TokenUsage::builder()
        .input_tokens(10)
        .output_tokens(5)
        .total_tokens(315)
        .cache_read_input_tokens(100)
        .cache_write_input_tokens(200)
        .build()
        .unwrap();
    let mapped = bedrock_usage(&usage);
    assert_eq!(mapped.input_tokens, 10);
    assert_eq!(mapped.output_tokens, 5);
    assert_eq!(mapped.cache_read_input_tokens, 100);
    assert_eq!(mapped.cache_creation_input_tokens, 200);
}

#[test]
fn bedrock_usage_defaults_cache_tokens_to_zero() {
    let usage = aws_sdk_bedrockruntime::types::TokenUsage::builder()
        .input_tokens(10)
        .output_tokens(5)
        .total_tokens(15)
        .build()
        .unwrap();
    let mapped = bedrock_usage(&usage);
    assert_eq!(mapped.cache_read_input_tokens, 0);
    assert_eq!(mapped.cache_creation_input_tokens, 0);
}

#[test]
fn build_inference_config_returns_none_when_unset() {
    let config = LanguageModelConfig::default();
    assert!(build_inference_config(&config).is_none());
}

#[test]
fn build_output_config_returns_none_for_text() {
    let cfg = build_output_config(&ResponseFormat::Text).unwrap();
    assert!(cfg.is_none());
}

#[test]
fn build_output_config_jsonobject_uses_permissive_schema() {
    let cfg = build_output_config(&ResponseFormat::JsonObject)
        .unwrap()
        .unwrap();
    let format = cfg.text_format().unwrap();
    assert_eq!(format.r#type(), &OutputFormatType::JsonSchema);
    let OutputFormatStructure::JsonSchema(def) = format.structure().unwrap() else {
        panic!("expected JsonSchema structure variant");
    };
    let parsed: serde_json::Value = serde_json::from_str(def.schema()).unwrap();
    assert_eq!(parsed, serde_json::json!({"type": "object"}));
}

#[test]
fn build_output_config_jsonschema_passes_through_schema_and_name() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    });
    let cfg = build_output_config(&ResponseFormat::JsonSchema {
        name: "answer_schema".into(),
        schema: schema.clone(),
        strict: true,
    })
    .unwrap()
    .unwrap();
    let format = cfg.text_format().unwrap();
    let OutputFormatStructure::JsonSchema(def) = format.structure().unwrap() else {
        panic!("expected JsonSchema structure variant");
    };
    assert_eq!(def.name(), Some("answer_schema"));
    let parsed: serde_json::Value = serde_json::from_str(def.schema()).unwrap();
    assert_eq!(parsed, schema);
}

#[test]
fn build_inference_config_populates_set_fields() {
    let config = LanguageModelConfig {
        max_tokens: Some(1024),
        temperature: Some(0.7),
        top_p: Some(0.9),
        stop: vec!["END".into()],
        ..LanguageModelConfig::default()
    };
    let inferred = build_inference_config(&config).unwrap();
    assert_eq!(inferred.max_tokens(), Some(1024));
    assert!((inferred.temperature().unwrap() - 0.7).abs() < 0.01);
    assert!((inferred.top_p().unwrap() - 0.9).abs() < 0.01);
    assert_eq!(inferred.stop_sequences(), &["END".to_owned()]);
}

#[test]
fn parse_converse_output_concatenates_text_blocks() {
    let msg = BedrockMessage::builder()
        .role(ConversationRole::Assistant)
        .content(ContentBlock::Text("hello ".into()))
        .content(ContentBlock::Text("world".into()))
        .build()
        .unwrap();
    let usage = TokenUsageBuilder::default()
        .input_tokens(10)
        .output_tokens(5)
        .total_tokens(15)
        .build()
        .unwrap();
    let response = parse_converse_output(
        "anthropic.claude-haiku-4-5",
        Some(BedrockConverseOutput::Message(msg)),
        Some(&usage),
        Some(&aws_sdk_bedrockruntime::types::StopReason::EndTurn),
    )
    .unwrap();
    assert_eq!(response.content, "hello world");
    assert_eq!(response.usage.unwrap().input_tokens, 10);
    assert_eq!(response.usage.unwrap().output_tokens, 5);
    assert_eq!(response.stop_reason, Some(crate::StopReason::EndTurn));
}

#[test]
fn parse_converse_output_empty_text_block_yields_empty_content() {
    let msg = BedrockMessage::builder()
        .role(ConversationRole::Assistant)
        .content(ContentBlock::Text(String::new()))
        .build()
        .unwrap();
    let result = parse_converse_output(
        "anthropic.claude-haiku-4-5",
        Some(BedrockConverseOutput::Message(msg)),
        None,
        None,
    )
    .unwrap();
    assert_eq!(result.content, "");
}

#[test]
fn additional_request_fields_gpt_oss_maps_reasoning_effort() {
    let low = ReasoningConfig::Adaptive {
        effort: ReasoningEffort::Low,
    };
    let doc = bedrock_additional_request_fields("openai.gpt-oss-20b-1:0", Some(&low))
        .unwrap()
        .expect("gpt-oss adaptive -> Some");
    assert_eq!(
        doc.as_object()
            .unwrap()
            .get("reasoning_effort")
            .unwrap()
            .as_string(),
        Some("low")
    );
    // A region-prefixed gpt-oss id resolves to the same family.
    let high = ReasoningConfig::Adaptive {
        effort: ReasoningEffort::High,
    };
    let doc = bedrock_additional_request_fields("us.openai.gpt-oss-120b-1:0", Some(&high))
        .unwrap()
        .unwrap();
    assert_eq!(
        doc.as_object()
            .unwrap()
            .get("reasoning_effort")
            .unwrap()
            .as_string(),
        Some("high")
    );
}

#[test]
fn additional_request_fields_gpt_oss_off_fails_loud() {
    let err =
        bedrock_additional_request_fields("openai.gpt-oss-20b-1:0", Some(&ReasoningConfig::Off))
            .unwrap_err();
    assert!(matches!(err, LanguageModelError::Provider { .. }));
    assert!(err.to_string().contains("cannot disable reasoning"));
}

#[test]
fn additional_request_fields_anthropic_emits_thinking_and_omits_on_off() {
    let cfg = ReasoningConfig::Adaptive {
        effort: ReasoningEffort::Medium,
    };
    let doc = bedrock_additional_request_fields("anthropic.claude-sonnet-4-6", Some(&cfg))
        .unwrap()
        .expect("claude adaptive -> Some");
    let obj = doc.as_object().unwrap();
    assert!(obj.contains_key("thinking"));
    assert!(obj.contains_key("output_config"));
    assert!(
        bedrock_additional_request_fields(
            "anthropic.claude-sonnet-4-6",
            Some(&ReasoningConfig::Off)
        )
        .unwrap()
        .is_none()
    );
}

#[test]
fn additional_request_fields_other_family_sends_nothing() {
    // Non-reasoning chat() family: Off is a no-op (the caller rejects
    // adaptive/manual against its `reasoning: None` capability).
    assert!(
        bedrock_additional_request_fields(
            "meta.llama3-3-70b-instruct-v1:0",
            Some(&ReasoningConfig::Off)
        )
        .unwrap()
        .is_none()
    );
    assert!(
        bedrock_additional_request_fields("meta.llama3-3-70b-instruct-v1:0", None)
            .unwrap()
            .is_none()
    );
}

#[test]
fn gpt_oss_capability_accepts_low_rejects_none_and_manual() {
    let cap = MODEL_CAPABILITIES["openai.gpt-oss-20b-1:0"]
        .reasoning
        .clone()
        .expect("gpt-oss carries a reasoning capability");
    assert!(
        cap.validate(
            "gpt-oss",
            &ReasoningConfig::Adaptive {
                effort: ReasoningEffort::Low
            }
        )
        .is_ok()
    );
    assert!(
        cap.validate(
            "gpt-oss",
            &ReasoningConfig::Adaptive {
                effort: ReasoningEffort::None
            }
        )
        .is_err()
    );
    assert!(
        cap.validate(
            "gpt-oss",
            &ReasoningConfig::Manual {
                budget_tokens: 1024
            }
        )
        .is_err()
    );
}

#[test]
fn parse_converse_output_captures_reasoning_content() {
    let reasoning = aws_sdk_bedrockruntime::types::ReasoningTextBlock::builder()
        .text("thinking hard")
        .build()
        .unwrap();
    let msg = BedrockMessage::builder()
        .role(ConversationRole::Assistant)
        .content(ContentBlock::ReasoningContent(
            ReasoningContentBlock::ReasoningText(reasoning),
        ))
        .content(ContentBlock::Text("final answer".into()))
        .build()
        .unwrap();
    let r = parse_converse_output(
        "openai.gpt-oss-20b-1:0",
        Some(BedrockConverseOutput::Message(msg)),
        None,
        Some(&aws_sdk_bedrockruntime::types::StopReason::EndTurn),
    )
    .unwrap();
    assert_eq!(r.content, "final answer");
    assert_eq!(r.thinking.as_deref(), Some("thinking hard"));
}

#[test]
fn parse_converse_output_reasoning_exhausted_budget_errors_actionably() {
    let reasoning = aws_sdk_bedrockruntime::types::ReasoningTextBlock::builder()
        .text("... lots of reasoning, no answer ...")
        .build()
        .unwrap();
    let msg = BedrockMessage::builder()
        .role(ConversationRole::Assistant)
        .content(ContentBlock::ReasoningContent(
            ReasoningContentBlock::ReasoningText(reasoning),
        ))
        .build()
        .unwrap();
    let err = parse_converse_output(
        "openai.gpt-oss-20b-1:0",
        Some(BedrockConverseOutput::Message(msg)),
        None,
        Some(&aws_sdk_bedrockruntime::types::StopReason::MaxTokens),
    )
    .unwrap_err();
    assert!(matches!(err, LanguageModelError::Provider { .. }));
    assert!(
        err.to_string()
            .contains("exhausted max_tokens during reasoning")
    );
}

#[test]
fn split_inline_reasoning_handles_tagged_unclosed_and_plain() {
    let (answer, reasoning) = split_inline_reasoning("<reasoning>think</reasoning>the answer");
    assert_eq!(answer, "the answer");
    assert_eq!(reasoning.as_deref(), Some("think"));
    // No tags -> verbatim text, no reasoning, no trimming.
    assert_eq!(
        split_inline_reasoning(" plain "),
        (" plain ".to_owned(), None)
    );
    // Unterminated tag (budget cut off mid-thought) -> all reasoning, empty answer.
    let (answer, reasoning) = split_inline_reasoning("<reasoning>cut off mid");
    assert_eq!(answer, "");
    assert_eq!(reasoning.as_deref(), Some("cut off mid"));
}

#[test]
fn convert_stream_event_text_delta_yields_non_final() {
    let event = ConverseStreamOutput::ContentBlockDelta(
        ContentBlockDeltaEvent::builder()
            .delta(ContentBlockDelta::Text("chunk".into()))
            .content_block_index(0)
            .build()
            .unwrap(),
    );
    let delta = convert_stream_event_to_delta(&event).unwrap();
    assert_eq!(delta.content, "chunk");
    assert!(!delta.is_final);
    assert!(delta.usage.is_none());
}

#[test]
fn convert_stream_event_metadata_yields_final_with_usage() {
    let usage = TokenUsageBuilder::default()
        .input_tokens(20)
        .output_tokens(8)
        .total_tokens(28)
        .build()
        .unwrap();
    let event =
        ConverseStreamOutput::Metadata(ConverseStreamMetadataEvent::builder().usage(usage).build());
    let delta = convert_stream_event_to_delta(&event).unwrap();
    assert!(delta.is_final);
    assert_eq!(delta.content, "");
    let u = delta.usage.unwrap();
    assert_eq!(u.input_tokens, 20);
    assert_eq!(u.output_tokens, 8);
}

#[test]
fn convert_stream_event_message_start_yields_nothing() {
    let event = ConverseStreamOutput::MessageStart(
        MessageStartEvent::builder()
            .role(ConversationRole::Assistant)
            .build()
            .unwrap(),
    );
    assert!(convert_stream_event_to_delta(&event).is_none());
}

#[test]
fn convert_stream_event_message_stop_yields_stop_reason() {
    let event = ConverseStreamOutput::MessageStop(
        MessageStopEvent::builder()
            .stop_reason(aws_sdk_bedrockruntime::types::StopReason::MaxTokens)
            .build()
            .unwrap(),
    );
    let delta = convert_stream_event_to_delta(&event).unwrap();
    assert_eq!(delta.stop_reason, Some(crate::StopReason::MaxTokens));
    assert!(delta.content.is_empty());
    assert!(delta.thinking.is_none());
    assert!(delta.usage.is_none());
    assert!(
        !delta.is_final,
        "MessageStop precedes the terminating Metadata event -- is_final stays false",
    );
}

#[test]
fn convert_stream_event_reasoning_text_yields_thinking() {
    let event = ConverseStreamOutput::ContentBlockDelta(
        ContentBlockDeltaEvent::builder()
            .delta(ContentBlockDelta::ReasoningContent(
                ReasoningContentBlockDelta::Text("step one of CoT".into()),
            ))
            .content_block_index(0)
            .build()
            .unwrap(),
    );
    let delta = convert_stream_event_to_delta(&event).unwrap();
    assert!(delta.content.is_empty());
    assert_eq!(delta.thinking.as_deref(), Some("step one of CoT"));
    assert!(!delta.is_final);
}

#[test]
fn convert_stream_event_reasoning_signature_is_dropped() {
    let event = ConverseStreamOutput::ContentBlockDelta(
        ContentBlockDeltaEvent::builder()
            .delta(ContentBlockDelta::ReasoningContent(
                ReasoningContentBlockDelta::Signature("sig".into()),
            ))
            .content_block_index(0)
            .build()
            .unwrap(),
    );
    assert!(convert_stream_event_to_delta(&event).is_none());
}

#[test]
fn strip_region_prefix_handles_known_regions() {
    assert_eq!(
        strip_region_prefix("us.anthropic.claude-haiku-4-5-20251001-v1:0"),
        "anthropic.claude-haiku-4-5-20251001-v1:0"
    );
    assert_eq!(
        strip_region_prefix("eu.anthropic.claude-sonnet-4-20250514-v1:0"),
        "anthropic.claude-sonnet-4-20250514-v1:0"
    );
    assert_eq!(
        strip_region_prefix("apac.anthropic.claude-haiku-4-5-20251001-v1:0"),
        "anthropic.claude-haiku-4-5-20251001-v1:0"
    );
    assert_eq!(
        strip_region_prefix("jp.anthropic.claude-haiku-4-5-20251001-v1:0"),
        "anthropic.claude-haiku-4-5-20251001-v1:0"
    );
    assert_eq!(
        strip_region_prefix("au.anthropic.claude-haiku-4-5-20251001-v1:0"),
        "anthropic.claude-haiku-4-5-20251001-v1:0"
    );
    assert_eq!(
        strip_region_prefix("us-gov.anthropic.claude-haiku-4-5-20251001-v1:0"),
        "anthropic.claude-haiku-4-5-20251001-v1:0"
    );
    assert_eq!(
        strip_region_prefix("global.anthropic.claude-haiku-4-5-20251001-v1:0"),
        "anthropic.claude-haiku-4-5-20251001-v1:0"
    );
    // No prefix -> returned as-is.
    assert_eq!(
        strip_region_prefix("anthropic.claude-haiku-4-5-20251001-v1:0"),
        "anthropic.claude-haiku-4-5-20251001-v1:0"
    );
}

#[test]
fn model_capabilities_seed_anthropic_marks_json_schema() {
    let caps = MODEL_CAPABILITIES
        .get("anthropic.claude-haiku-4-5-20251001-v1:0")
        .unwrap();
    assert_eq!(caps.context_window, 200_000);
    assert!(caps.features.contains(ChatFeature::Streaming));
    assert!(caps.features.contains(ChatFeature::JsonSchema));
}

#[test]
fn model_capabilities_seed_non_anthropic_marks_no_json_schema() {
    let nova = MODEL_CAPABILITIES.get("amazon.nova-pro-v1:0").unwrap();
    assert_eq!(nova.context_window, 300_000);
    assert!(!nova.features.contains(ChatFeature::JsonSchema));
    let llama = MODEL_CAPABILITIES
        .get("meta.llama3-3-70b-instruct-v1:0")
        .unwrap();
    assert_eq!(llama.context_window, 128_000);
    assert!(!llama.features.contains(ChatFeature::JsonSchema));
    let mistral_old = MODEL_CAPABILITIES
        .get("mistral.mixtral-8x7b-instruct-v0:1")
        .unwrap();
    assert_eq!(mistral_old.context_window, 32_000);
    let mistral_new = MODEL_CAPABILITIES
        .get("mistral.mistral-large-3-675b-instruct")
        .unwrap();
    assert_eq!(mistral_new.context_window, 128_000);
}

// ----- Media translation + capability table -----

#[test]
fn translate_image_inline_bytes_emits_image_block_bytes() {
    let part = ContentPart::image(MediaSource::InlineBytes {
        mime: crate::MediaType::parse("image/png").unwrap(),
        data: b"PNGDATA".to_vec(),
    });
    let block = translate_part_for_bedrock(&part).unwrap();
    match block {
        ContentBlock::Image(img) => {
            assert_eq!(img.format(), &ImageFormat::Png);
            match img.source().unwrap() {
                ImageSource::Bytes(blob) => assert_eq!(blob.as_ref(), b"PNGDATA"),
                other => panic!("expected ImageSource::Bytes, got {other:?}"),
            }
        }
        other => panic!("expected ContentBlock::Image, got {other:?}"),
    }
}

#[test]
fn translate_image_s3_emits_image_block_s3_location() {
    let part = ContentPart::image(MediaSource::S3 {
        uri: crate::S3Uri::parse("s3://my-bucket/key.png").unwrap(),
        bucket_owner: Some(crate::AwsAccountId::parse("123456789012").unwrap()),
    });
    let block = translate_part_for_bedrock(&part).unwrap();
    match block {
        ContentBlock::Image(img) => match img.source().unwrap() {
            ImageSource::S3Location(loc) => {
                assert_eq!(loc.uri(), "s3://my-bucket/key.png");
                assert_eq!(loc.bucket_owner(), Some("123456789012"));
            }
            other => panic!("expected S3Location, got {other:?}"),
        },
        other => panic!("expected ContentBlock::Image, got {other:?}"),
    }
}

#[test]
fn translate_image_url_drops_with_warning() {
    let part = ContentPart::image(MediaSource::Url {
        url: crate::HttpsUrl::parse("https://example.com/x.png").unwrap(),
    });
    assert!(translate_part_for_bedrock(&part).is_none());
}

#[test]
fn translate_document_inline_bytes_emits_document_block() {
    let part = ContentPart::document(
        MediaSource::InlineBytes {
            mime: crate::MediaType::parse("application/pdf").unwrap(),
            data: b"%PDF".to_vec(),
        },
        Some("report.pdf".into()),
    );
    let block = translate_part_for_bedrock(&part).unwrap();
    match block {
        ContentBlock::Document(doc) => {
            assert_eq!(doc.format(), &DocumentFormat::Pdf);
            assert_eq!(doc.name(), "report.pdf");
        }
        other => panic!("expected ContentBlock::Document, got {other:?}"),
    }
}

#[test]
fn translate_video_inline_bytes_emits_video_block() {
    let part = ContentPart::video(MediaSource::InlineBytes {
        mime: crate::MediaType::parse("video/mp4").unwrap(),
        data: b"VIDEO".to_vec(),
    });
    let block = translate_part_for_bedrock(&part).unwrap();
    match block {
        ContentBlock::Video(video) => {
            assert_eq!(video.format(), &VideoFormat::Mp4);
        }
        other => panic!("expected ContentBlock::Video, got {other:?}"),
    }
}

#[test]
fn media_support_for_claude_has_image_and_doc_no_video() {
    let support = bedrock_media_support("anthropic.claude-haiku-4-5-20251001-v1:0");
    assert!(support.contains_key(&crate::MediaKind::Image));
    assert!(support.contains_key(&crate::MediaKind::Document));
    assert!(!support.contains_key(&crate::MediaKind::Video));
    let image = support.get(&crate::MediaKind::Image).unwrap();
    assert!(image.sources.contains(crate::SourceKind::InlineBytes));
    assert!(image.sources.contains(crate::SourceKind::S3));
    assert!(!image.sources.contains(crate::SourceKind::Url));
}

#[test]
fn media_support_for_nova_pro_adds_video() {
    let support = bedrock_media_support("amazon.nova-pro-v1:0");
    assert!(support.contains_key(&crate::MediaKind::Image));
    assert!(support.contains_key(&crate::MediaKind::Document));
    assert!(support.contains_key(&crate::MediaKind::Video));
}

#[test]
fn media_support_for_text_only_models_is_empty() {
    assert!(bedrock_media_support("amazon.nova-micro-v1:0").is_empty());
    assert!(bedrock_media_support("meta.llama3-1-70b-instruct-v1:0").is_empty());
    assert!(bedrock_media_support("deepseek.r1-v1:0").is_empty());
}

#[test]
fn media_support_for_voxtral_has_audio() {
    let support = bedrock_media_support("mistral.voxtral-small-24b-2507");
    assert!(support.contains_key(&crate::MediaKind::Audio));
}
