// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Live AWS Bedrock Mantle integration tests.
//!
//! These hit the real `bedrock-mantle.{region}.api.aws` endpoint family and need AWS credentials,
//! so they're `#[ignore]`d. Run with a Bedrock-Mantle-capable AWS identity:
//!
//! ```sh
//! cargo nextest run -p modelplease \
//!     bedrock_mantle_live --run-ignored all
//! ```
//!
//! Each test exercises one of the four Mantle surfaces end-to-end (SigV4 signing, request
//! encoding, response parsing). Failures here prove the production path doesn't work against
//! the real upstream — unit tests only prove the wire shape is well-formed locally.
//!
//! Env overrides:
//! - `BEDROCK_MANTLE_DEFAULT_REGION` (default `us-west-2`)
//! - `BEDROCK_MANTLE_OPENAI_GPT5_REGION` (default `us-east-2`)
//! - `BEDROCK_MANTLE_ANTHROPIC_REGION` (default `us-east-1`)

use std::sync::Arc;

use aws_config::BehaviorVersion;
use futures::StreamExt as _;
use modelplease::{
    BedrockMantleAuth, BedrockMantleProvider, BedrockMantleProviderConfig,
    BedrockMantleProviderDeps, GenerateRequest, LanguageModelConfig, LanguageModelProvider,
    Message, ModelId, ReasoningConfig, ReasoningEffort, RetryConfig,
};

/// Build a Mantle provider against the default AWS credential chain. Returns `Result` so
/// the helper itself stays unwrap/expect-free; each test calls `.expect(...)` inside its own
/// `#[tokio::test]` body where clippy's `allow-expect-in-tests` carve-out applies.
async fn make_provider() -> Result<BedrockMantleProvider, &'static str> {
    let default_region = std::env::var("BEDROCK_MANTLE_DEFAULT_REGION")
        .unwrap_or_else(|_| BedrockMantleProviderConfig::DEFAULT_REGION.to_owned());
    let openai_gpt5_region = std::env::var("BEDROCK_MANTLE_OPENAI_GPT5_REGION")
        .unwrap_or_else(|_| BedrockMantleProviderConfig::DEFAULT_OPENAI_GPT5_REGION.to_owned());
    let anthropic_region = std::env::var("BEDROCK_MANTLE_ANTHROPIC_REGION")
        .unwrap_or_else(|_| BedrockMantleProviderConfig::DEFAULT_ANTHROPIC_REGION.to_owned());

    let sdk = aws_config::defaults(BehaviorVersion::latest())
        .region(aws_config::Region::new(default_region.clone()))
        .load()
        .await;
    let credentials_provider = sdk
        .credentials_provider()
        .ok_or("AWS credential chain produced no provider — run via aws-vault")?;

    Ok(BedrockMantleProvider::new(
        BedrockMantleProviderDeps {
            client: Arc::new(reqwest::Client::new()),
            auth: BedrockMantleAuth::Sigv4 {
                credentials_provider,
            },
        },
        BedrockMantleProviderConfig {
            default_region,
            openai_gpt5_region,
            anthropic_region,
            retry_config: RetryConfig::default(),
        },
    ))
}

fn capital_question() -> Vec<Message> {
    vec![
        Message::system("You are a helpful assistant. Be concise."),
        Message::user("What is the capital of France? Answer in one word."),
    ]
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "hits real AWS Bedrock Mantle; run via ... --run-ignored all"]
async fn bedrock_mantle_live_list_models_merges_per_region_catalogs() {
    let provider = make_provider().await.expect("build mantle provider");
    let models = provider.list_models().await.expect("list_models");
    // Sanity: at least the catalog floor we saw on 2026-06-04 (40 in us-west-2). Multi-region
    // merge should push this higher — the assertion uses a generous lower bound rather than the
    // exact 45 we observed, so a new model addition doesn't flake this test.
    assert!(
        models.len() >= 40,
        "expected ≥40 merged models from Mantle catalog, got {}",
        models.len()
    );
    // Spot-check IDs that must exist on a healthy catalog. These cover all three API surfaces:
    let ids: std::collections::HashSet<String> =
        models.iter().map(|m| m.id.as_str().to_owned()).collect();
    assert!(
        ids.contains("openai.gpt-oss-120b"),
        "Chat Completions exemplar missing"
    );
    assert!(
        ids.contains("anthropic.claude-haiku-4-5"),
        "Messages exemplar missing"
    );
    assert!(ids.contains("openai.gpt-5.5"), "Responses exemplar missing");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "hits real AWS Bedrock Mantle; run via ... --run-ignored all"]
async fn bedrock_mantle_live_chat_completions_returns_text() {
    let provider = make_provider().await.expect("build mantle provider");
    let model = ModelId::new("mistral.ministral-3-8b-instruct");
    let config = LanguageModelConfig {
        temperature: Some(0.0),
        max_tokens: Some(32),
        ..Default::default()
    };
    let messages = capital_question();
    let response = provider
        .generate(GenerateRequest {
            model: &model,
            messages: &messages,
            config: &config,
        })
        .await
        .expect("generate");
    assert!(
        !response.content.is_empty(),
        "empty response from Chat Completions"
    );
    assert!(
        response.content.to_lowercase().contains("paris"),
        "unexpected reply: {}",
        response.content
    );
    let usage = response.usage.expect("usage missing");
    assert!(usage.input_tokens > 0);
    assert!(usage.output_tokens > 0);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "hits real AWS Bedrock Mantle; run via ... --run-ignored all"]
async fn bedrock_mantle_live_chat_completions_streams_text() {
    let provider = make_provider().await.expect("build mantle provider");
    let model = ModelId::new("deepseek.v3.2");
    let config = LanguageModelConfig {
        temperature: Some(0.0),
        max_tokens: Some(32),
        ..Default::default()
    };
    let messages = capital_question();
    let mut stream = provider
        .generate_stream(GenerateRequest {
            model: &model,
            messages: &messages,
            config: &config,
        })
        .await
        .expect("generate_stream");

    // Mantle's Chat Completions stream terminates with a usage chunk and then closes the HTTP
    // connection — it does NOT emit OpenAI's `data: [DONE]` sentinel that fires our
    // `is_final = true` signal. Either stream-end form (explicit `[DONE]` from OpenAI-compatible
    // servers, or silent EOF from Mantle) is a valid termination; we assert on content + usage
    // rather than the optional sentinel.
    let mut accumulated = String::new();
    let mut final_usage = None;
    while let Some(delta) = stream.next().await {
        let delta = delta.expect("stream delta");
        accumulated.push_str(&delta.content);
        if delta.usage.is_some() {
            final_usage = delta.usage;
        }
        if delta.is_final {
            break;
        }
    }
    assert!(
        accumulated.to_lowercase().contains("paris"),
        "unexpected streamed text: {accumulated}"
    );
    let usage = final_usage.expect("final usage chunk missing — Mantle didn't emit include_usage");
    assert!(usage.input_tokens > 0);
    assert!(usage.output_tokens > 0);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "hits real AWS Bedrock Mantle; run via ... --run-ignored all"]
async fn bedrock_mantle_live_anthropic_messages_returns_text() {
    let provider = make_provider().await.expect("build mantle provider");
    let model = ModelId::new("anthropic.claude-haiku-4-5");
    let config = LanguageModelConfig {
        temperature: Some(0.0),
        max_tokens: Some(32),
        ..Default::default()
    };
    let messages = capital_question();
    let response = provider
        .generate(GenerateRequest {
            model: &model,
            messages: &messages,
            config: &config,
        })
        .await
        .expect("generate");
    assert!(
        response.content.to_lowercase().contains("paris"),
        "unexpected Messages reply: {}",
        response.content
    );
}

/// Confirms the Mantle Chat Completions buffered wire path forwards
/// `prompt_tokens_details.cached_tokens` end-to-end onto our internal
/// [`Usage`]. Two identical ~11.7k-token prompts <1s apart;
/// the assertion is loose because Mantle's implicit cache TTL is single-
/// digit seconds — both calls can be cold, or just one can hit, or both
/// can hit, depending on upstream timing. What's NOT acceptable is
/// `cache_read_input_tokens > 0` accompanied by a still-large
/// `input_tokens` on the same call — that would mean we forgot to
/// subtract `cached_tokens` in `openai_wire::openai_usage`. The
/// `input + cache_read` invariant is what this test pins.
///
/// Origin: 2026-06-05 — the streaming probe surfaced `prompt=26`
/// vs `prompt=1850` on cached `zai.glm-4.7-flash` calls, raising the
/// question of whether the wire path was dropping the breakdown. This
/// probe confirmed it isn't: the buffered surface returns
/// `Usage { input_tokens: 17, cache_read_input_tokens: 11760, … }` on
/// warm calls (the user message tail + the cached system prefix). The
/// streaming-side observation was the streaming usage-drop bug
/// instead — fixed in `streaming_language_model.rs::usageSource`.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "hits real AWS Bedrock Mantle; run via ... --run-ignored all"]
async fn bedrock_mantle_live_chat_completions_cache_attribution_probe() {
    let provider = make_provider().await.expect("build mantle provider");
    let model = ModelId::new(
        std::env::var("BEDROCK_MANTLE_PROBE_MODEL").unwrap_or_else(|_| "zai.glm-4.7-flash".into()),
    );
    let config = LanguageModelConfig {
        temperature: Some(0.0),
        max_tokens: Some(16),
        ..Default::default()
    };

    // ~1700-token system prompt: deterministic content (no nonce) so the
    // two calls hit the same cache key. Mantle's implicit caching keys on
    // prompt content; a nonce would defeat the probe.
    let mut system_body =
        String::from("You are a precise field extractor. Follow the rules below strictly.\n");
    for i in 0..400 {
        use std::fmt::Write as _;
        let _ = writeln!(
            system_body,
            "Rule {i}: extract each requested value exactly and respond using the documented \
             field markers, never inventing data that is not present in the input.",
        );
    }
    let messages = vec![
        Message::system(system_body),
        Message::user("Reply with the single word: ok."),
    ];

    let first = provider
        .generate(GenerateRequest {
            model: &model,
            messages: &messages,
            config: &config,
        })
        .await
        .expect("generate call 1");
    let u1 = first.usage.expect("call 1 usage");

    let second = provider
        .generate(GenerateRequest {
            model: &model,
            messages: &messages,
            config: &config,
        })
        .await
        .expect("generate call 2");
    let u2 = second.usage.expect("call 2 usage");

    // Wire-shape invariant on each call: when cache is reported, the
    // upstream `prompt_tokens` must be split into a small `input_tokens`
    // (the uncached remainder) plus `cache_read_input_tokens` (the
    // cached prefix). Forgetting to subtract in `openai_wire::openai_usage`
    // would surface here as `input_tokens >= cache_read_input_tokens`
    // on a confirmed cache hit.
    for (label, u) in [("call 1", &u1), ("call 2", &u2)] {
        if u.cache_read_input_tokens > 0 {
            assert!(
                u.input_tokens < u.cache_read_input_tokens,
                "{label}: cache_read_input_tokens populated but input_tokens not normalized — \
                 wire mapping is dropping the subtraction. {u:?}",
            );
        }
    }

    // Visibility: print on every run so the test output documents the
    // observed upstream cache state, since it varies per run.
    #[allow(
        clippy::print_stderr,
        reason = "manual live probe reports upstream usage"
    )]
    {
        eprintln!("bedrock_mantle chat-completions cache probe: call_1 = {u1:?}, call_2 = {u2:?}");
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "hits real AWS Bedrock Mantle; run via ... --run-ignored all"]
async fn bedrock_mantle_live_responses_reasoning_returns_text() {
    let provider = make_provider().await.expect("build mantle provider");
    let model = ModelId::new("openai.gpt-5.5");
    // Use the cheapest reasoning effort to keep the test fast and the bill small.
    let config = LanguageModelConfig {
        max_tokens: Some(128),
        reasoning: Some(ReasoningConfig::Adaptive {
            effort: ReasoningEffort::Low,
        }),
        ..Default::default()
    };
    let messages = capital_question();
    let response = provider
        .generate(GenerateRequest {
            model: &model,
            messages: &messages,
            config: &config,
        })
        .await
        .expect("generate");
    assert!(
        response.content.to_lowercase().contains("paris"),
        "unexpected Responses reply: {}",
        response.content
    );
}
