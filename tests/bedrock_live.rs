// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Live AWS Bedrock prompt-caching tests.
//!
//! These hit the real Bedrock Converse API and need credentials, so they
//! are `#[ignore]`d. Run them with a Bedrock-capable AWS identity:
//!
//! ```sh
//! cargo nextest run -p modelplease \
//!     bedrock_live --run-ignored all
//! ```
//!
//! Override the models via `BEDROCK_TEST_MODEL` (a caching family —
//! Claude/Nova) and `BEDROCK_TEST_NONCACHING_MODEL` (e.g. Llama) if the
//! defaults aren't enabled in your account/region.
//!
//! What they prove that the unit tests can't: that the model *actually*
//! honors the `cachePoint` (token thresholds, real support matrix) and
//! reports the cache counters — i.e. the production path works end to end
//! against the real upstream, not just that the request is well-formed.

use modelplease::{
    BedrockProvider, BedrockProviderConfig, BedrockProviderDeps, CacheTtl, ContentPart,
    GenerateRequest, LanguageModelConfig, LanguageModelProvider, Message, ModelId, RetryConfig,
    Role,
};

const DEFAULT_CACHING_MODEL: &str = "us.anthropic.claude-haiku-4-5-20251001-v1:0";
const DEFAULT_NONCACHING_MODEL: &str = "us.meta.llama3-3-70b-instruct-v1:0";

/// A system prompt large enough to clear the model's minimum cacheable
/// token count (Claude Haiku ≈ 2048). The `nonce` makes the cached prefix
/// unique to this run, so call 1 is a guaranteed cache miss (creation)
/// regardless of earlier runs sharing the 5-minute TTL window.
fn big_system_prompt(nonce: &str) -> String {
    use std::fmt::Write as _;
    let mut s = format!("Cache test run {nonce}. You are a precise field extractor.\n");
    for i in 0..600 {
        let _ = writeln!(
            s,
            "Rule {i}: extract each requested value exactly and respond using the documented \
             field markers, never inventing data that is not present in the input."
        );
    }
    s
}

async fn make_provider() -> BedrockProvider {
    let sdk = aws_config::defaults(aws_config::BehaviorVersion::latest())
        .load()
        .await;
    let deps = BedrockProviderDeps {
        runtime_client: aws_sdk_bedrockruntime::Client::new(&sdk),
        control_client: aws_sdk_bedrock::Client::new(&sdk),
    };
    BedrockProvider::new(
        deps,
        BedrockProviderConfig {
            region: None,
            retry_config: RetryConfig::default(),
        },
    )
}

fn cached_messages(nonce: &str) -> Vec<Message> {
    vec![
        Message::with_parts(
            Role::System,
            vec![
                ContentPart::text(big_system_prompt(nonce)),
                ContentPart::cache_breakpoint(),
            ],
        ),
        Message::user("Reply with the single word: ok."),
    ]
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "hits real AWS Bedrock; run via ... --run-ignored all"]
async fn bedrock_live_prompt_cache_creation_then_read() {
    let model = ModelId::new(
        std::env::var("BEDROCK_TEST_MODEL").unwrap_or_else(|_| DEFAULT_CACHING_MODEL.to_owned()),
    );
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .to_string();
    let messages = cached_messages(&nonce);
    let config = LanguageModelConfig {
        max_tokens: Some(16),
        ..Default::default()
    };
    let provider = make_provider().await;

    // Call 1: cold — the prefix is written to the cache.
    let first = provider
        .generate(GenerateRequest {
            model: &model,
            messages: &messages,
            config: &config,
        })
        .await
        .unwrap();
    let u1 = first.usage.unwrap();
    assert!(
        u1.cache_creation_input_tokens > 0,
        "call 1 should write the cache; got {u1:?} (is the prompt above the model min, and does \
         this model support cachePoint?)"
    );
    assert_eq!(u1.cache_read_input_tokens, 0, "call 1 is a cold miss");

    // Call 2: warm — the identical prefix is served from cache.
    let second = provider
        .generate(GenerateRequest {
            model: &model,
            messages: &messages,
            config: &config,
        })
        .await
        .unwrap();
    let u2 = second.usage.unwrap();
    assert!(
        u2.cache_read_input_tokens > 0,
        "call 2 should read the cache; got {u2:?} (ran within the 5-minute TTL?)"
    );

    // Accounting sanity: input_tokens is the uncached remainder, so the
    // cached prefix dominates the prompt on the warm call.
    assert!(
        u2.cache_read_input_tokens > u2.input_tokens,
        "warm call should serve most of the prompt from cache; got {u2:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "hits real AWS Bedrock; run via ... --run-ignored all"]
async fn bedrock_live_non_caching_model_succeeds() {
    // A model whose family doesn't support cachePoint: the gate must drop
    // the marker so the call still succeeds (no ValidationException). This
    // is the primary protection; the in-flight fail-safe retry is the
    // backstop if the gate is ever wrong.
    let model = ModelId::new(
        std::env::var("BEDROCK_TEST_NONCACHING_MODEL")
            .unwrap_or_else(|_| DEFAULT_NONCACHING_MODEL.to_owned()),
    );
    let messages = cached_messages("noncaching");
    let config = LanguageModelConfig {
        max_tokens: Some(16),
        ..Default::default()
    };
    let provider = make_provider().await;

    let result = provider
        .generate(GenerateRequest {
            model: &model,
            messages: &messages,
            config: &config,
        })
        .await;
    assert!(
        result.is_ok(),
        "non-caching model should still succeed: {result:?}"
    );
    let usage = result.unwrap().usage.unwrap();
    assert_eq!(usage.cache_creation_input_tokens, 0);
    assert_eq!(usage.cache_read_input_tokens, 0);
}

/// Large stable user-message head used to push the cumulative cached
/// span over the model minimum even when the system block alone is
/// below the threshold.
fn big_user_head(nonce: &str) -> String {
    use std::fmt::Write as _;
    let mut s =
        format!("Stable context for run {nonce}. The following rules describe the extraction:\n");
    for i in 0..600 {
        let _ = writeln!(
            s,
            "Rule {i}: extract each requested value exactly using the documented field markers."
        );
    }
    s
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "hits real AWS Bedrock; run via ... --run-ignored all"]
async fn bedrock_live_below_min_system_folds_into_user_head_cache() {
    // Settles the #4 open risk: when the system block is below the
    // model's min-cache size, does Bedrock IGNORE the below-min
    // `cachePoint` (leaving the larger user-head breakpoint to cache the
    // cumulative span) or REJECT it (triggering our fail-safe to strip
    // ALL cache points)? The adapter emits breakpoint-1 unconditionally,
    // so this matters for any application whose system prompt is sub-min
    // but whose `dataContext` would push the cumulative span over.
    //
    // A `cache_read_input_tokens > 0` on call 2 proves Bedrock ignored
    // the below-min cachePoint (caching wasn't stripped). A zero would
    // tell us we need to suppress breakpoint-1 when a later cacheable
    // breakpoint exists.
    let model = ModelId::new(
        std::env::var("BEDROCK_TEST_MODEL").unwrap_or_else(|_| DEFAULT_CACHING_MODEL.to_owned()),
    );
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .to_string();
    let head = big_user_head(&nonce);
    let messages = vec![
        // ~30-token system, well under Haiku's ~2048 min.
        Message::with_parts(
            Role::System,
            vec![
                ContentPart::text("You are a precise field extractor."),
                ContentPart::cache_breakpoint(),
            ],
        ),
        Message::with_parts(
            Role::User,
            vec![
                ContentPart::text(head),
                ContentPart::cache_breakpoint(),
                ContentPart::text("Reply with the single word: ok."),
            ],
        ),
    ];
    let config = LanguageModelConfig {
        max_tokens: Some(16),
        ..Default::default()
    };
    let provider = make_provider().await;

    let first = provider
        .generate(GenerateRequest {
            model: &model,
            messages: &messages,
            config: &config,
        })
        .await
        .unwrap();
    let u1 = first.usage.unwrap();
    assert!(
        u1.cache_creation_input_tokens > 0,
        "call 1 must cache the cumulative system+head span; got {u1:?}. \
         If 0, either Bedrock rejected the below-min system breakpoint (fail-safe stripped \
         all cache) or the user head wasn't large enough to clear the model min."
    );

    let second = provider
        .generate(GenerateRequest {
            model: &model,
            messages: &messages,
            config: &config,
        })
        .await
        .unwrap();
    let u2 = second.usage.unwrap();
    assert!(
        u2.cache_read_input_tokens > 0,
        "call 2 must read the cumulative cache; got {u2:?}. \
         A zero here means caching was stripped — Bedrock likely errored on the below-min \
         breakpoint-1, and the fix is to suppress breakpoint-1 when a later cacheable \
         breakpoint exists."
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "hits real AWS Bedrock; run via ... --run-ignored all"]
async fn bedrock_live_one_hour_ttl_request_accepted() {
    // Verifies real Bedrock accepts a `cachePoint` with `CacheTtl::OneHour`.
    // Cross-TTL behaviour (a hit beyond the 5-min ephemeral window) is
    // impractical to assert in CI; this test confirms request acceptance
    // and that a write still happens. The unit test
    // `build_bedrock_messages_sets_requested_ttl_on_cachepoint` covers
    // the serialised TTL value.
    let model = ModelId::new(
        std::env::var("BEDROCK_TEST_MODEL").unwrap_or_else(|_| DEFAULT_CACHING_MODEL.to_owned()),
    );
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .to_string();
    let messages = cached_messages(&nonce);
    let config = LanguageModelConfig {
        max_tokens: Some(16),
        cache_ttl: CacheTtl::OneHour,
        ..Default::default()
    };
    let provider = make_provider().await;

    let resp = provider
        .generate(GenerateRequest {
            model: &model,
            messages: &messages,
            config: &config,
        })
        .await
        .expect("bedrock must accept a 1h-TTL cachePoint request");
    let usage = resp.usage.unwrap();
    assert!(
        usage.cache_creation_input_tokens > 0,
        "1h-TTL call must still write the cache; got {usage:?}"
    );
}
