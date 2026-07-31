// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#![expect(
    clippy::print_stdout,
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "examples exist to demo the API and print results to the terminal; the workspace \
              bans on print_*/unwrap/expect target production code, not example binaries"
)]

//! Test `AnthropicLanguageModel` against the real Anthropic API.
//!
//! Three sections:
//!   1. Text only (non-streaming + streaming)
//!   2. Vision via HTTPS URL — exercises the `MediaSource::Url` path
//!   3. Vision via inline base64 bytes — exercises `MediaSource::InlineBytes`
//!
//! Usage:
//!   dotenvx run -f .env.local -- cargo run -p modelplease --example anthropic_test

use std::sync::Arc;

mod common;

use common::build_http_client;
use futures::StreamExt;
use modelplease::{
    AnthropicConfig, AnthropicDeps, AnthropicLanguageModel, ApiKey, ContentPart, GenerateRequest,
    HttpsUrl, LanguageModelConfig, LanguageModelProvider, MediaSource, MediaType, Message, ModelId,
    ReasoningConfig, ReasoningEffort, RetryConfig, Role,
};

/// Test image for the vision sections. Anthropic's image-fetch service
/// rejects `upload.wikimedia.org` URLs (HTTP 400 "Unable to download
/// the file") even though OpenAI's fetcher accepts the same URL, so
/// this example points at Google's `gstatic` demo CDN which is
/// permissive across providers.
const SAMPLE_IMAGE_URL: &str = "https://www.gstatic.com/webp/gallery/1.jpg";
const SAMPLE_IMAGE_MIME: &str = "image/jpeg";

/// Wikimedia's CDN rejects requests with reqwest's default User-Agent
/// (HTTP 403). Their policy requires a real UA string identifying the
/// requester — populate it explicitly for both the example fetch and
/// any provider that fetches the URL on our behalf.
const SAMPLE_IMAGE_USER_AGENT: &str =
    "modelplease-example/0.1 (https://github.com/moderately-ai/modelplease)";

/// Fetch [`SAMPLE_IMAGE_URL`] via the shared HTTP client so the vision
/// path that hands bytes to the model has something real to look at.
async fn fetch_sample_image(client: &reqwest::Client) -> (MediaType, Vec<u8>) {
    let bytes = client
        .get(SAMPLE_IMAGE_URL)
        .header(reqwest::header::USER_AGENT, SAMPLE_IMAGE_USER_AGENT)
        .send()
        .await
        .expect("fetch sample image")
        .error_for_status()
        .expect("sample image returned non-2xx")
        .bytes()
        .await
        .expect("read sample image bytes")
        .to_vec();
    (MediaType::parse(SAMPLE_IMAGE_MIME).unwrap(), bytes)
}

#[tokio::main]
async fn main() {
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
    let client = Arc::new(build_http_client().expect("build reqwest client"));
    let lm = AnthropicLanguageModel::new(
        AnthropicDeps {
            client: Arc::clone(&client),
        },
        AnthropicConfig {
            api_key: ApiKey::parse(api_key).unwrap(),
            base_url: AnthropicConfig::DEFAULT_BASE_URL.to_owned(),
            retry_config: RetryConfig::default(),
        },
    );
    let model = ModelId::new("claude-sonnet-4-6");

    let config = LanguageModelConfig {
        temperature: Some(0.0),
        max_tokens: Some(256),
        ..Default::default()
    };

    let messages = vec![
        Message::system("You are a helpful assistant. Be concise."),
        Message::user("What is the capital of France? Answer in one word."),
    ];

    // --- Non-streaming ---
    println!("=== Non-streaming (text) ===");
    let request = GenerateRequest {
        model: &model,
        messages: &messages,
        config: &config,
    };
    let r = lm
        .generate(request)
        .await
        .expect("non-streaming text generate");
    println!("Model: {:?}", r.model);
    println!("Content: {}", r.content);
    if let Some(u) = &r.usage {
        println!(
            "Usage: {} input, {} output tokens",
            u.input_tokens, u.output_tokens
        );
    }
    assert!(
        !r.content.is_empty(),
        "non-streaming text produced empty content"
    );

    // --- Streaming ---
    println!("\n=== Streaming (text) ===");
    let request = GenerateRequest {
        model: &model,
        messages: &messages,
        config: &config,
    };
    let mut stream = lm
        .generate_stream(request)
        .await
        .expect("streaming text generate");
    print!("Content: ");
    let mut got_any = false;
    while let Some(result) = stream.next().await {
        let delta = result.expect("streaming delta error");
        print!("{}", delta.content);
        if !delta.content.is_empty() {
            got_any = true;
        }
        if delta.is_final {
            println!(" [done]");
        }
    }
    assert!(got_any, "streaming produced no content");

    // --- Reasoning: adaptive (Sonnet 4.6 / Opus 4.7 — qualitative effort) ---
    // Sonnet 4.6 supports both adaptive and manual; this exercises the
    // adaptive path. Override with ANTHROPIC_REASONING_MODEL=<id>
    // (e.g. claude-opus-4-7 to exercise the xhigh / max effort enum).
    println!("\n=== Reasoning (adaptive, effort=high) ===");
    let reasoning_model = ModelId::new(
        std::env::var("ANTHROPIC_REASONING_MODEL")
            .unwrap_or_else(|_| "claude-sonnet-4-6".to_owned()),
    );
    let reasoning_config = LanguageModelConfig {
        // Anthropic rejects `temperature` whenever thinking is on; leave
        // it unset so the provider's wire layer doesn't try to send it.
        max_tokens: Some(8192),
        reasoning: Some(ReasoningConfig::Adaptive {
            effort: ReasoningEffort::High,
        }),
        ..LanguageModelConfig::default()
    };
    let reasoning_messages = vec![Message::user(
        "If a train leaves Boston at 60 mph and another leaves NYC at 75 mph on the same \
         track 200 miles apart, when do they meet? Show your reasoning step by step.",
    )];
    let request = GenerateRequest {
        model: &reasoning_model,
        messages: &reasoning_messages,
        config: &reasoning_config,
    };
    let r = lm
        .generate(request)
        .await
        .expect("reasoning adaptive generate");
    println!("Model: {:?}", r.model);
    if let Some(thinking) = &r.thinking {
        println!(
            "Thinking ({} chars): {}",
            thinking.len(),
            &thinking.chars().take(280).collect::<String>()
        );
    } else {
        println!("Thinking: <none surfaced — model omitted or summary returned empty>");
    }
    println!("Content: {}", r.content);
    if let Some(u) = &r.usage {
        println!(
            "Usage: {} input, {} output tokens",
            u.input_tokens, u.output_tokens
        );
    }
    assert!(
        !r.content.is_empty(),
        "reasoning adaptive call produced no visible content"
    );

    // --- Reasoning: manual budget (Sonnet 4.6 — fixed budget_tokens) ---
    println!("\n=== Reasoning (manual, budget_tokens=4096) ===");
    let manual_config = LanguageModelConfig {
        max_tokens: Some(8192),
        reasoning: Some(ReasoningConfig::Manual {
            budget_tokens: 4096,
        }),
        ..LanguageModelConfig::default()
    };
    let request = GenerateRequest {
        model: &reasoning_model,
        messages: &reasoning_messages,
        config: &manual_config,
    };
    let r = lm
        .generate(request)
        .await
        .expect("reasoning manual generate");
    if let Some(thinking) = &r.thinking {
        println!("Thinking length: {} chars", thinking.len());
    }
    println!("Content: {}", r.content);
    if let Some(u) = &r.usage {
        println!(
            "Usage: {} input, {} output tokens",
            u.input_tokens, u.output_tokens
        );
    }
    assert!(
        !r.content.is_empty(),
        "reasoning manual call produced no visible content"
    );

    // --- Vision: image via HTTPS URL ---
    println!("\n=== Vision (image_url, MediaSource::Url) ===");
    let url_messages = vec![
        Message::system("You are a helpful assistant. Be concise."),
        Message::with_parts(
            Role::User,
            vec![
                ContentPart::text("Describe this image in one sentence."),
                ContentPart::image(MediaSource::Url {
                    url: HttpsUrl::parse(SAMPLE_IMAGE_URL).unwrap(),
                }),
            ],
        ),
    ];
    let request = GenerateRequest {
        model: &model,
        messages: &url_messages,
        config: &config,
    };
    let r = lm.generate(request).await.expect("vision url generate");
    println!("Content: {}", r.content);
    if let Some(u) = &r.usage {
        println!(
            "Usage: {} input, {} output tokens",
            u.input_tokens, u.output_tokens
        );
    }
    assert!(
        !r.content.is_empty(),
        "vision url content must be non-empty"
    );

    // --- Vision: image via inline base64 bytes ---
    println!("\n=== Vision (image bytes, MediaSource::InlineBytes) ===");
    let (mime, image_bytes) = fetch_sample_image(&client).await;
    println!(
        "Fetched {} bytes from {} ({})",
        image_bytes.len(),
        SAMPLE_IMAGE_URL,
        mime.as_str()
    );
    let bytes_messages = vec![
        Message::system("You are a helpful assistant. Be concise."),
        Message::with_parts(
            Role::User,
            vec![
                ContentPart::text("Describe this image in one sentence."),
                ContentPart::image(MediaSource::InlineBytes {
                    mime,
                    data: image_bytes,
                }),
            ],
        ),
    ];
    let request = GenerateRequest {
        model: &model,
        messages: &bytes_messages,
        config: &config,
    };
    let r = lm.generate(request).await.expect("vision bytes generate");
    println!("Content: {}", r.content);
    if let Some(u) = &r.usage {
        println!(
            "Usage: {} input, {} output tokens",
            u.input_tokens, u.output_tokens
        );
    }
    assert!(
        !r.content.is_empty(),
        "vision bytes content must be non-empty"
    );
}
