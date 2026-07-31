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

//! Test `OpenAiLanguageModel` against the real `OpenAI` API.
//!
//! Usage:
//!   dotenvx run -f .env.local -- cargo run -p modelplease --example `openai_test`

use std::sync::Arc;

mod common;

use common::build_http_client;
use futures::StreamExt;
use modelplease::{
    ApiKey, ContentPart, GenerateRequest, HttpsUrl, LanguageModelConfig, LanguageModelProvider,
    MediaSource, MediaType, Message, ModelId, OpenAiConfig, OpenAiDeps, OpenAiLanguageModel,
    ReasoningConfig, ReasoningEffort, ResponseFormat, RetryConfig, Role,
};

/// Standard test image used across the four provider examples: the
/// canonical Lenna PNG hosted on Wikimedia. ~30 KB — big enough for a
/// vision model to say something meaningful, small enough to ship over
/// an inline-bytes wire path without complaint.
const SAMPLE_IMAGE_URL: &str =
    "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png";
const SAMPLE_IMAGE_MIME: &str = "image/png";

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
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let client = Arc::new(build_http_client().expect("build reqwest client"));
    let lm = OpenAiLanguageModel::new(
        OpenAiDeps {
            client: Arc::clone(&client),
        },
        OpenAiConfig {
            api_key: ApiKey::parse(api_key).unwrap(),
            base_url: OpenAiConfig::DEFAULT_BASE_URL.to_owned(),
            retry_config: RetryConfig::default(),
        },
    );
    // gpt-4o-mini is a vision-capable model so the same instance exercises
    // text + image; gpt-4.1-nano is text-only.
    let text_model = ModelId::new("gpt-4.1-nano");
    let vision_model = ModelId::new("gpt-4o-mini");

    let config = LanguageModelConfig {
        temperature: Some(0.0),
        max_tokens: Some(256),
        ..Default::default()
    };

    let messages = vec![
        Message::system("You are a helpful assistant. Be concise."),
        Message::user("What is the capital of France? Answer in one word."),
    ];

    // --- Non-streaming (text) ---
    println!("=== Non-streaming (text) ===");
    let model = text_model.clone();
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
        "non-streaming text content must be non-empty"
    );

    // --- Streaming (text) ---
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

    // --- Structured output: json_schema ---
    println!("\n=== Structured: json_schema ===");
    let schema_config = LanguageModelConfig {
        response_format: ResponseFormat::JsonSchema {
            name: "capital".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "country": {"type": "string"},
                    "capital": {"type": "string"}
                },
                "required": ["country", "capital"],
                "additionalProperties": false
            }),
            strict: true,
        },
        ..config.clone()
    };
    let schema_messages = vec![Message::user("What is the capital of Japan?")];
    let request = GenerateRequest {
        model: &model,
        messages: &schema_messages,
        config: &schema_config,
    };
    let r = lm.generate(request).await.expect("json_schema generate");
    println!("Content: {}", r.content);
    let parsed: serde_json::Value =
        serde_json::from_str(&r.content).expect("json_schema response must be valid JSON");
    println!("Parsed: {parsed}");

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
        model: &vision_model,
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

    // --- Reasoning: adaptive effort against a reasoning-capable model ---
    // gpt-5.4-mini is the cheapest reasoning-capable model in the cap
    // table; override with OPENAI_REASONING_MODEL=<id> to exercise the
    // full GPT-5 effort enum on a different model (e.g. gpt-5.5).
    //
    // `temperature` and `max_tokens` are intentionally omitted: gpt-5
    // reasoning models reject non-default temperature and use the
    // `max_completion_tokens` wire field (provider routes to that
    // automatically per the cap table).
    println!("\n=== Reasoning (adaptive, effort=high) ===");
    let reasoning_model = ModelId::new(
        std::env::var("OPENAI_REASONING_MODEL").unwrap_or_else(|_| "gpt-5.4-mini".to_owned()),
    );
    let reasoning_config = LanguageModelConfig {
        max_tokens: Some(2048),
        reasoning: Some(ReasoningConfig::Adaptive {
            effort: ReasoningEffort::High,
        }),
        ..LanguageModelConfig::default()
    };
    let reasoning_messages = vec![Message::user(
        "If a train leaves Boston at 60 mph and another leaves NYC at 75 mph on the same \
         track 200 miles apart, when do they meet? Show your reasoning.",
    )];
    let request = GenerateRequest {
        model: &reasoning_model,
        messages: &reasoning_messages,
        config: &reasoning_config,
    };
    let r = lm
        .generate(request)
        .await
        .expect("reasoning adaptive high generate");
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
        "adaptive reasoning produced no visible content"
    );

    // --- Reasoning: explicit off on the same model ---
    println!("\n=== Reasoning (off) ===");
    let off_config = LanguageModelConfig {
        max_tokens: Some(256),
        reasoning: Some(ReasoningConfig::Off),
        ..LanguageModelConfig::default()
    };
    let request = GenerateRequest {
        model: &reasoning_model,
        messages: &reasoning_messages,
        config: &off_config,
    };
    let r = lm.generate(request).await.expect("reasoning off generate");
    println!("Content: {}", r.content);
    if let Some(u) = &r.usage {
        println!(
            "Usage: {} input, {} output tokens",
            u.input_tokens, u.output_tokens
        );
    }
    assert!(
        !r.content.is_empty(),
        "reasoning-off call produced no content"
    );

    // --- Vision: image via inline base64 bytes (data: URI) ---
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
        model: &vision_model,
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
