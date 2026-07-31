// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#![expect(
    clippy::print_stdout,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    reason = "examples exist to demo the API and print results to the terminal; the workspace \
              bans on print_*/unwrap/expect/panic target production code, not example binaries \
              that use panic! to surface 'forgot to pull this model' setup errors loudly"
)]

//! Quick test: call a local Ollama instance using `OllamaLanguageModel`.
//!
//! Tests non-streaming, streaming, structured output, and a vision
//! demo against a local llava model. Ollama does not fetch URLs from
//! the daemon — image bytes ride inline as a base64 `data:` URI on the
//! OpenAI-compatible content-part array.
//!
//! Usage:
//!   cargo run -p modelplease --example ollama_test
//!
//! `OLLAMA_VISION_MODEL` overrides the model used for the vision demo
//! (default `llava:latest`). Any model the local prefix classifier
//! marks as vision-capable works:
//!   - `llava:7b`, `bakllava`, `llama3.2-vision`, `qwen2.5-vl`, ...
//!   - `gemma3:4b` and larger (the 1B Gemma variants are text-only)
//!   - `gemma4:latest` — Google's multimodal Gemma 4 release
//!
//! Pull whichever model you prefer with `ollama pull <name>` first.

use std::sync::Arc;

mod common;

use common::build_http_client;
use futures::StreamExt;
use modelplease::{
    ContentPart, GenerateRequest, LanguageModelConfig, LanguageModelProvider, MediaSource,
    MediaType, Message, ModelId, OllamaConfig, OllamaDeps, OllamaLanguageModel, ReasoningConfig,
    ReasoningEffort, ResponseFormat, Role,
};

/// Standard test image used across the four provider examples: the
/// canonical Lenna PNG hosted on Wikimedia. The Ollama daemon doesn't
/// fetch external URLs — image bytes ride inline as a base64 `data:`
/// URI on the OpenAI-compatible content-part array.
const SAMPLE_IMAGE_URL: &str =
    "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png";
const SAMPLE_IMAGE_MIME: &str = "image/png";

/// Wikimedia's CDN rejects requests with reqwest's default User-Agent.
const SAMPLE_IMAGE_USER_AGENT: &str =
    "modelplease-example/0.1 (https://github.com/moderately-ai/modelplease)";

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
    let client = Arc::new(build_http_client().expect("build reqwest client"));
    let lm = OllamaLanguageModel::new(
        OllamaDeps {
            client: Arc::clone(&client),
        },
        OllamaConfig::default(),
    );
    let model = ModelId::new("gemma4:latest");
    let vision_model = ModelId::new(
        std::env::var("OLLAMA_VISION_MODEL").unwrap_or_else(|_| "llava:latest".to_owned()),
    );

    // Config with reasoning explicitly off for text generation.
    let config = LanguageModelConfig {
        temperature: Some(0.0),
        max_tokens: Some(8192),
        reasoning: Some(ReasoningConfig::Off),
        ..Default::default()
    };

    // --- Non-streaming ---
    println!("=== Non-streaming ===");
    let messages = vec![
        Message::system("You are a helpful assistant. Be concise."),
        Message::user("What is the capital of France? Answer in one word."),
    ];
    let request = GenerateRequest {
        model: &model,
        messages: &messages,
        config: &config,
    };
    let r = lm.generate(request).await.expect("non-streaming generate");
    println!("Content: {}", r.content);
    assert!(
        !r.content.is_empty(),
        "non-streaming content must be non-empty"
    );

    // --- Streaming ---
    println!("\n=== Streaming ===");
    let request = GenerateRequest {
        model: &model,
        messages: &messages,
        config: &config,
    };
    let mut stream = lm.generate_stream(request).await.expect("generate_stream");
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

    // --- Structured output: json_object ---
    // OllamaLanguageModel automatically strips reasoning_effort when
    // response_format is set, so we can use the same config.
    println!("\n=== Structured: json_object ===");
    let json_config = LanguageModelConfig {
        response_format: ResponseFormat::JsonObject,
        ..config.clone()
    };
    let json_messages = vec![Message::user(
        "Return a JSON object with fields \"name\" (string) and \"age\" (integer) for a fictional person.",
    )];
    let request = GenerateRequest {
        model: &model,
        messages: &json_messages,
        config: &json_config,
    };
    let r = lm.generate(request).await.expect("json_object generate");
    println!("Content: {}", r.content);
    let parsed: serde_json::Value =
        serde_json::from_str(&r.content).expect("json_object response must be valid JSON");
    println!("Parsed: {parsed}");

    // --- Structured output: json_schema ---
    println!("\n=== Structured: json_schema ===");
    let schema_config = LanguageModelConfig {
        response_format: ResponseFormat::JsonSchema {
            name: "person".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            }),
            strict: true,
        },
        ..config.clone()
    };
    let schema_messages = vec![Message::user("Create a fictional person.")];
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

    // --- Vision: image via inline base64 bytes ---
    //
    // The Ollama daemon doesn't fetch external URLs; the provider's
    // capability table excludes `SourceKind::Url` so `validate_request`
    // rejects URL sources before they reach the wire. Inline bytes are
    // emitted as a `data:` URI on the OpenAI-compatible content-part
    // array (shared with the OpenAI translation path).
    println!(
        "\n=== Vision (image bytes via {}) ===",
        vision_model.as_str()
    );
    let vision_config = LanguageModelConfig {
        temperature: Some(0.0),
        max_tokens: Some(256),
        // Most Ollama vision models don't expose a reasoning_effort
        // knob; leave it unset so the upstream defaults apply.
        ..Default::default()
    };
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
        config: &vision_config,
    };
    let r = lm.generate(request).await.unwrap_or_else(|e| {
        panic!(
            "vision generate failed (is `{}` pulled? `ollama pull {}`): {e}",
            vision_model.as_str(),
            vision_model.as_str()
        )
    });
    println!("Content: {}", r.content);
    assert!(!r.content.is_empty(), "vision content must be non-empty");

    // --- Reasoning: adaptive effort on a thinking-capable model ---
    // Override via OLLAMA_REASONING_MODEL (e.g. qwen3:8b, deepseek-r1:7b)
    // — defaults to qwen3 which the cap table classifies as thinking.
    let reasoning_model = ModelId::new(
        std::env::var("OLLAMA_REASONING_MODEL").unwrap_or_else(|_| "qwen3:8b".to_owned()),
    );
    println!(
        "\n=== Reasoning (adaptive, effort=high) on {} ===",
        reasoning_model.as_str()
    );
    let reasoning_config = LanguageModelConfig {
        max_tokens: Some(8192),
        reasoning: Some(ReasoningConfig::Adaptive {
            effort: ReasoningEffort::High,
        }),
        ..Default::default()
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
    let r = lm.generate(request).await.unwrap_or_else(|e| {
        panic!(
            "reasoning generate failed (is `{}` pulled? `ollama pull {}`): {e}",
            reasoning_model.as_str(),
            reasoning_model.as_str()
        )
    });
    if let Some(thinking) = &r.thinking {
        println!("Thinking ({} chars).", thinking.len());
    }
    println!("Content: {}", r.content);
    if let Some(u) = &r.usage {
        println!(
            "Usage: {} input, {} output tokens",
            u.input_tokens, u.output_tokens
        );
    }
    assert!(!r.content.is_empty(), "reasoning content must be non-empty");
}
