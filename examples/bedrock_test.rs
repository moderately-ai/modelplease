// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "examples exist to demo the API and print results to the terminal; the workspace \
              bans on print_*/unwrap/expect target production code, not example binaries"
)]

//! Test `BedrockProvider` against the real AWS Bedrock API.
//!
//! Usage (with AWS credentials):
//!   cargo run -p modelplease --example bedrock_test
//!
//! BEDROCK_REGION (default us-east-1) and BEDROCK_MODEL (default claude
//! haiku 4.5 inference profile) override the region / model.

use aws_config::BehaviorVersion;
mod common;

use common::build_http_client;
use futures::StreamExt;
use modelplease::{
    BedrockProvider, BedrockProviderConfig, BedrockProviderDeps, ContentPart, GenerateRequest,
    LanguageModelConfig, LanguageModelProvider, MediaSource, MediaType, Message, ModelId,
    ReasoningConfig, ReasoningEffort, ResponseFormat, RetryConfig, Role,
};

/// Standard test image used across the four provider examples: the
/// canonical Lenna PNG hosted on Wikimedia. Bedrock's `ImageBlock`
/// requires raw bytes or an S3 URI — no fetch-by-URL — so the example
/// downloads it once and hands bytes to the provider's
/// `MediaSource::InlineBytes` translation.
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
    let region = std::env::var("BEDROCK_REGION").ok();
    let model_id = std::env::var("BEDROCK_MODEL")
        .unwrap_or_else(|_| "us.anthropic.claude-haiku-4-5-20251001-v1:0".to_owned());

    // AWS SDK client construction lives in the composition root in
    // Match the client construction used by production consumers.
    // examples replicate that here so the provider stays sync.
    let mut loader = aws_config::defaults(BehaviorVersion::latest());
    if let Some(ref r) = region {
        loader = loader.region(aws_config::Region::new(r.clone()));
    }
    let sdk_config = loader.load().await;
    let provider = BedrockProvider::new(
        BedrockProviderDeps {
            runtime_client: aws_sdk_bedrockruntime::Client::new(&sdk_config),
            control_client: aws_sdk_bedrock::Client::new(&sdk_config),
        },
        BedrockProviderConfig {
            region: region.clone(),
            retry_config: RetryConfig::default(),
        },
    );
    let model = ModelId::new(model_id);
    println!(
        "Region: {} (resolved from {})",
        provider.region().unwrap_or("<sdk chain>"),
        if region.is_some() {
            "BEDROCK_REGION"
        } else {
            "AWS_REGION / profile / IMDS"
        }
    );
    println!("Model:  {}", model.as_str());

    let config = LanguageModelConfig {
        temperature: Some(0.0),
        max_tokens: Some(256),
        ..Default::default()
    };
    let messages = vec![
        Message::system("You are a helpful assistant. Be concise."),
        Message::user("What is the capital of France? Answer in one word."),
    ];

    println!("\n=== list_models ===");
    let models = provider.list_models().await.expect("list_models");
    println!("returned {} models", models.len());
    for m in models.iter().take(5) {
        println!(
            "  - {} ({})",
            m.id.as_str(),
            m.display_name.as_deref().unwrap_or("-")
        );
    }
    if models.len() > 5 {
        println!("  ... ({} more)", models.len() - 5);
    }
    assert!(!models.is_empty(), "list_models returned empty catalog");

    println!("\n=== Non-streaming ===");
    let request = GenerateRequest {
        model: &model,
        messages: &messages,
        config: &config,
    };
    let r = provider
        .generate(request)
        .await
        .expect("non-streaming generate");
    println!("Model:   {:?}", r.model);
    println!("Content: {}", r.content);
    if let Some(u) = &r.usage {
        println!(
            "Usage:   {} input, {} output tokens",
            u.input_tokens, u.output_tokens
        );
    }
    assert!(
        !r.content.is_empty(),
        "non-streaming content must be non-empty"
    );

    println!("\n=== Streaming ===");
    let request = GenerateRequest {
        model: &model,
        messages: &messages,
        config: &config,
    };
    let mut stream = provider
        .generate_stream(request)
        .await
        .expect("generate_stream");
    print!("Content: ");
    let mut got_any = false;
    while let Some(result) = stream.next().await {
        let delta = result.expect("streaming delta error");
        print!("{}", delta.content);
        if !delta.content.is_empty() {
            got_any = true;
        }
        if delta.is_final {
            if let Some(u) = delta.usage {
                println!(
                    " [done — {} input / {} output tokens]",
                    u.input_tokens, u.output_tokens
                );
            } else {
                println!(" [done]");
            }
        }
    }
    assert!(got_any, "streaming produced no content");

    println!("\n=== JsonSchema (structured output) ===");
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"},
            "population_millions": {"type": "number"}
        },
        "required": ["city", "country", "population_millions"],
        "additionalProperties": false
    });
    let json_config = LanguageModelConfig {
        temperature: Some(0.0),
        max_tokens: Some(256),
        response_format: ResponseFormat::JsonSchema {
            name: "city_fact".into(),
            schema,
            strict: true,
        },
        ..Default::default()
    };
    let json_messages = vec![
        Message::system("You answer with structured JSON only."),
        Message::user("Give me a fact about Paris."),
    ];
    let request = GenerateRequest {
        model: &model,
        messages: &json_messages,
        config: &json_config,
    };
    let r = provider
        .generate(request)
        .await
        .expect("json_schema generate");
    println!("Content: {}", r.content);
    let parsed: serde_json::Value =
        serde_json::from_str(&r.content).expect("json_schema response must be valid JSON");
    println!(
        "Parsed:  {}",
        serde_json::to_string_pretty(&parsed).unwrap()
    );
    if let Some(u) = &r.usage {
        println!(
            "Usage:   {} input, {} output tokens",
            u.input_tokens, u.output_tokens
        );
    }

    // --- Vision: image via inline bytes ---
    //
    // Bedrock's ContentBlock::Image takes raw bytes or an S3 URI;
    // there is no fetch-by-URL path. The provider's translation maps
    // `MediaSource::InlineBytes` to `ImageSource::Bytes(Blob)`.
    println!("\n=== Vision (image bytes, MediaSource::InlineBytes) ===");
    let http_client = build_http_client().expect("build reqwest client for image fetch");
    let (mime, image_bytes) = fetch_sample_image(&http_client).await;
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
    let r = provider
        .generate(request)
        .await
        .expect("vision bytes generate");
    println!("Content: {}", r.content);
    if let Some(u) = &r.usage {
        println!(
            "Usage:   {} input, {} output tokens",
            u.input_tokens, u.output_tokens
        );
    }
    assert!(!r.content.is_empty(), "vision content must be non-empty");

    // --- Capability lookup ---
    //
    // The new `LanguageModelProvider::capabilities` accessor surfaces
    // per-model media support, suitable for application configuration
    // preflight to gate unsupported requests
    // before the first execute.
    println!("\n=== Capability lookup ===");
    match provider.capabilities(&model) {
        Some(caps) => {
            println!("Model:  {}", caps.model_id);
            if caps.media_support.is_empty() {
                println!("Media:  (text-only)");
            } else {
                for (kind, support) in &caps.media_support {
                    println!(
                        "Media:  {kind:?} → sources={:?}, formats={:?}, max_bytes={:?}",
                        support.sources, support.formats, support.max_bytes
                    );
                }
            }
            match &caps.reasoning {
                Some(r) => println!(
                    "Reasoning: modes={:?}, efforts={:?}, manual_range={:?}",
                    r.supported_modes, r.supported_efforts, r.manual_budget_range
                ),
                None => println!("Reasoning: (not supported)"),
            }
        }
        None => eprintln!("(no capability metadata for `{}`)", model.as_str()),
    }

    // --- Reasoning: adaptive against a reasoning-capable Claude model ---
    // Sonnet 4.6 supports both adaptive and manual; this exercises the
    // adaptive path via additionalModelRequestFields.thinking. Override
    // with BEDROCK_REASONING_MODEL=<id> (e.g.
    // us.anthropic.claude-opus-4-7) for the xhigh/max enum.
    let reasoning_model_id = std::env::var("BEDROCK_REASONING_MODEL")
        .unwrap_or_else(|_| "us.anthropic.claude-sonnet-4-6".to_owned());
    let reasoning_model = ModelId::new(reasoning_model_id);
    println!(
        "\n=== Reasoning (adaptive, effort=high) on {} ===",
        reasoning_model.as_str()
    );
    let reasoning_config = LanguageModelConfig {
        // Anthropic rejects `temperature` when thinking is on; leave unset.
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
    let r = provider
        .generate(request)
        .await
        .expect("reasoning adaptive generate");
    if let Some(thinking) = &r.thinking {
        println!("Thinking ({} chars).", thinking.len());
    } else {
        println!("Thinking: <none surfaced>");
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
        "adaptive reasoning produced empty content"
    );

    // --- Reasoning: manual budget ---
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
    let r = provider
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
        "manual reasoning produced empty content"
    );
}
