// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::expect_used,
    reason = "examples exist to demo the API and print results to the terminal; the workspace \
              bans on print_*/expect target production code, not example binaries"
)]

//! Test `BedrockMantleProvider` against the real AWS Bedrock Mantle endpoint family.
//!
//! Usage (with AWS credentials):
//!   cargo run -p modelplease --example
//! bedrock_mantle_test
//!
//! Currently exercises:
//! - `list_models()` against `us-west-2` (default region; open-weight catalog).
//! - `generate()` against `mistral.ministral-3-8b-instruct` (Chat Completions surface, us-west-2).
//! - `generate_stream()` against `deepseek.v3.2` (Chat Completions surface, SSE, us-west-2).
//! - `generate()` against `anthropic.claude-haiku-4-5` (Anthropic Messages surface, us-east-1).
//! - `generate()` against `openai.gpt-5.5` (Responses surface, reasoning=low, us-east-2).
//!
//! Env overrides:
//! - `BEDROCK_MANTLE_DEFAULT_REGION` (default `us-west-2`)
//! - `BEDROCK_MANTLE_OPENAI_GPT5_REGION` (default `us-east-2`)
//! - `BEDROCK_MANTLE_ANTHROPIC_REGION` (default `us-east-1`)
//! - `BEDROCK_MANTLE_API_KEY` — if set, uses Bearer auth instead of SigV4.

use std::sync::Arc;

use aws_config::BehaviorVersion;
mod common;

use common::build_http_client;
use futures::StreamExt;
use modelplease::{
    BedrockMantleAuth, BedrockMantleProvider, BedrockMantleProviderConfig,
    BedrockMantleProviderDeps, GenerateRequest, LanguageModelConfig, LanguageModelProvider,
    Message, ModelId, ReasoningConfig, ReasoningEffort, RetryConfig,
};

#[tokio::main]
async fn main() {
    let default_region = std::env::var("BEDROCK_MANTLE_DEFAULT_REGION")
        .unwrap_or_else(|_| BedrockMantleProviderConfig::DEFAULT_REGION.to_owned());
    let openai_gpt5_region = std::env::var("BEDROCK_MANTLE_OPENAI_GPT5_REGION")
        .unwrap_or_else(|_| BedrockMantleProviderConfig::DEFAULT_OPENAI_GPT5_REGION.to_owned());
    let anthropic_region = std::env::var("BEDROCK_MANTLE_ANTHROPIC_REGION")
        .unwrap_or_else(|_| BedrockMantleProviderConfig::DEFAULT_ANTHROPIC_REGION.to_owned());

    let auth = if let Ok(key) = std::env::var("BEDROCK_MANTLE_API_KEY") {
        println!("Auth: Bedrock API key (Bearer)");
        BedrockMantleAuth::ApiKey(key)
    } else {
        println!("Auth: AWS SigV4 (default credential chain)");
        let sdk_config = aws_config::defaults(BehaviorVersion::latest())
            .region(aws_config::Region::new(default_region.clone()))
            .load()
            .await;
        let credentials_provider = sdk_config
            .credentials_provider()
            .expect("AWS credential chain returned no credentials_provider; set BEDROCK_MANTLE_API_KEY or run under aws-vault");
        BedrockMantleAuth::Sigv4 {
            credentials_provider,
        }
    };

    let client = Arc::new(build_http_client().expect("build reqwest client"));
    let provider = BedrockMantleProvider::new(
        BedrockMantleProviderDeps { client, auth },
        BedrockMantleProviderConfig {
            default_region: default_region.clone(),
            openai_gpt5_region: openai_gpt5_region.clone(),
            anthropic_region: anthropic_region.clone(),
            retry_config: RetryConfig::default(),
        },
    );

    println!(
        "Regions: default={default_region} openai_gpt5={openai_gpt5_region} anthropic={anthropic_region}"
    );

    println!(
        "\n=== list_models (merged: {default_region} + {openai_gpt5_region} + {anthropic_region}) ===",
    );
    let models = provider.list_models().await.expect("list_models");
    println!("returned {} models", models.len());
    for m in models.iter().take(8) {
        println!("  - {}", m.id.as_str());
    }
    if models.len() > 8 {
        println!("  ... ({} more)", models.len() - 8);
    }

    let config = LanguageModelConfig {
        temperature: Some(0.0),
        max_tokens: Some(64),
        ..Default::default()
    };
    let messages = vec![
        Message::system("You are a helpful assistant. Be concise."),
        Message::user("What is the capital of France? Answer in one word."),
    ];

    println!("\n=== generate (mistral.ministral-3-8b-instruct, ChatCompletions) ===");
    let model = ModelId::new("mistral.ministral-3-8b-instruct");
    match provider
        .generate(GenerateRequest {
            model: &model,
            messages: &messages,
            config: &config,
        })
        .await
    {
        Ok(response) => {
            println!("content: {}", response.content);
            if let Some(usage) = &response.usage {
                println!(
                    "usage:   input={} output={} cache_read={} cache_create={}",
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cache_read_input_tokens,
                    usage.cache_creation_input_tokens
                );
            }
            if let Some(stop) = &response.stop_reason {
                println!("stop:    {stop:?}");
            }
        }
        Err(e) => eprintln!("generate failed: {e}"),
    }

    println!("\n=== generate_stream (deepseek.v3.2, ChatCompletions SSE) ===");
    let model = ModelId::new("deepseek.v3.2");
    match provider
        .generate_stream(GenerateRequest {
            model: &model,
            messages: &messages,
            config: &config,
        })
        .await
    {
        Ok(mut stream) => {
            let mut total = String::new();
            let mut final_usage = None;
            while let Some(delta) = stream.next().await {
                match delta {
                    Ok(d) => {
                        if !d.content.is_empty() {
                            total.push_str(&d.content);
                        }
                        if d.usage.is_some() {
                            final_usage = d.usage;
                        }
                        if d.is_final {
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!("stream error: {e}");
                        break;
                    }
                }
            }
            println!("streamed: {total}");
            if let Some(usage) = final_usage {
                println!(
                    "usage:    input={} output={}",
                    usage.input_tokens, usage.output_tokens
                );
            }
        }
        Err(e) => eprintln!("generate_stream failed: {e}"),
    }

    println!("\n=== generate (anthropic.claude-haiku-4-5, AnthropicMessages, us-east-1) ===");
    let model = ModelId::new("anthropic.claude-haiku-4-5");
    match provider
        .generate(GenerateRequest {
            model: &model,
            messages: &messages,
            config: &config,
        })
        .await
    {
        Ok(response) => {
            println!("content: {}", response.content);
            if let Some(usage) = &response.usage {
                println!(
                    "usage:   input={} output={} cache_read={} cache_create={}",
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cache_read_input_tokens,
                    usage.cache_creation_input_tokens
                );
            }
            if let Some(stop) = &response.stop_reason {
                println!("stop:    {stop:?}");
            }
        }
        Err(e) => eprintln!("generate failed: {e}"),
    }

    println!("\n=== generate (openai.gpt-5.5, Responses, reasoning=low, us-east-2) ===");
    let model = ModelId::new("openai.gpt-5.5");
    // Use reasoning=low for cheapest gpt-5.5 call; temperature is stripped automatically when
    // reasoning is on (gpt-5 rejects temperature with non-default reasoning). max_tokens here
    // maps to Responses' max_output_tokens.
    let gpt5_config = LanguageModelConfig {
        max_tokens: Some(128),
        reasoning: Some(ReasoningConfig::Adaptive {
            effort: ReasoningEffort::Low,
        }),
        ..Default::default()
    };
    match provider
        .generate(GenerateRequest {
            model: &model,
            messages: &messages,
            config: &gpt5_config,
        })
        .await
    {
        Ok(response) => {
            println!("content: {}", response.content);
            if let Some(usage) = &response.usage {
                println!(
                    "usage:   input={} output={} cache_read={}",
                    usage.input_tokens, usage.output_tokens, usage.cache_read_input_tokens,
                );
            }
            if let Some(stop) = &response.stop_reason {
                println!("stop:    {stop:?}");
            }
        }
        Err(e) => eprintln!("generate failed: {e}"),
    }
}
