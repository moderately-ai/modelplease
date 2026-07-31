// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core language model types, traits, and provider implementations.
//!
//! Provides the foundational types used across the predict engine and
//! any other crate that communicates with language models. Includes
//! concrete implementations for Anthropic and `OpenAI` APIs, plus a
//! spec-compliant SSE parser for streaming responses.
//!
//! ## Providers
//!
//! - [`AnthropicLanguageModel`] — Anthropic Messages API
//! - [`OpenAiLanguageModel`] — `OpenAI` Chat Completions API (also works with any
//!   `OpenAI`-compatible endpoint via [`OpenAiConfig::base_url`])
//! - [`OllamaLanguageModel`] — local Ollama daemon, OpenAI-compat surface
//! - [`BedrockProvider`] — AWS Bedrock via Converse / ConverseStream
//! - [`DummyLM`] — test mock
//!
//! ## OpenAI-Compatible Providers
//!
//! The [`OpenAiLanguageModel`] works out of the box with any provider that
//! implements the `OpenAI` Chat Completions API:
//!
//! ```no_run
//! use std::sync::Arc;
//! use modelplease::{ApiKey, OpenAiConfig, OpenAiDeps, OpenAiLanguageModel, RetryConfig};
//!
//! # fn build() -> Result<OpenAiLanguageModel, modelplease::ApiKeyError> {
//! let lm = OpenAiLanguageModel::new(
//!     OpenAiDeps { client: Arc::new(reqwest::Client::new()) },
//!     OpenAiConfig {
//!         api_key: ApiKey::parse("local")?,
//!         base_url: "http://localhost:8080/v1".to_owned(),
//!         retry_config: RetryConfig::default(),
//!     },
//! );
//! # Ok(lm)
//! # }
//! ```
//!
//! ## Quick Start
//!
//! ```rust
//! use modelplease::{ContentPart, HttpsUrl, MediaSource, Message, Role};
//!
//! // Simple text message
//! let _msg = Message::user("What is the capital of France?");
//!
//! // Message with mixed content parts
//! let url = HttpsUrl::parse("https://example.com/photo.jpg").unwrap();
//! let _msg = Message::with_parts(
//!     Role::User,
//!     vec![
//!         ContentPart::text("Describe this image:"),
//!         ContentPart::image(MediaSource::Url { url }),
//!     ],
//! );
//! ```
//!
//! ## Modules
//!
//! - Messages — `Message`, `Role`, `ContentPart`
//! - Providers — `LanguageModelProvider`, `GenerateRequest`
//! - Configuration — `LanguageModelConfig`, `ResponseFormat`
//! - Responses — `LanguageModelResponse`, `StreamDelta`, `Usage`
//! - Errors — `LanguageModelError`
//! - Retry — `RetryConfig`, `with_retry`
//! - SSE — `SseEvent`, `parse_sse_stream`
//! - Provider implementations — `AnthropicLanguageModel`, `OpenAiLanguageModel`, and `DummyLM`

// Crate-local promotion of the workspace `warn` baseline to `deny` —
// production sites are clean apart from one `unreachable!` on an
// impossible-by-construction match arm which carries a scoped
// `#[expect(clippy::unreachable, reason = ...)]`.
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

#[cfg(feature = "anthropic")]
pub(crate) mod anthropic;
#[cfg(any(feature = "anthropic", feature = "bedrock-mantle"))]
pub(crate) mod anthropic_wire;
#[cfg(feature = "bedrock")]
pub(crate) mod bedrock;
#[cfg(feature = "bedrock-mantle")]
pub(crate) mod bedrock_mantle;
pub(crate) mod capabilities;
pub(crate) mod config;
pub(crate) mod dummy;
pub(crate) mod error;
#[cfg(any(
    feature = "anthropic",
    feature = "openai",
    feature = "ollama",
    feature = "bedrock-mantle"
))]
mod http_error;
pub(crate) mod identifiers;
pub(crate) mod media;
pub(crate) mod message;
#[cfg(feature = "ollama")]
pub(crate) mod ollama;
#[cfg(feature = "openai")]
pub(crate) mod openai;
#[cfg(any(feature = "openai", feature = "ollama", feature = "bedrock-mantle"))]
pub(crate) mod openai_wire;
pub(crate) mod provider;
pub(crate) mod response;
pub(crate) mod retry;
pub(crate) mod sse;
#[cfg(any(
    feature = "anthropic",
    feature = "openai",
    feature = "ollama",
    feature = "bedrock",
    feature = "bedrock-mantle"
))]
pub(crate) mod streaming_timing;

#[cfg(feature = "anthropic")]
pub use anthropic::{AnthropicConfig, AnthropicDeps, AnthropicLanguageModel};
#[cfg(feature = "bedrock")]
pub use bedrock::{BedrockProvider, BedrockProviderConfig, BedrockProviderDeps};
#[cfg(feature = "bedrock-mantle")]
pub use bedrock_mantle::{
    BedrockMantleAuth, BedrockMantleProvider, BedrockMantleProviderConfig,
    BedrockMantleProviderDeps,
};
pub use capabilities::{
    AcceptsAudioBytes, AcceptsDocumentBytes, AcceptsImageBytes, AcceptsImageS3, AcceptsImageUrl,
    AcceptsVideoBytes, AcceptsVideoS3, CapabilityError, MediaKind, MediaSupport, ModelCapabilities,
    ReasoningCapability, ReasoningMode, ReasoningParamConflicts, ReasoningValidationError,
};
pub use config::{
    CacheTtl, InvalidReasoningEffort, LanguageModelConfig, LatencyMode, PromptCaching,
    ReasoningConfig, ReasoningEffort, ResponseFormat,
};
pub use dummy::DummyLM;
pub use error::LanguageModelError;
pub use identifiers::{ApiKey, ApiKeyError, ModelId};
pub use media::{
    AwsAccountId, AwsAccountIdError, HttpsUrl, HttpsUrlError, MediaSource, MediaType,
    MediaTypeError, ProviderFileId, ProviderFileIdError, S3Uri, S3UriError, SourceKind,
    all_source_kinds,
};
pub use message::{ContentPart, Message, Role};
#[cfg(feature = "ollama")]
pub use ollama::{OllamaConfig, OllamaDeps, OllamaLanguageModel};
#[cfg(feature = "openai")]
pub use openai::{OpenAiConfig, OpenAiDeps, OpenAiLanguageModel};
pub use provider::{ChatModelInfo, GenerateRequest, LanguageModelProvider, ResponseFormatKind};
pub use response::{LanguageModelResponse, StopReason, StreamDelta, Usage};
pub use retry::{RetryConfig, with_retry};
pub use sse::{SseEvent, SseStream, parse_sse_stream};
