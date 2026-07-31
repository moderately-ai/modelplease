// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! AWS Bedrock Mantle provider — multi-surface OpenAI / Anthropic-compatible inference against
//! the `bedrock-mantle.{region}.api.aws` endpoint family.
//!
//! Mantle is the second Bedrock endpoint family (alongside `bedrock-runtime` / Converse). It
//! serves a disjoint catalog: open-weight models (DeepSeek, Qwen, Mistral, Gemma, Kimi, GLM,
//! MiniMax, Nemotron, Palmyra, Voxtral), OpenAI's open-weight `gpt-oss-*`, the proprietary
//! `openai.gpt-5.{4,5}` family, and Anthropic Claude variants. Three API surfaces share the same
//! host:
//!
//! - `POST {base}/v1/chat/completions` — OpenAI Chat Completions (~36 of 42 models).
//! - `POST {base}/openai/v1/responses` — OpenAI Responses (gpt-5.x; called in stateless mode).
//! - `POST {base}/anthropic/v1/messages` — Anthropic Messages (Claude on Mantle).
//!
//! The provider routes per request based on model-id prefix:
//! - `openai.gpt-5.*` → Responses.
//! - `anthropic.*` → Anthropic Messages.
//! - everything else → Chat Completions.
//!
//! Auth is AWS SigV4 against service `bedrock` by default; a Bedrock API key (Bearer) is the
//! alternative for OpenAI-SDK-compatible flows.
//!
//! Region routing is per-model-family because Mantle's catalog varies by region — gpt-5.x lives
//! only on us-east-2, the opus-class Claude variants only on us-east-1, Haiku 4.5 on us-east-1 and
//! us-west-2, the open-weight catalog in every region. The provider holds three separate regions
//! and selects per request from the router.
//!
//! Phase 1 lands the skeleton: signing, region selection, single-region [`list_models`]. Phases
//! 2–6 fill in the three adapters and the capability table.
//!
//! # Implicit server-side prompt caching
//!
//! Mantle's OpenAI Chat Completions surface does **implicit, server-side prompt caching** for many
//! model families independent of any explicit caching API. The
//! [`PromptCaching::Off`](crate::PromptCaching::Off) flag on
//! [`LanguageModelConfig`](crate::LanguageModelConfig) suppresses our own
//! [`ContentPart::CacheBreakpoint`](crate::ContentPart::CacheBreakpoint) markers from being
//! emitted upstream — it does **not** disable the inference server's automatic caching. A repeated
//! prompt prefix can be cached and served from cache on subsequent calls regardless of the flag.
//!
//! The June 2026 benchmark sweep observed cache hit rates of 80–100% on roughly half of the open-
//! weight catalog (Mistral Devstral / Ministral / Magistral, MiniMax M2 family, Gemma 3 4B/12B,
//! GLM-4.7 / 4.7-flash, Qwen3 Coder 30B/480B, and others) with `PromptCaching::Off` set on every
//! request. Consumers reading
//! [`Usage::cache_read_input_tokens`](crate::Usage::cache_read_input_tokens) on Mantle responses
//! may therefore see non-zero values regardless of their configuration. The
//! [`Usage`](crate::Usage) invariant still holds: `input_tokens` is the uncached portion;
//! `cache_read_input_tokens` is what was served from cache; sum them for the true prompt size.
//!
//! Bedrock Runtime / Converse, by contrast, has not been observed to surface a non-zero
//! `cache_read_input_tokens` on the same prompts — either Converse does not implicitly cache, or
//! the AWS SDK does not surface the counter via
//! [`TokenUsage`](aws_sdk_bedrockruntime::types::TokenUsage). See
//! [`crate::bedrock::response::bedrock_usage`] for the matching note on the Runtime side.

pub(super) mod capabilities;
pub(super) mod responses_wire;
#[cfg(test)]
mod tests;

use std::{pin::Pin, sync::Arc, time::Duration};

use async_trait::async_trait;
use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider};
use aws_sigv4::{
    http_request::{SignableBody, SignableRequest, SigningSettings, sign},
    sign::v4,
};
use aws_smithy_runtime_api::client::identity::Identity;
use capabilities::MODEL_CAPABILITIES;
use futures::{Stream, stream::StreamExt};
use moka::future::Cache;
use responses_wire::{
    ResponsesResponse, build_responses_request, convert_responses_sse_event,
    parse_responses_response,
};
use serde::Deserialize;

use crate::{
    anthropic_wire::{
        AnthropicResponse, build_request as build_anthropic_request, convert_anthropic_sse_event,
        map_anthropic_error, parse_anthropic_response,
    },
    capabilities::ModelCapabilities,
    config::{LanguageModelConfig, PromptCaching},
    error::LanguageModelError,
    identifiers::ModelId,
    message::Message,
    openai_wire::{
        OpenAiResponse, OpenAiStreamOptions, build_request as build_openai_request,
        convert_openai_sse_event, parse_openai_response,
    },
    provider::{ChatModelInfo, GenerateRequest, LanguageModelProvider, ResponseFormatKind},
    response::{LanguageModelResponse, StreamDelta},
    retry::{RetryConfig, with_retry},
    sse::parse_sse_stream,
};

/// AWS service name SigV4 signs against for the Mantle endpoint family.
const SIGV4_SERVICE_NAME: &str = "bedrock";

/// `list_models` cache TTL — 1 h, matching the other providers.
const LIST_MODELS_TTL_SECS: u64 = 3600;

/// Headers required on the Anthropic Messages surface (`/anthropic/v1/messages`).
///
/// `anthropic-version` is mandatory on Mantle just like on direct Anthropic — the Mantle proxy
/// does NOT auto-select a version, and a missing header yields HTTP 400 with
/// `anthropic_version: Field required`. We pin to the same `2023-06-01` version the direct
/// Anthropic provider uses so wire-shape parity holds across both surfaces.
const ANTHROPIC_HEADERS: &[(&str, &str)] = &[("anthropic-version", "2023-06-01")];

/// Per-call input for [`BedrockMantleProvider::send_signed`].
///
/// Bundles the destination (`method` + `url` + `region`), the request body, and any
/// surface-specific headers (e.g. `anthropic-version` for the Anthropic Messages surface).
/// Region is part of this struct because SigV4 signing is region-scoped — the Mantle endpoint
/// host names embed the region, and the SigV4 credential scope must match it.
struct SignedRequest<'a> {
    method: reqwest::Method,
    url: &'a str,
    region: &'a str,
    body: Vec<u8>,
    /// Extra headers added to the request beyond `content-type: application/json`. Included in
    /// the SigV4 signable set so the signature matches the wire.
    extra_headers: &'a [(&'a str, &'a str)],
}

/// AWS Bedrock Mantle language-model provider.
///
/// Wraps the three on-the-wire surfaces Mantle exposes — OpenAI Chat Completions, OpenAI
/// Responses, Anthropic Messages — behind a single [`LanguageModelProvider`] impl. Per-call
/// routing picks the surface from the model-id prefix; see module docs for the table.
///
/// One instance serves every Mantle model in the upstream catalog. The model identifier rides
/// per-call on `GenerateRequest` rather than being baked into the struct, matching the other
/// HTTP-based providers in the crate.
pub struct BedrockMantleProvider {
    client: Arc<reqwest::Client>,
    auth: BedrockMantleAuth,
    config: BedrockMantleProviderConfig,
    /// Single-entry TTL cache for [`LanguageModelProvider::list_models`].
    /// `time_to_live = 1 h`, `max_capacity = 1`. moka's `try_get_with` coalesces concurrent
    /// misses to one upstream call.
    list_models_cache: Cache<(), Vec<ChatModelInfo>>,
}

/// Authentication for [`BedrockMantleProvider`].
///
/// `Sigv4` is the production default — same posture as `BedrockProvider` (IAM-aligned, no
/// long-lived secrets in env). `ApiKey` exposes a Bedrock-issued Bearer key for local-dev /
/// OpenAI-SDK-shaped clients that can't sign requests themselves.
pub enum BedrockMantleAuth {
    /// AWS SigV4 signing against service name `bedrock`. The credentials provider resolves
    /// through the standard AWS chain (env / instance profile / SSO / web-identity).
    Sigv4 {
        /// AWS credential provider — resolved per request, fast once the chain is warm.
        credentials_provider: SharedCredentialsProvider,
    },
    /// Bedrock API key applied as `Authorization: Bearer <key>`. The key is AWS-issued from
    /// the Bedrock console, distinct from any OpenAI / Anthropic key.
    ApiKey(String),
}

/// Injected dependencies for [`BedrockMantleProvider`].
///
/// The shared `Arc<reqwest::Client>` is constructed once at the application composition root
/// and cloned into every HTTP-based provider so the
/// macOS reqwest system-proxy trap fires at most once per process and the connection pool is
/// shared.
pub struct BedrockMantleProviderDeps {
    /// Shared reqwest client.
    pub client: Arc<reqwest::Client>,
    /// SigV4 credential provider or static Bedrock API key.
    pub auth: BedrockMantleAuth,
}

/// Per-family region routing + retry knobs for [`BedrockMantleProvider`].
///
/// Mantle's per-region catalog is asymmetric:
/// - `openai.gpt-5.{4,5}` lives only on `us-east-2`.
/// - `anthropic.claude-opus-{4-7,4-8}` lives only on `us-east-1`.
/// - `anthropic.claude-haiku-4-5` lives on `us-east-1` and `us-west-2`.
/// - The 30+ open-weight Chat Completions models live in every region.
///
/// Operators configure one region per family; [`DEFAULT_REGION`](Self::DEFAULT_REGION),
/// [`DEFAULT_OPENAI_GPT5_REGION`](Self::DEFAULT_OPENAI_GPT5_REGION), and
/// [`DEFAULT_ANTHROPIC_REGION`](Self::DEFAULT_ANTHROPIC_REGION) pick the smallest region set that
/// reaches every Mantle model.
pub struct BedrockMantleProviderConfig {
    /// Region for the broad open-weight + gpt-oss-* catalog routed via Chat Completions.
    pub default_region: String,
    /// Region for `openai.gpt-5.*` routed via Responses.
    pub openai_gpt5_region: String,
    /// Region for `anthropic.*` routed via Messages.
    pub anthropic_region: String,
    /// Retry config for transient errors (429 / 5xx).
    pub retry_config: RetryConfig,
}

impl BedrockMantleProviderConfig {
    /// Default region for the open-weight / Chat Completions catalog.
    pub const DEFAULT_REGION: &'static str = "us-west-2";
    /// Default region for the OpenAI gpt-5 family (Responses surface).
    pub const DEFAULT_OPENAI_GPT5_REGION: &'static str = "us-east-2";
    /// Default region for the Anthropic Messages surface.
    pub const DEFAULT_ANTHROPIC_REGION: &'static str = "us-east-1";
}

/// API-surface dispatch for [`BedrockMantleProvider`].
///
/// Routing is by model-id prefix — see the module-level docs for the prefix → surface mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Route {
    /// `POST /v1/chat/completions`.
    ChatCompletions,
    /// `POST /openai/v1/responses` in stateless mode (`store: false`).
    Responses,
    /// `POST /anthropic/v1/messages`.
    Messages,
}

impl Route {
    /// Pick the Mantle wire surface to dispatch to for the given model id.
    pub(crate) fn from_model_id(model: &str) -> Self {
        if model.starts_with("openai.gpt-5.") {
            Self::Responses
        } else if model.starts_with("anthropic.") {
            Self::Messages
        } else {
            Self::ChatCompletions
        }
    }
}

impl BedrockMantleProvider {
    /// Construct the provider. AWS credential-chain probing happens once at the application
    /// composition root and arrives wrapped in [`BedrockMantleAuth::Sigv4`]; this constructor
    /// itself stays sync.
    #[must_use]
    pub fn new(deps: BedrockMantleProviderDeps, config: BedrockMantleProviderConfig) -> Self {
        Self {
            client: deps.client,
            auth: deps.auth,
            config,
            list_models_cache: Cache::builder()
                .time_to_live(Duration::from_secs(LIST_MODELS_TTL_SECS))
                .max_capacity(1)
                .build(),
        }
    }

    /// `https://bedrock-mantle.{region}.api.aws` — the host portion of every Mantle URL.
    fn endpoint_base(region: &str) -> String {
        format!("https://bedrock-mantle.{region}.api.aws")
    }

    /// Pick the configured region for a given API-surface route.
    fn region_for(&self, route: Route) -> &str {
        match route {
            Route::Responses => &self.config.openai_gpt5_region,
            Route::Messages => &self.config.anthropic_region,
            Route::ChatCompletions => &self.config.default_region,
        }
    }

    /// Distinct configured regions across the three per-family knobs (default / openai_gpt5 /
    /// anthropic). When operators set the same region for multiple families the duplicate is
    /// dropped so [`list_models`](Self::list_models) doesn't fan out a redundant request.
    fn unique_regions(&self) -> Vec<&str> {
        let mut out: Vec<&str> = Vec::with_capacity(3);
        for r in [
            self.config.default_region.as_str(),
            self.config.openai_gpt5_region.as_str(),
            self.config.anthropic_region.as_str(),
        ] {
            if !out.contains(&r) {
                out.push(r);
            }
        }
        out
    }

    /// Adapter A — execute one non-streaming Chat Completions request against
    /// `{base}/v1/chat/completions`.
    ///
    /// Body shape and response parsing reuse [`crate::openai_wire`] — both providers speak the
    /// same Chat Completions wire format. We only swap auth (SigV4 instead of Bearer) and base
    /// URL (Mantle host instead of api.openai.com).
    async fn chat_completions_generate(
        &self,
        model: &str,
        messages: &[Message],
        config: &LanguageModelConfig,
        region: &str,
    ) -> Result<LanguageModelResponse, LanguageModelError> {
        let request_body = build_openai_request(
            model,
            messages,
            config,
            mantle_uses_completion_tokens(model),
        );
        let body_bytes = serde_json::to_vec(&request_body).map_err(|e| {
            LanguageModelError::provider(format!("failed to serialize chat-completions body: {e}"))
        })?;
        let url = format!("{}/v1/chat/completions", Self::endpoint_base(region));

        let response = self
            .send_signed(SignedRequest {
                method: reqwest::Method::POST,
                url: &url,
                region,
                body: body_bytes,
                extra_headers: &[],
            })
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = crate::http_error::read_error_body_or_warn(
                response,
                "bedrock-mantle",
                status.as_u16(),
            )
            .await;
            return Err(map_mantle_error(status.as_u16(), &body));
        }

        let api_response: OpenAiResponse = response.json().await.map_err(|e| {
            LanguageModelError::provider(format!("failed to parse chat-completions response: {e}"))
        })?;
        parse_openai_response(api_response)
    }

    /// Build the Anthropic Messages request body for the Mantle surface — same wire shape as
    /// direct Anthropic but with `cache_control` suppressed (Mantle rejects it; caching against
    /// Claude on Mantle is unavailable per AWS Mythos docs — operators wanting caching go through
    /// `bedrock-runtime` / Converse instead).
    fn build_messages_body(
        model: &str,
        messages: &[Message],
        config: &LanguageModelConfig,
    ) -> Result<Vec<u8>, LanguageModelError> {
        // Force PromptCaching::Off so build_request never emits cache_control. Mantle rejects
        // cache_control entirely on the Anthropic Messages surface — wrong-too-permissive here
        // would translate to a 400 response.
        let mut mantle_config = config.clone();
        mantle_config.prompt_caching = PromptCaching::Off;
        let (request, _needs_files_beta) = build_anthropic_request(model, messages, &mantle_config);
        serde_json::to_vec(&request).map_err(|e| {
            LanguageModelError::provider(format!("failed to serialize messages body: {e}"))
        })
    }

    /// Adapter C — execute one non-streaming Anthropic Messages request against
    /// `{base}/anthropic/v1/messages`.
    async fn messages_generate(
        &self,
        model: &str,
        messages: &[Message],
        config: &LanguageModelConfig,
        region: &str,
    ) -> Result<LanguageModelResponse, LanguageModelError> {
        let body_bytes = Self::build_messages_body(model, messages, config)?;
        let url = format!("{}/anthropic/v1/messages", Self::endpoint_base(region));
        let response = self
            .send_signed(SignedRequest {
                method: reqwest::Method::POST,
                url: &url,
                region,
                body: body_bytes,
                extra_headers: ANTHROPIC_HEADERS,
            })
            .await?;

        let status = response.status();
        if !status.is_success() {
            let headers = response.headers().clone();
            let body = crate::http_error::read_error_body_or_warn(
                response,
                "bedrock-mantle",
                status.as_u16(),
            )
            .await;
            return Err(map_anthropic_error(status.as_u16(), &body, &headers));
        }

        let api_response: AnthropicResponse = response.json().await.map_err(|e| {
            LanguageModelError::provider(format!("failed to parse messages response: {e}"))
        })?;
        parse_anthropic_response(api_response)
    }

    /// Adapter C streaming — execute a streaming Anthropic Messages request, returning the SSE
    /// stream projected onto [`StreamDelta`].
    async fn messages_stream(
        &self,
        model: &str,
        messages: &[Message],
        config: &LanguageModelConfig,
        region: &str,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamDelta, LanguageModelError>> + Send>>,
        LanguageModelError,
    > {
        // Set `stream: true` by serializing the request via a JSON-Value rewrite — build_request
        // sets `stream: None` since it doesn't know whether this is the streaming or
        // non-streaming path. Cheapest correction: deserialize → patch → reserialize.
        let mut body_value: serde_json::Value = serde_json::from_slice(&Self::build_messages_body(
            model, messages, config,
        )?)
        .map_err(|e| {
            LanguageModelError::provider(format!("failed to round-trip messages body: {e}"))
        })?;
        body_value["stream"] = serde_json::Value::Bool(true);
        let body_bytes = serde_json::to_vec(&body_value).map_err(|e| {
            LanguageModelError::provider(format!(
                "failed to serialize streaming messages body: {e}"
            ))
        })?;
        let url = format!("{}/anthropic/v1/messages", Self::endpoint_base(region));
        let response = self
            .send_signed(SignedRequest {
                method: reqwest::Method::POST,
                url: &url,
                region,
                body: body_bytes,
                extra_headers: ANTHROPIC_HEADERS,
            })
            .await?;

        let status = response.status();
        if !status.is_success() {
            let headers = response.headers().clone();
            let body = crate::http_error::read_error_body_or_warn(
                response,
                "bedrock-mantle",
                status.as_u16(),
            )
            .await;
            return Err(map_anthropic_error(status.as_u16(), &body, &headers));
        }

        let byte_stream = response.bytes_stream();
        let sse_stream = parse_sse_stream(byte_stream);
        let delta_stream = sse_stream
            .filter_map(|event_result| async move { convert_anthropic_sse_event(event_result) });
        Ok(Box::pin(delta_stream))
    }

    /// Adapter B — execute one non-streaming Responses request against
    /// `{base}/openai/v1/responses` in stateless mode (`store: false`).
    ///
    /// Used for `openai.gpt-5.*` (the only Mantle models that route exclusively through Responses
    /// today). Wire shape differs from Chat Completions: `input` array instead of `messages`,
    /// nested `reasoning.effort`, `text.format` for structured output, typed envelope on the
    /// response (`output[]` items with `type: message` / `type: reasoning`).
    async fn responses_generate(
        &self,
        model: &str,
        messages: &[Message],
        config: &LanguageModelConfig,
        region: &str,
    ) -> Result<LanguageModelResponse, LanguageModelError> {
        let request_body = build_responses_request(model, messages, config, false);
        let body_bytes = serde_json::to_vec(&request_body).map_err(|e| {
            LanguageModelError::provider(format!("failed to serialize responses body: {e}"))
        })?;
        let url = format!("{}/openai/v1/responses", Self::endpoint_base(region));
        let response = self
            .send_signed(SignedRequest {
                method: reqwest::Method::POST,
                url: &url,
                region,
                body: body_bytes,
                extra_headers: &[],
            })
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = crate::http_error::read_error_body_or_warn(
                response,
                "bedrock-mantle",
                status.as_u16(),
            )
            .await;
            return Err(map_mantle_error(status.as_u16(), &body));
        }

        let api_response: ResponsesResponse = response.json().await.map_err(|e| {
            LanguageModelError::provider(format!("failed to parse responses response: {e}"))
        })?;
        parse_responses_response(api_response)
    }

    /// Adapter B streaming — typed SSE events (`response.output_text.delta`, `response.completed`,
    /// `response.failed`) projected onto [`StreamDelta`].
    async fn responses_stream(
        &self,
        model: &str,
        messages: &[Message],
        config: &LanguageModelConfig,
        region: &str,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamDelta, LanguageModelError>> + Send>>,
        LanguageModelError,
    > {
        let request_body = build_responses_request(model, messages, config, true);
        let body_bytes = serde_json::to_vec(&request_body).map_err(|e| {
            LanguageModelError::provider(format!(
                "failed to serialize streaming responses body: {e}"
            ))
        })?;
        let url = format!("{}/openai/v1/responses", Self::endpoint_base(region));
        let response = self
            .send_signed(SignedRequest {
                method: reqwest::Method::POST,
                url: &url,
                region,
                body: body_bytes,
                extra_headers: &[],
            })
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = crate::http_error::read_error_body_or_warn(
                response,
                "bedrock-mantle",
                status.as_u16(),
            )
            .await;
            return Err(map_mantle_error(status.as_u16(), &body));
        }

        let byte_stream = response.bytes_stream();
        let sse_stream = parse_sse_stream(byte_stream);
        let delta_stream = sse_stream
            .filter_map(|event_result| async move { convert_responses_sse_event(event_result) });
        Ok(Box::pin(delta_stream))
    }

    /// Adapter A streaming — execute one streaming Chat Completions request, returning the SSE
    /// delta stream projected onto [`StreamDelta`].
    async fn chat_completions_stream(
        &self,
        model: &str,
        messages: &[Message],
        config: &LanguageModelConfig,
        region: &str,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamDelta, LanguageModelError>> + Send>>,
        LanguageModelError,
    > {
        let mut request_body = build_openai_request(
            model,
            messages,
            config,
            mantle_uses_completion_tokens(model),
        );
        request_body.stream = Some(true);
        // Mantle's Chat Completions surface follows OpenAI's stream-options contract — without
        // `include_usage = true` the final pre-`[DONE]` chunk would arrive with `usage: null`.
        request_body.stream_options = Some(OpenAiStreamOptions {
            include_usage: true,
        });
        let body_bytes = serde_json::to_vec(&request_body).map_err(|e| {
            LanguageModelError::provider(format!(
                "failed to serialize streaming chat-completions body: {e}"
            ))
        })?;
        let url = format!("{}/v1/chat/completions", Self::endpoint_base(region));

        let response = self
            .send_signed(SignedRequest {
                method: reqwest::Method::POST,
                url: &url,
                region,
                body: body_bytes,
                extra_headers: &[],
            })
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = crate::http_error::read_error_body_or_warn(
                response,
                "bedrock-mantle",
                status.as_u16(),
            )
            .await;
            return Err(map_mantle_error(status.as_u16(), &body));
        }

        let byte_stream = response.bytes_stream();
        let sse_stream = parse_sse_stream(byte_stream);
        let delta_stream = sse_stream
            .filter_map(|event_result| async move { convert_openai_sse_event(event_result) });
        Ok(Box::pin(delta_stream))
    }

    /// Send one HTTP request, applying SigV4 (or Bearer API key) to it first.
    ///
    /// SigV4 path: resolve credentials, build [`SignableRequest`] with the same body/headers we
    /// intend to send, run `aws-sigv4`'s signer, transcribe the signing headers it produces onto
    /// a fresh `reqwest::RequestBuilder`, and execute. The intermediate `http::Request` is a
    /// header carrier only — the signer applies `host` / `x-amz-date` / `x-amz-security-token` /
    /// `authorization` to it, then we copy them across.
    ///
    /// `req.extra_headers` adds headers to the request beyond `content-type: application/json`.
    /// They are included in the SigV4 signable header set (so the signature matches what's on
    /// the wire) and applied to the final request after signing. Used for surfaces like
    /// Anthropic Messages on Mantle that require `anthropic-version` for version pinning.
    async fn send_signed(
        &self,
        req: SignedRequest<'_>,
    ) -> Result<reqwest::Response, LanguageModelError> {
        let body_bytes = bytes::Bytes::from(req.body);

        let builder = match &self.auth {
            BedrockMantleAuth::ApiKey(key) => {
                let mut b = self
                    .client
                    .request(req.method, req.url)
                    .header("authorization", format!("Bearer {key}"))
                    .header("content-type", "application/json");
                for (k, v) in req.extra_headers {
                    b = b.header(*k, *v);
                }
                b.body(body_bytes)
            }
            BedrockMantleAuth::Sigv4 {
                credentials_provider,
            } => {
                let creds = credentials_provider
                    .provide_credentials()
                    .await
                    .map_err(|e| {
                        LanguageModelError::authentication(format!("aws credentials: {e}"))
                    })?;
                let identity: Identity = creds.into();
                let signing_params = v4::SigningParams::builder()
                    .identity(&identity)
                    .region(req.region)
                    .name(SIGV4_SERVICE_NAME)
                    .time(std::time::SystemTime::now())
                    .settings(SigningSettings::default())
                    .build()
                    .map_err(|e| LanguageModelError::provider(format!("sigv4 params: {e}")))?
                    .into();

                // Sign content-type + every extra header. AWS computes the signature against this
                // exact set, so the final request must include the same values (verified below by
                // adding them to the stub via `apply_to_request_http1x` + reqwest builder).
                let mut signable_headers: Vec<(&str, &str)> =
                    Vec::with_capacity(1 + req.extra_headers.len());
                signable_headers.push(("content-type", "application/json"));
                signable_headers.extend(req.extra_headers.iter().copied());
                let signable = SignableRequest::new(
                    req.method.as_str(),
                    req.url,
                    signable_headers.iter().copied(),
                    SignableBody::Bytes(&body_bytes),
                )
                .map_err(|e| LanguageModelError::provider(format!("signable request: {e}")))?;

                let signing_output = sign(signable, &signing_params)
                    .map_err(|e| LanguageModelError::provider(format!("sign: {e}")))?;
                let (instructions, _signature) = signing_output.into_parts();

                // Stub `http::Request` whose only role is to receive the signing instructions
                // (host / x-amz-date / authorization / x-amz-security-token) plus the signed
                // request headers — so we can read everything back out and apply it to the
                // reqwest builder uniformly.
                let mut stub_builder = http::Request::builder()
                    .method(req.method.as_str())
                    .uri(req.url)
                    .header("content-type", "application/json");
                for (k, v) in req.extra_headers {
                    stub_builder = stub_builder.header(*k, *v);
                }
                let mut stub: http::Request<()> = stub_builder.body(()).map_err(|e| {
                    LanguageModelError::provider(format!("build stub http req: {e}"))
                })?;
                instructions.apply_to_request_http1x(&mut stub);

                let mut signed = self.client.request(req.method, req.url).body(body_bytes);
                for (k, v) in stub.headers() {
                    signed = signed.header(k.as_str(), v.as_bytes());
                }
                signed
            }
        };

        builder
            .send()
            .await
            .map_err(|e| LanguageModelError::provider(e.to_string()))
    }

    /// Fetch one region's `GET /v1/models` catalog.
    async fn fetch_models_in_region(
        &self,
        region: &str,
    ) -> Result<Vec<MantleModelEntry>, LanguageModelError> {
        let url = format!("{}/v1/models", Self::endpoint_base(region));
        let response = self
            .send_signed(SignedRequest {
                method: reqwest::Method::GET,
                url: &url,
                region,
                body: Vec::new(),
                extra_headers: &[],
            })
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = crate::http_error::read_error_body_or_warn(
                response,
                "bedrock-mantle",
                status.as_u16(),
            )
            .await;
            return Err(map_mantle_error(status.as_u16(), &body));
        }

        let parsed: MantleListModelsResponse = response.json().await.map_err(|e| {
            LanguageModelError::provider(format!("failed to parse mantle models: {e}"))
        })?;
        Ok(parsed.data)
    }
}

/// Whether the given Mantle Chat Completions model uses `max_completion_tokens` rather than the
/// legacy `max_tokens` field. Driven by [`MODEL_CAPABILITIES`]: any model with a reasoning entry
/// is on the new field. Unknown models fall back to `max_tokens` for backwards-compat — a clear
/// API error on an unrecognised model beats a silent override.
fn mantle_uses_completion_tokens(model: &str) -> bool {
    MODEL_CAPABILITIES
        .get(model)
        .is_some_and(|c| c.reasoning.is_some())
}

/// Map an HTTP status + error body from a Mantle response to a [`LanguageModelError`].
fn map_mantle_error(status: u16, body: &str) -> LanguageModelError {
    match status {
        401 | 403 => LanguageModelError::authentication(body.to_owned()),
        429 => LanguageModelError::rate_limited(body.to_owned()),
        _ => LanguageModelError::provider(format!("HTTP {status}: {body}")),
    }
}

#[async_trait]
impl LanguageModelProvider for BedrockMantleProvider {
    fn name(&self) -> &'static str {
        "bedrock-mantle"
    }

    fn capabilities(&self, model: &ModelId) -> Option<ModelCapabilities> {
        let caps = MODEL_CAPABILITIES.get(model.as_str())?;
        Some(ModelCapabilities {
            model_id: model.as_str().to_owned(),
            media_support: caps.media_support.clone(),
            reasoning: caps.reasoning.clone(),
            // Bedrock-runtime / Converse-only — Mantle doesn't expose latency tiers.
            latency_optimized_supported: false,
            // Mantle doesn't expose cache control on either Chat Completions or Messages
            // — per AWS Mythos docs, prompt caching against Claude on Mantle is unavailable
            // (use bedrock-runtime for that). For the open-weight Chat Completions catalog
            // Mantle has no documented cache TTL knob.
            extended_cache_ttl_supported: false,
        })
    }

    #[tracing::instrument(
        skip(self),
        fields(
            provider = "bedrock-mantle",
            regions = tracing::field::Empty,
            model_count = tracing::field::Empty,
        ),
        err(Display),
    )]
    async fn list_models(&self) -> Result<Vec<ChatModelInfo>, LanguageModelError> {
        let result = self
            .list_models_cache
            .try_get_with((), async {
                // Fan out across every distinct configured region. Mantle's catalog is
                // asymmetric — gpt-5.* lives only on us-east-2, claude variants only on us-east-1,
                // open-weight catalog everywhere — so the union of per-region catalogs is the
                // real surface this provider serves. `try_join_all` short-circuits on the first
                // region error: a partial catalog would lie to the caller about which models are
                // actually callable, so any region failure becomes the whole call's failure.
                let regions = self.unique_regions();
                tracing::Span::current().record("regions", regions.join(","));
                let per_region = futures::future::try_join_all(
                    regions
                        .iter()
                        .map(|region| self.fetch_models_in_region(region)),
                )
                .await?;

                // Merge: dedup by model id (entries appearing in multiple regions are identical
                // — Mantle returns the same `id` everywhere it serves a given model). BTreeMap
                // keeps iteration order deterministic for downstream consumers that diff
                // catalogs.
                let mut by_id: std::collections::BTreeMap<String, MantleModelEntry> =
                    std::collections::BTreeMap::new();
                for batch in per_region {
                    for entry in batch {
                        by_id.entry(entry.id.clone()).or_insert(entry);
                    }
                }

                let merged: Vec<ChatModelInfo> = by_id
                    .into_values()
                    .map(|m| {
                        let caps = MODEL_CAPABILITIES.get(m.id.as_str()).cloned();
                        if caps.is_none() {
                            tracing::warn!(
                                provider = "bedrock-mantle",
                                model_id = %m.id,
                                "model returned by upstream but no local capability metadata; \
                                 ChatModelInfo will have minimal fields. Update MODEL_CAPABILITIES \
                                 when this model is ready for first-class support."
                            );
                        }
                        let mut formats = vec![ResponseFormatKind::Text];
                        if caps.as_ref().is_some_and(|c| c.supports_json_schema) {
                            formats.push(ResponseFormatKind::JsonObject);
                            formats.push(ResponseFormatKind::JsonSchema);
                        }
                        ChatModelInfo {
                            id: ModelId::new(m.id),
                            display_name: None,
                            context_window: caps.as_ref().map(|c| c.context_window),
                            supports_streaming: caps.as_ref().is_some_and(|c| c.supports_streaming),
                            supported_response_formats: formats,
                            media_support: caps
                                .as_ref()
                                .map(|c| c.media_support.clone())
                                .unwrap_or_default(),
                            reasoning: caps.as_ref().and_then(|c| c.reasoning.clone()),
                        }
                    })
                    .collect();
                Ok::<_, LanguageModelError>(merged)
            })
            .await
            .map_err(|arc_err| (*arc_err).clone());
        if let Ok(ref v) = result {
            tracing::Span::current().record("model_count", v.len());
        }
        result
    }

    #[tracing::instrument(
        skip(self, request),
        fields(
            provider = "bedrock-mantle",
            model = %request.model,
            messages = request.messages.len(),
            surface = tracing::field::Empty,
            prompt_tokens = tracing::field::Empty,
            completion_tokens = tracing::field::Empty,
            total_tokens = tracing::field::Empty,
            cache_creation_input_tokens = tracing::field::Empty,
            cache_read_input_tokens = tracing::field::Empty,
        ),
        err(Display),
    )]
    async fn generate(
        &self,
        request: GenerateRequest<'_>,
    ) -> Result<LanguageModelResponse, LanguageModelError> {
        self.validate_request(&request)?;
        let model = request.model.as_str();
        let route = Route::from_model_id(model);
        let region = self.region_for(route).to_owned();
        let span = tracing::Span::current();
        span.record(
            "surface",
            tracing::field::display(format_args!("{route:?}")),
        );

        let response = match route {
            Route::ChatCompletions => {
                with_retry(&self.config.retry_config, || {
                    self.chat_completions_generate(model, request.messages, request.config, &region)
                })
                .await?
            }
            Route::Responses => {
                with_retry(&self.config.retry_config, || {
                    self.responses_generate(model, request.messages, request.config, &region)
                })
                .await?
            }
            Route::Messages => {
                with_retry(&self.config.retry_config, || {
                    self.messages_generate(model, request.messages, request.config, &region)
                })
                .await?
            }
        };

        if let Some(usage) = &response.usage {
            span.record("prompt_tokens", usage.input_tokens);
            span.record("completion_tokens", usage.output_tokens);
            span.record("total_tokens", usage.input_tokens + usage.output_tokens);
            span.record(
                "cache_creation_input_tokens",
                usage.cache_creation_input_tokens,
            );
            span.record("cache_read_input_tokens", usage.cache_read_input_tokens);
        }
        Ok(response)
    }

    #[tracing::instrument(
        skip(self, request),
        fields(
            provider = "bedrock-mantle",
            model = %request.model,
            messages = request.messages.len(),
            surface = tracing::field::Empty,
            first_token_ms = tracing::field::Empty,
            prompt_tokens = tracing::field::Empty,
            completion_tokens = tracing::field::Empty,
            total_tokens = tracing::field::Empty,
            cache_creation_input_tokens = tracing::field::Empty,
            cache_read_input_tokens = tracing::field::Empty,
        ),
        err(Display),
    )]
    async fn generate_stream(
        &self,
        request: GenerateRequest<'_>,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamDelta, LanguageModelError>> + Send>>,
        LanguageModelError,
    > {
        let started_at = std::time::Instant::now();
        self.validate_request(&request)?;
        let model = request.model.as_str();
        let route = Route::from_model_id(model);
        let region = self.region_for(route).to_owned();
        tracing::Span::current().record(
            "surface",
            tracing::field::display(format_args!("{route:?}")),
        );

        let inner = match route {
            Route::ChatCompletions => {
                self.chat_completions_stream(model, request.messages, request.config, &region)
                    .await?
            }
            Route::Responses => {
                self.responses_stream(model, request.messages, request.config, &region)
                    .await?
            }
            Route::Messages => {
                self.messages_stream(model, request.messages, request.config, &region)
                    .await?
            }
        };
        let wrapped =
            crate::streaming_timing::instrument_stream(tracing::Span::current(), started_at, inner);
        Ok(Box::pin(wrapped))
    }
}

// --- /v1/models serde ---

#[derive(Deserialize)]
struct MantleListModelsResponse {
    data: Vec<MantleModelEntry>,
}

#[derive(Deserialize)]
struct MantleModelEntry {
    id: String,
    // `owned_by` is `"system"` on every model in the live catalog as of 2026-06-04 — ignore.
}
