// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! AWS Bedrock implementation of [`LanguageModelProvider`] via the
//! Converse / ConverseStream APIs.
//!
//! The provider holds two SDK clients — `aws_sdk_bedrockruntime::Client`
//! for chat/generate and `aws_sdk_bedrock::Client` for catalog
//! discovery. Authentication uses the standard AWS credential chain
//! (env vars, profile, IMDS / IAM role, SSO, web identity); operators
//! running on EC2 / ECS / EKS need no explicit credential config.
//!
//! Bedrock surfaces both raw foundation-model IDs (e.g.
//! `anthropic.claude-haiku-4-5-20251001-v1:0`) and inference-profile
//! IDs (e.g. `us.anthropic.claude-haiku-4-5-20251001-v1:0`); the
//! Converse `model_id` field accepts either form. `list_models`
//! returns both kinds in a single merged catalog.
//!
//! # Request visibility for debugging
//!
//! `execute_converse` / `execute_converse_stream` emit the outgoing request at
//! `TRACE` on the target `modelplease::bedrock::request`.
//! `messages_json` is the provider-agnostic content serialized to JSON
//! (parseable); `wire_system` / `wire_messages` are the exact AWS Converse shape
//! in `Debug` form (the AWS SDK types aren't `Serialize`), so the `cachePoint`
//! translation and system/message split are visible. Off by default; opt in:
//!
//! - Standalone binaries / examples (env filter reads `RUST_LOG`):
//!   `RUST_LOG=modelplease::bedrock::request=trace`
//! - Applications with a broader filter can pass a complete directive:
//!   `RUST_LOG="warn,modelplease=info,modelplease::bedrock::request=trace"`
//!
//! Prompts can contain sensitive user data, so
//! this is `TRACE`-gated and intended for `… 2>&1 | tee` inspection — never
//! enable it in a production log shipping.

pub(super) mod capabilities;
pub(super) mod request;
pub(super) mod response;
#[cfg(test)]
mod tests;

use std::{pin::Pin, time::Duration};

use async_trait::async_trait;
use aws_sdk_bedrock::types::{
    FoundationModelLifecycleStatus, InferenceProfileStatus, InferenceProfileType, ModelModality,
};
use capabilities::{
    ChatFeature, MODEL_CAPABILITIES, bedrock_media_support, bedrock_model_capabilities,
};
use futures::stream::Stream;
use moka::future::Cache;
use parking_lot::RwLock;
use request::{
    bedrock_additional_request_fields, build_bedrock_messages, build_inference_config,
    build_output_config, build_performance_config, foundation_id_from_model_arn,
    is_cachepoint_rejection, is_cachepoint_stream_rejection, strip_region_prefix,
    supports_prompt_caching_with,
};
use response::{
    convert_stream_event_to_delta, map_converse_error, map_converse_stream_error,
    parse_converse_output,
};
use rustc_hash::FxHashMap;

use crate::{
    capabilities::ModelCapabilities,
    config::{LanguageModelConfig, PromptCaching},
    error::LanguageModelError,
    identifiers::ModelId,
    message::Message,
    provider::{ChatModelInfo, GenerateRequest, LanguageModelProvider, ResponseFormatKind},
    response::{LanguageModelResponse, StreamDelta},
    retry::{RetryConfig, with_retry},
};

const PROVIDER_NAME: &str = "bedrock";

/// AWS Bedrock language model provider.
///
/// Wraps `aws_sdk_bedrockruntime::Client` for Converse/ConverseStream
/// and `aws_sdk_bedrock::Client` for `list_foundation_models` /
/// `list_inference_profiles`. Constructed once at boot via
/// [`BedrockProvider::new`]; one instance serves every Bedrock model.
pub struct BedrockProvider {
    runtime: aws_sdk_bedrockruntime::Client,
    control: aws_sdk_bedrock::Client,
    region: Option<String>,
    retry_config: RetryConfig,
    list_models_cache: Cache<(), Vec<ChatModelInfo>>,
    /// Application-profile ARN → resolved foundation-model id. Populated
    /// as a side effect of `fetch_and_merge_models`; backs the synchronous
    /// `capabilities()` and `supports_prompt_caching` ARN lookups, which
    /// can't await the async moka `list_models_cache`.
    resolved_arns: RwLock<FxHashMap<String, String>>,
}

/// Configuration for [`BedrockProvider`]. Pure value-shaped data —
/// AWS SDK client construction (credential chain probing, profile
/// resolution) happens in the application composition root, not here.
pub struct BedrockProviderConfig {
    /// Region carried separately for log fields; the SDK clients
    /// already have it baked in.
    pub region: Option<String>,
    pub retry_config: RetryConfig,
}

/// Injected AWS SDK clients for [`BedrockProvider`].
///
/// Constructed in the application composition root via
/// `aws_config::defaults(...).load()` so the async credential-chain
/// probe (IMDS / SSO files) happens once at boot and the provider
/// stays sync.
pub struct BedrockProviderDeps {
    pub runtime_client: aws_sdk_bedrockruntime::Client,
    pub control_client: aws_sdk_bedrock::Client,
}

impl BedrockProvider {
    /// Build the provider from pre-constructed AWS SDK clients.
    #[must_use]
    pub fn new(deps: BedrockProviderDeps, config: BedrockProviderConfig) -> Self {
        Self {
            runtime: deps.runtime_client,
            control: deps.control_client,
            region: config.region,
            retry_config: config.retry_config,
            list_models_cache: Cache::builder()
                .time_to_live(Duration::from_secs(3600))
                .max_capacity(1)
                .build(),
            resolved_arns: RwLock::new(FxHashMap::default()),
        }
    }

    #[must_use]
    pub fn region(&self) -> Option<&str> {
        self.region.as_deref()
    }

    /// Whether `cachePoint` blocks should be emitted for `model_id`.
    ///
    /// Resolves the family the id ultimately routes to:
    /// - an application-profile ARN in `resolved_arns` → its foundation family (so a profile
    ///   wrapping Claude/Nova gets caching);
    /// - any other id (bare foundation, cross-region prefix, or an unresolved ARN such as a
    ///   provisioned-model ARN) → the family of the region-prefix-stripped id.
    ///
    /// A provisioned-model ARN is never in `resolved_arns` and its raw
    /// string matches no caching family, so caching stays off — which is
    /// correct, since AWS forbids prompt caching on provisioned throughput.
    fn supports_prompt_caching(&self, model_id: &str) -> bool {
        let resolved = self.resolved_arns.read();
        supports_prompt_caching_with(&resolved, model_id)
    }

    #[tracing::instrument(skip_all, fields(model, msg_count = messages.len(), http_status = tracing::field::Empty), level = "debug", err(Display))]
    async fn execute_converse(
        &self,
        model: &str,
        messages: &[Message],
        config: &LanguageModelConfig,
    ) -> Result<LanguageModelResponse, LanguageModelError> {
        let cache_enabled = matches!(config.prompt_caching, PromptCaching::Auto)
            && self.supports_prompt_caching(model);
        let inference_config = build_inference_config(config);
        let performance_config = build_performance_config(config);
        let output_config = build_output_config(&config.response_format)?;
        let additional_fields =
            bedrock_additional_request_fields(model, config.reasoning.as_ref())?;

        let (system, bedrock_messages) =
            build_bedrock_messages(messages, cache_enabled, config.cache_ttl)?;
        // Operator-opt-in wire-shape visibility, gated to TRACE. `messages_json`
        // is the provider-agnostic content (Serialize-able, parseable); `wire_*`
        // is the exact AWS Converse shape (Debug) so cachePoint placement and the
        // system/message split are visible. AWS SDK types aren't Serialize, so the
        // wire shape can't be JSON without a hand-rolled mirror of #[non_exhaustive]
        // enums. See the module doc for the RUST_LOG / RUST_LOG invocation.
        tracing::trace!(
            target: "modelplease::bedrock::request",
            provider = PROVIDER_NAME,
            model,
            cache_enabled,
            messages_json = %serde_json::to_string(messages).unwrap_or_default(),
            system_blocks = system.len(),
            messages_count = bedrock_messages.len(),
            wire_system = ?system,
            wire_messages = ?bedrock_messages,
            inference_config = ?inference_config,
            performance_config = ?performance_config,
            "bedrock converse request",
        );
        let send_result = self
            .runtime
            .converse()
            .model_id(model)
            .set_system(if system.is_empty() {
                None
            } else {
                Some(system)
            })
            .set_messages(Some(bedrock_messages))
            .set_inference_config(inference_config.clone())
            .set_performance_config(performance_config.clone())
            .set_output_config(output_config.clone())
            .set_additional_model_request_fields(additional_fields.clone())
            .send()
            .await;

        let response = match send_result {
            Ok(resp) => resp,
            Err(err) if cache_enabled && is_cachepoint_rejection(&err) => {
                // Fail-safe: the model rejected the cachePoint (AWS support
                // matrix drift, or a too-broad gate). Caching is a cost
                // optimization and must never turn a working extraction
                // into a failure, so retry once with cache points stripped.
                tracing::warn!(
                    provider = PROVIDER_NAME,
                    model,
                    "bedrock rejected cachePoint; retrying without prompt cache",
                );
                let (system, bedrock_messages) =
                    build_bedrock_messages(messages, false, config.cache_ttl)?;
                self.runtime
                    .converse()
                    .model_id(model)
                    .set_system(if system.is_empty() {
                        None
                    } else {
                        Some(system)
                    })
                    .set_messages(Some(bedrock_messages))
                    .set_inference_config(inference_config)
                    .set_performance_config(performance_config)
                    .set_output_config(output_config)
                    .set_additional_model_request_fields(additional_fields)
                    .send()
                    .await
                    .map_err(map_converse_error)?
            }
            Err(err) => return Err(map_converse_error(err)),
        };

        // Snapshot stop_reason + usage by value before moving `output`
        // out of the response. Re-borrowing `response` after a partial
        // move of `output` would fail to compile.
        let stop_reason_snapshot = response.stop_reason().clone();
        let usage_snapshot = response.usage.clone();
        parse_converse_output(
            model,
            response.output,
            usage_snapshot.as_ref(),
            Some(&stop_reason_snapshot),
        )
    }

    #[tracing::instrument(skip_all, fields(model, msg_count = messages.len()), level = "debug", err(Display))]
    #[expect(
        clippy::type_complexity,
        reason = "Pin<Box<dyn Stream<...>>> matches the LanguageModelProvider \
                  trait signature; aliasing one site doesn't help readability"
    )]
    async fn execute_converse_stream(
        &self,
        model: &str,
        messages: &[Message],
        config: &LanguageModelConfig,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamDelta, LanguageModelError>> + Send>>,
        LanguageModelError,
    > {
        let cache_enabled = matches!(config.prompt_caching, PromptCaching::Auto)
            && self.supports_prompt_caching(model);
        let inference_config = build_inference_config(config);
        let performance_config = build_performance_config(config);
        let output_config = build_output_config(&config.response_format)?;
        let additional_fields =
            bedrock_additional_request_fields(model, config.reasoning.as_ref())?;

        let (system, bedrock_messages) =
            build_bedrock_messages(messages, cache_enabled, config.cache_ttl)?;
        // See execute_converse: operator-opt-in wire-shape visibility, gated to TRACE.
        tracing::trace!(
            target: "modelplease::bedrock::request",
            provider = PROVIDER_NAME,
            model,
            cache_enabled,
            messages_json = %serde_json::to_string(messages).unwrap_or_default(),
            system_blocks = system.len(),
            messages_count = bedrock_messages.len(),
            wire_system = ?system,
            wire_messages = ?bedrock_messages,
            inference_config = ?inference_config,
            performance_config = ?performance_config,
            "bedrock converse request",
        );
        let send_result = self
            .runtime
            .converse_stream()
            .model_id(model)
            .set_system(if system.is_empty() {
                None
            } else {
                Some(system)
            })
            .set_messages(Some(bedrock_messages))
            .set_inference_config(inference_config.clone())
            .set_performance_config(performance_config.clone())
            .set_output_config(output_config.clone())
            .set_additional_model_request_fields(additional_fields.clone())
            .send()
            .await;

        let response = match send_result {
            Ok(resp) => resp,
            Err(err) if cache_enabled && is_cachepoint_stream_rejection(&err) => {
                // Fail-safe (see execute_converse): never let a cachePoint
                // rejection break a working call — retry without caching.
                tracing::warn!(
                    provider = PROVIDER_NAME,
                    model,
                    "bedrock rejected cachePoint; retrying stream without prompt cache",
                );
                let (system, bedrock_messages) =
                    build_bedrock_messages(messages, false, config.cache_ttl)?;
                self.runtime
                    .converse_stream()
                    .model_id(model)
                    .set_system(if system.is_empty() {
                        None
                    } else {
                        Some(system)
                    })
                    .set_messages(Some(bedrock_messages))
                    .set_inference_config(inference_config)
                    .set_performance_config(performance_config)
                    .set_output_config(output_config)
                    .set_additional_model_request_fields(additional_fields)
                    .send()
                    .await
                    .map_err(map_converse_stream_error)?
            }
            Err(err) => return Err(map_converse_stream_error(err)),
        };

        let stream = futures::stream::unfold(
            (response.stream, false),
            |(mut events, finished)| async move {
                if finished {
                    return None;
                }
                loop {
                    match events.recv().await {
                        Ok(Some(event)) => {
                            if let Some(delta) = convert_stream_event_to_delta(&event) {
                                let is_final = delta.is_final;
                                return Some((Ok(delta), (events, is_final)));
                            }
                        }
                        Ok(None) => return None,
                        Err(e) => {
                            return Some((
                                Err(LanguageModelError::provider(format!("bedrock stream: {e}"))),
                                (events, true),
                            ));
                        }
                    }
                }
            },
        );

        Ok(Box::pin(stream))
    }

    async fn fetch_and_merge_models(&self) -> Result<Vec<ChatModelInfo>, LanguageModelError> {
        let foundation = self
            .control
            .list_foundation_models()
            .send()
            .await
            .map_err(|e| {
                LanguageModelError::provider(format!("bedrock list_foundation_models: {e}"))
            })?;

        let mut out: Vec<ChatModelInfo> = Vec::new();
        let mut warned: std::collections::HashSet<String> = std::collections::HashSet::new();

        for summary in foundation.model_summaries() {
            // Skip image-gen / embedding / rerank — chat surface only.
            let inputs = summary.input_modalities();
            let outputs = summary.output_modalities();
            if !inputs.contains(&ModelModality::Text) || !outputs.contains(&ModelModality::Text) {
                continue;
            }
            // Drop LEGACY models — provider has marked them for retirement
            // and they often refuse new traffic ("not actively used in 30 days").
            if matches!(
                summary
                    .model_lifecycle()
                    .map(aws_sdk_bedrock::types::FoundationModelLifecycle::status),
                Some(FoundationModelLifecycleStatus::Legacy)
            ) {
                continue;
            }
            let id = summary.model_id();
            let caps = MODEL_CAPABILITIES.get(id).cloned();
            if caps.is_none() && warned.insert(id.to_owned()) {
                tracing::warn!(
                    provider = PROVIDER_NAME,
                    model_id = %id,
                    "model returned by upstream but no local capability metadata; \
                     ChatModelInfo will have minimal fields. \
                     Update the local capability table when ready."
                );
            }
            let mut formats = vec![ResponseFormatKind::Text];
            if caps
                .as_ref()
                .is_some_and(|c| c.features.contains(ChatFeature::JsonSchema))
            {
                formats.push(ResponseFormatKind::JsonObject);
                formats.push(ResponseFormatKind::JsonSchema);
            }
            out.push(ChatModelInfo {
                id: ModelId::new(id.to_owned()),
                display_name: summary.model_name().map(str::to_owned),
                context_window: caps.as_ref().map(|c| c.context_window),
                supports_streaming: summary.response_streaming_supported().unwrap_or_else(|| {
                    caps.as_ref()
                        .is_some_and(|c| c.features.contains(ChatFeature::Streaming))
                }),
                supported_response_formats: formats,
                media_support: bedrock_media_support(id),
                reasoning: caps.as_ref().and_then(|c| c.reasoning.clone()),
            });
        }

        // Two profile passes: SYSTEM_DEFINED (cross-region / global CRIS
        // profiles, keyed by their region-prefixed id) and APPLICATION
        // (customer-created profiles, keyed by ARN, resolved to a foundation
        // id for the sync capability/caching paths). Older code issued one
        // unfiltered `list_inference_profiles`; whether the default returns
        // APPLICATION isn't documented, so we request each type explicitly.
        let mut resolved: FxHashMap<String, String> = FxHashMap::default();
        self.fetch_inference_profiles(
            InferenceProfileType::SystemDefined,
            &mut out,
            &mut warned,
            &mut resolved,
        )
        .await?;
        self.fetch_inference_profiles(
            InferenceProfileType::Application,
            &mut out,
            &mut warned,
            &mut resolved,
        )
        .await?;
        *self.resolved_arns.write() = resolved;

        Ok(out)
    }

    /// Paginate `list_inference_profiles` for one profile type, pushing a
    /// `ChatModelInfo` per Active profile. For APPLICATION profiles — whose
    /// id is an opaque ARN carrying no foundation id — also resolve the
    /// wrapped foundation from the profile's `models` and record
    /// `ARN → foundation id` in `resolved`, which backs the synchronous
    /// `capabilities()` / `supports_prompt_caching` lookups.
    async fn fetch_inference_profiles(
        &self,
        profile_type: InferenceProfileType,
        out: &mut Vec<ChatModelInfo>,
        warned: &mut std::collections::HashSet<String>,
        resolved: &mut FxHashMap<String, String>,
    ) -> Result<(), LanguageModelError> {
        let is_application = matches!(profile_type, InferenceProfileType::Application);
        let mut next_token: Option<String> = None;
        loop {
            let mut req = self
                .control
                .list_inference_profiles()
                .type_equals(profile_type.clone());
            if let Some(token) = next_token.take() {
                req = req.next_token(token);
            }
            let page = req.send().await.map_err(|e| {
                LanguageModelError::provider(format!("bedrock list_inference_profiles: {e}"))
            })?;

            for profile in page.inference_profile_summaries() {
                if !matches!(profile.status(), InferenceProfileStatus::Active) {
                    continue;
                }
                // SYSTEM profiles carry the foundation id in their id
                // (`us.anthropic.…`); APPLICATION profiles are keyed by ARN
                // and the foundation comes from the first wrapped model.
                let (id, foundation): (&str, String) = if is_application {
                    let Some(model_arn) = profile.models().iter().find_map(|m| m.model_arn())
                    else {
                        // No resolvable wrapped model — can't classify
                        // capabilities or caching, so skip rather than emit a
                        // half-known entry.
                        continue;
                    };
                    (
                        profile.inference_profile_arn(),
                        foundation_id_from_model_arn(model_arn),
                    )
                } else {
                    let id = profile.inference_profile_id();
                    (id, strip_region_prefix(id).to_owned())
                };
                let caps = MODEL_CAPABILITIES.get(foundation.as_str()).cloned();
                if caps.is_none() && warned.insert(id.to_owned()) {
                    tracing::warn!(
                        provider = PROVIDER_NAME,
                        model_id = %id,
                        "inference profile returned by upstream but no local capability \
                         metadata for the underlying model; ChatModelInfo will have minimal \
                         fields. Update the local capability table when ready."
                    );
                }
                let mut formats = vec![ResponseFormatKind::Text];
                if caps
                    .as_ref()
                    .is_some_and(|c| c.features.contains(ChatFeature::JsonSchema))
                {
                    formats.push(ResponseFormatKind::JsonObject);
                    formats.push(ResponseFormatKind::JsonSchema);
                }
                out.push(ChatModelInfo {
                    id: ModelId::new(id.to_owned()),
                    display_name: Some(profile.inference_profile_name().to_owned()),
                    context_window: caps.as_ref().map(|c| c.context_window),
                    // Profiles route to streaming-capable foundations; assume true on miss.
                    supports_streaming: caps
                        .as_ref()
                        .is_none_or(|c| c.features.contains(ChatFeature::Streaming)),
                    supported_response_formats: formats,
                    // Profiles re-route to their foundation model, so media
                    // support follows the resolved foundation id.
                    media_support: bedrock_media_support(&foundation),
                    reasoning: caps.as_ref().and_then(|c| c.reasoning.clone()),
                });
                if is_application {
                    resolved.insert(id.to_owned(), foundation);
                }
            }

            match page.next_token() {
                Some(t) if !t.is_empty() => next_token = Some(t.to_owned()),
                _ => break,
            }
        }

        Ok(())
    }
}

#[async_trait]
impl LanguageModelProvider for BedrockProvider {
    fn name(&self) -> &'static str {
        PROVIDER_NAME
    }

    fn capabilities(&self, model: &ModelId) -> Option<ModelCapabilities> {
        // Two-stage lookup. Stage 1 covers raw foundation ids
        // (`anthropic.claude-...`) and cross-region inference-profile ids
        // (`us.anthropic.claude-...`, `global.…`), normalized via
        // `strip_region_prefix`. Stage 2 covers application-profile ARNs,
        // which carry no foundation id in the string: `fetch_and_merge_models`
        // resolves them to a foundation id and records it in `resolved_arns`,
        // and we read that here — the sync path the async `list_models`
        // moka cache can't serve.
        let raw = model.as_str();
        let lookup_id = strip_region_prefix(raw);
        if let Some(caps) = MODEL_CAPABILITIES.get(lookup_id) {
            return Some(bedrock_model_capabilities(raw, lookup_id, caps));
        }
        let foundation = self.resolved_arns.read().get(raw).cloned()?;
        let caps = MODEL_CAPABILITIES.get(foundation.as_str())?;
        Some(bedrock_model_capabilities(raw, &foundation, caps))
    }

    #[tracing::instrument(skip(self), fields(provider = PROVIDER_NAME, model_count = tracing::field::Empty), err(Display))]
    async fn list_models(&self) -> Result<Vec<ChatModelInfo>, LanguageModelError> {
        let result = self
            .list_models_cache
            .try_get_with((), self.fetch_and_merge_models())
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
            provider = "bedrock",
            model = %request.model,
            messages = request.messages.len(),
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
        let response = with_retry(&self.retry_config, || {
            self.execute_converse(model, request.messages, request.config)
        })
        .await?;
        let span = tracing::Span::current();
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
            provider = "bedrock",
            model = %request.model,
            messages = request.messages.len(),
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
        let inner = self
            .execute_converse_stream(request.model.as_str(), request.messages, request.config)
            .await?;
        let wrapped =
            crate::streaming_timing::instrument_stream(tracing::Span::current(), started_at, inner);
        Ok(Box::pin(wrapped))
    }
}
