// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use aws_sdk_bedrockruntime::{
    error::SdkError,
    operation::{converse::ConverseError, converse_stream::ConverseStreamError},
    primitives::Blob,
    types::{
        CachePointBlock, CachePointType, CacheTtl as BedrockCacheTtl, ContentBlock,
        ConversationRole, DocumentBlock, DocumentFormat, DocumentSource, ImageBlock, ImageFormat,
        ImageSource, InferenceConfiguration, JsonSchemaDefinition, Message as BedrockMessage,
        OutputConfig, OutputFormat, OutputFormatStructure, OutputFormatType,
        PerformanceConfigLatency, PerformanceConfiguration, S3Location, SystemContentBlock,
        VideoBlock, VideoFormat, VideoSource,
    },
};
use aws_smithy_types::Document as SmithyDocument;
use rustc_hash::FxHashMap;

use crate::{
    config::{
        CacheTtl, LanguageModelConfig, LatencyMode, ReasoningConfig, ReasoningEffort,
        ResponseFormat,
    },
    error::LanguageModelError,
    media::{MediaSource, MediaType},
    message::{ContentPart, Role},
};

/// Build a [`CachePointBlock`] for the requested TTL.
///
/// Specifying *any* explicit `ttl` opts into Bedrock's "extended TTL prompt
/// caching" feature, which only Anthropic models accept — sending it to
/// Nova fails with "Extended TTL prompt caching is only supported for
/// Anthropic models". Bedrock's implicit default is already 5 minutes, so
/// we set the field only for the 1-hour case. The caller validates that a
/// `OneHour` request targets an extended-TTL-capable model before it reaches
/// here, and the cachePoint fail-safe
/// covers any residual rejection.
pub(super) fn cache_point_block(ttl: CacheTtl) -> Result<CachePointBlock, LanguageModelError> {
    let mut builder = CachePointBlock::builder().r#type(CachePointType::Default);
    if matches!(ttl, CacheTtl::OneHour) {
        builder = builder.ttl(BedrockCacheTtl::OneHour);
    }
    builder
        .build()
        .map_err(|e| LanguageModelError::provider(format!("build cache point: {e}")))
}

/// Whether a Bedrock foundation-model family accepts Converse `cachePoint`
/// blocks. A `cachePoint` on a non-supporting model is rejected with
/// `ValidationException` or `AccessDeniedException` ("your request did not
/// allow prompt caching"), so we gate emission rather than send it
/// speculatively. Conservative by design: Anthropic Claude and Amazon Nova
/// are the only families AWS documents as supporting prompt caching today
/// (see <https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html>);
/// everything else (Llama, DeepSeek, Mistral, gpt-oss, Kimi, GLM, …) would
/// pay a wasted-roundtrip penalty per call via the fail-safe without any
/// cache benefit.
///
/// Takes the bare foundation id (region prefix already stripped, or an ARN
/// already resolved to its foundation) — see
/// [`BedrockProvider::supports_prompt_caching`](super::BedrockProvider::supports_prompt_caching)
/// for the id-normalizing wrapper.
pub(super) fn caching_family_supported(foundation_id: &str) -> bool {
    // The original Claude 3 (2024) models predate prompt caching and reject
    // a cachePoint ("your request did not allow prompt caching"); Claude
    // 3.5/3.7/4.x support it. The `-2024` date disambiguates the originals
    // from `claude-3-5-*` (Claude 3.5).
    let claude3_original = foundation_id.starts_with("anthropic.claude-3-haiku-2024")
        || foundation_id.starts_with("anthropic.claude-3-sonnet-2024")
        || foundation_id.starts_with("anthropic.claude-3-opus-2024");
    if claude3_original {
        return false;
    }
    foundation_id.starts_with("anthropic.claude") || foundation_id.starts_with("amazon.nova")
}

/// Caching decision for `model_id` given the resolved-ARN map. Split from
/// [`BedrockProvider::supports_prompt_caching`](super::BedrockProvider::supports_prompt_caching)
/// (which only holds the lock) so the full id-normalization — application-profile
/// ARN resolution and region-prefix stripping — is unit-testable without a
/// provider instance.
pub(super) fn supports_prompt_caching_with(
    resolved: &FxHashMap<String, String>,
    model_id: &str,
) -> bool {
    if let Some(foundation) = resolved.get(model_id) {
        return caching_family_supported(foundation);
    }
    caching_family_supported(strip_region_prefix(model_id))
}

/// Build the Bedrock `performanceConfig` for a request. Returns `Some` only
/// for [`LatencyMode::Optimized`]; `Standard` (the default) sends no
/// performance config so Bedrock applies its standard latency tier. The
/// caller has already gated `Optimized` against the model's
/// `latency_optimized_supported` capability, so any id reaching here with
/// `Optimized` is known to accept it.
pub(super) fn build_performance_config(
    config: &LanguageModelConfig,
) -> Option<PerformanceConfiguration> {
    match config.latency {
        LatencyMode::Standard => None,
        LatencyMode::Optimized => Some(
            PerformanceConfiguration::builder()
                .latency(PerformanceConfigLatency::Optimized)
                .build(),
        ),
    }
}

pub(super) fn build_bedrock_messages(
    messages: &[crate::message::Message],
    cache_enabled: bool,
    cache_ttl: CacheTtl,
) -> Result<(Vec<SystemContentBlock>, Vec<BedrockMessage>), LanguageModelError> {
    let mut system: Vec<SystemContentBlock> = Vec::new();
    let mut out: Vec<BedrockMessage> = Vec::new();

    for msg in messages {
        match msg.role {
            Role::System => {
                // Bedrock's system field is text-only — non-text parts in
                // a system message are silently dropped. A CacheBreakpoint
                // marks the end of the cacheable prefix: emit a CachePoint
                // block when the model supports it, else drop it.
                for part in &msg.content {
                    match part {
                        ContentPart::Text { text } => {
                            system.push(SystemContentBlock::Text(text.clone()));
                        }
                        ContentPart::CacheBreakpoint if cache_enabled => {
                            system.push(SystemContentBlock::CachePoint(cache_point_block(
                                cache_ttl,
                            )?));
                        }
                        _ => {}
                    }
                }
            }
            Role::User | Role::Assistant => {
                let role = match msg.role {
                    Role::User => ConversationRole::User,
                    Role::Assistant => ConversationRole::Assistant,
                    Role::System => unreachable!("outer match excludes System"),
                };
                let mut blocks: Vec<ContentBlock> = Vec::new();
                for part in &msg.content {
                    if part.is_cache_breakpoint() {
                        if cache_enabled {
                            blocks.push(ContentBlock::CachePoint(cache_point_block(cache_ttl)?));
                        }
                        continue;
                    }
                    if let Some(block) = translate_part_for_bedrock(part) {
                        blocks.push(block);
                    }
                }
                if blocks.is_empty() {
                    return Err(LanguageModelError::provider(
                        "bedrock requires at least one content block per non-system message",
                    ));
                }
                let bm = BedrockMessage::builder()
                    .role(role)
                    .set_content(Some(blocks))
                    .build()
                    .map_err(|e| LanguageModelError::provider(format!("build message: {e}")))?;
                out.push(bm);
            }
        }
    }

    Ok((system, out))
}

/// Translate one [`ContentPart`] into the Bedrock `ContentBlock` shape.
///
/// `validate_request` runs before this fn and rejects unsupported
/// (modality, source) pairs at the request boundary. Anything that
/// reaches here and can't be mapped is a `tracing::warn!` + drop —
/// signals a capability-table bug.
pub(super) fn translate_part_for_bedrock(part: &ContentPart) -> Option<ContentBlock> {
    match part {
        ContentPart::Text { text } => Some(ContentBlock::Text(text.clone())),
        ContentPart::Image { source } => {
            let mime = match source {
                MediaSource::InlineBytes { mime, .. } => Some(mime),
                _ => None,
            };
            let format = mime
                .and_then(bedrock_image_format)
                .unwrap_or(ImageFormat::Png);
            let image_source = bedrock_image_source(source)?;
            ImageBlock::builder()
                .format(format)
                .source(image_source)
                .build()
                .ok()
                .map(ContentBlock::Image)
        }
        ContentPart::Document { source, name } => {
            let (format, mime) = match source {
                MediaSource::InlineBytes { mime, .. } => (
                    bedrock_document_format(mime).unwrap_or(DocumentFormat::Pdf),
                    Some(mime),
                ),
                // For non-inline sources we can't read the MIME from the
                // bytes — default to PDF (the most common case). When we
                // grow per-format S3/file plumbing this should accept an
                // explicit format hint from the caller.
                _ => (DocumentFormat::Pdf, None),
            };
            let _ = mime; // Reserved for richer per-source format inference.
            let doc_source = bedrock_document_source(source)?;
            let mut builder = DocumentBlock::builder().format(format).source(doc_source);
            if let Some(n) = name {
                builder = builder.name(n.clone());
            } else {
                // Bedrock requires a non-empty `name` on every document
                // block. Use a stable placeholder when callers don't
                // provide one — matches the `<unnamed>` pattern other
                // providers reject filenames for prompt-injection
                // reasons.
                builder = builder.name("document");
            }
            builder.build().ok().map(ContentBlock::Document)
        }
        ContentPart::Video { source } => {
            let format = match source {
                MediaSource::InlineBytes { mime, .. } => {
                    bedrock_video_format(mime).unwrap_or(VideoFormat::Mp4)
                }
                _ => VideoFormat::Mp4,
            };
            let video_source = bedrock_video_source(source)?;
            VideoBlock::builder()
                .format(format)
                .source(video_source)
                .build()
                .ok()
                .map(ContentBlock::Video)
        }
        ContentPart::Audio { .. } => {
            // AudioBlock support is gated to the Voxtral model family;
            // the SDK exposes an `AudioBlock` variant on `ContentBlock`
            // that needs to be enumerated when we wire Voxtral. Until
            // then the capability table advertises Audio for Voxtral
            // only, and validate_request gates non-Voxtral calls.
            tracing::warn!(
                provider = "bedrock",
                "audio translation not yet implemented — Voxtral wiring is future work",
            );
            None
        }
        // Cache markers are handled by the caller before translation; a
        // marker reaching here produces no content block.
        ContentPart::CacheBreakpoint => None,
    }
}

fn bedrock_image_format(mime: &MediaType) -> Option<ImageFormat> {
    match mime.subtype() {
        "png" => Some(ImageFormat::Png),
        "jpeg" | "jpg" => Some(ImageFormat::Jpeg),
        "gif" => Some(ImageFormat::Gif),
        "webp" => Some(ImageFormat::Webp),
        _ => None,
    }
}

fn bedrock_image_source(source: &MediaSource) -> Option<ImageSource> {
    match source {
        MediaSource::InlineBytes { data, .. } => Some(ImageSource::Bytes(Blob::new(data.clone()))),
        MediaSource::S3 { uri, bucket_owner } => {
            let mut builder = S3Location::builder().uri(uri.as_str());
            if let Some(owner) = bucket_owner {
                builder = builder.bucket_owner(owner.as_str());
            }
            builder.build().ok().map(ImageSource::S3Location)
        }
        other => {
            tracing::warn!(
                provider = "bedrock",
                source_kind = ?other.kind(),
                "validate_request should have rejected this image source kind",
            );
            None
        }
    }
}

fn bedrock_document_format(mime: &MediaType) -> Option<DocumentFormat> {
    match mime.subtype() {
        "pdf" => Some(DocumentFormat::Pdf),
        "csv" => Some(DocumentFormat::Csv),
        "doc" | "msword" => Some(DocumentFormat::Doc),
        "docx" | "vnd.openxmlformats-officedocument.wordprocessingml.document" => {
            Some(DocumentFormat::Docx)
        }
        "xls" | "vnd.ms-excel" => Some(DocumentFormat::Xls),
        "xlsx" | "vnd.openxmlformats-officedocument.spreadsheetml.sheet" => {
            Some(DocumentFormat::Xlsx)
        }
        "html" => Some(DocumentFormat::Html),
        "plain" | "txt" => Some(DocumentFormat::Txt),
        "markdown" | "md" => Some(DocumentFormat::Md),
        _ => None,
    }
}

fn bedrock_document_source(source: &MediaSource) -> Option<DocumentSource> {
    match source {
        MediaSource::InlineBytes { data, .. } => {
            Some(DocumentSource::Bytes(Blob::new(data.clone())))
        }
        MediaSource::S3 { uri, bucket_owner } => {
            let mut builder = S3Location::builder().uri(uri.as_str());
            if let Some(owner) = bucket_owner {
                builder = builder.bucket_owner(owner.as_str());
            }
            builder.build().ok().map(DocumentSource::S3Location)
        }
        other => {
            tracing::warn!(
                provider = "bedrock",
                source_kind = ?other.kind(),
                "validate_request should have rejected this document source kind",
            );
            None
        }
    }
}

fn bedrock_video_format(mime: &MediaType) -> Option<VideoFormat> {
    match mime.subtype() {
        "mp4" => Some(VideoFormat::Mp4),
        "quicktime" | "mov" => Some(VideoFormat::Mov),
        "x-matroska" | "mkv" => Some(VideoFormat::Mkv),
        "webm" => Some(VideoFormat::Webm),
        "x-flv" | "flv" => Some(VideoFormat::Flv),
        "mpeg" => Some(VideoFormat::Mpeg),
        "mpg" => Some(VideoFormat::Mpg),
        "x-ms-wmv" | "wmv" => Some(VideoFormat::Wmv),
        "3gpp" | "three_gp" => Some(VideoFormat::ThreeGp),
        _ => None,
    }
}

fn bedrock_video_source(source: &MediaSource) -> Option<VideoSource> {
    match source {
        MediaSource::InlineBytes { data, .. } => Some(VideoSource::Bytes(Blob::new(data.clone()))),
        MediaSource::S3 { uri, bucket_owner } => {
            let mut builder = S3Location::builder().uri(uri.as_str());
            if let Some(owner) = bucket_owner {
                builder = builder.bucket_owner(owner.as_str());
            }
            builder.build().ok().map(VideoSource::S3Location)
        }
        other => {
            tracing::warn!(
                provider = "bedrock",
                source_kind = ?other.kind(),
                "validate_request should have rejected this video source kind",
            );
            None
        }
    }
}

pub(super) fn build_inference_config(
    config: &LanguageModelConfig,
) -> Option<InferenceConfiguration> {
    let thinking_active = config
        .reasoning
        .as_ref()
        .is_some_and(|r| !matches!(r, ReasoningConfig::Off));
    let temperature = if thinking_active {
        None
    } else {
        config.temperature
    };
    let top_p_in_range = config.top_p.filter(|p| (0.95..=1.0).contains(p));
    let top_p = if thinking_active {
        top_p_in_range
    } else {
        config.top_p
    };

    if config.max_tokens.is_none()
        && temperature.is_none()
        && top_p.is_none()
        && config.stop.is_empty()
    {
        return None;
    }
    let mut builder = InferenceConfiguration::builder();
    if let Some(max) = config.max_tokens {
        builder = builder.max_tokens(i32::try_from(max).unwrap_or(i32::MAX));
    }
    if let Some(t) = temperature {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "sampling knob; f64->f32 nudge has no semantic effect"
        )]
        {
            builder = builder.temperature(t as f32);
        }
    }
    if let Some(p) = top_p {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "sampling knob; f64->f32 nudge has no semantic effect"
        )]
        {
            builder = builder.top_p(p as f32);
        }
    }
    if !config.stop.is_empty() {
        builder = builder.set_stop_sequences(Some(config.stop.clone()));
    }
    Some(builder.build())
}

/// Build the `additionalModelRequestFields` document for a Bedrock request,
/// translating the provider-agnostic [`ReasoningConfig`] into the per-family
/// wire shape:
///
/// - **Anthropic Claude** — `{ thinking: {...}, output_config: {...} }` (adaptive effort, or an
///   enabled token budget); `Off`/absent → no fields.
/// - **OpenAI gpt-oss** — `{ reasoning_effort: "low"|"medium"|"high" }`. gpt-oss always reasons, so
///   there is no wire value to disable it: `Off` is rejected loudly. The caller's capability check
///   already keeps `manual` / unsupported efforts out, so only `adaptive` low/medium/high reach
///   here.
/// - **Every other family** — no reasoning fields. The caller rejects adaptive/manual against
///   their `reasoning: None` capability, so only `Off`/absent arrive, which carry no wire effect
///   anyway.
///
/// `model` is the id as invoked; the family is read after `strip_region_prefix`
/// so cross-region / global profile ids resolve correctly.
pub(super) fn bedrock_additional_request_fields(
    model: &str,
    reasoning: Option<&ReasoningConfig>,
) -> Result<Option<SmithyDocument>, LanguageModelError> {
    let Some(&reasoning) = reasoning else {
        return Ok(None);
    };
    let family = strip_region_prefix(model);
    if family.starts_with("openai.gpt-oss") {
        return gpt_oss_reasoning_fields(model, reasoning);
    }
    if family.starts_with("anthropic.claude") {
        return Ok(anthropic_thinking_document(reasoning));
    }
    Ok(None)
}

/// gpt-oss reasoning → `{ reasoning_effort: <low|medium|high> }`. gpt-oss has
/// no off switch, so `Off` errors with a pointer to the minimizing config; a
/// manual budget is meaningless for gpt-oss and also errors. Both are
/// defence-in-depth — the caller's `ReasoningCapability::validate` already
/// rejects them — so the messages name the model and the supported shape.
fn gpt_oss_reasoning_fields(
    model: &str,
    reasoning: ReasoningConfig,
) -> Result<Option<SmithyDocument>, LanguageModelError> {
    let effort = match reasoning {
        ReasoningConfig::Off => {
            return Err(LanguageModelError::provider(format!(
                "model `{model}` (gpt-oss) cannot disable reasoning; set \
                 reasoning: {{ mode: adaptive, effort: low }} to minimize it"
            )));
        }
        ReasoningConfig::Manual { .. } => {
            return Err(LanguageModelError::provider(format!(
                "model `{model}` (gpt-oss) has no manual reasoning budget; use \
                 reasoning: {{ mode: adaptive, effort: low|medium|high }}"
            )));
        }
        // gpt-oss advertises only low/medium/high, so the caller rejects the
        // rest; the off-list efforts are clamped defensively to a valid value.
        ReasoningConfig::Adaptive { effort } => match effort {
            ReasoningEffort::Low | ReasoningEffort::None => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High | ReasoningEffort::XHigh | ReasoningEffort::Max => "high",
        },
    };
    let mut root = HashMap::new();
    root.insert("reasoning_effort".to_owned(), SmithyDocument::from(effort));
    Ok(Some(SmithyDocument::from(root)))
}

/// Anthropic-on-Bedrock `thinking` / `output_config` document. `Off` → `None`
/// (omit the field, no extended thinking).
fn anthropic_thinking_document(reasoning: ReasoningConfig) -> Option<SmithyDocument> {
    let mut root = HashMap::new();
    match reasoning {
        ReasoningConfig::Off => return None,
        ReasoningConfig::Adaptive { effort } => {
            let mut thinking = HashMap::new();
            thinking.insert("type".to_owned(), SmithyDocument::from("adaptive"));
            root.insert("thinking".to_owned(), SmithyDocument::from(thinking));
            let mut output_config = HashMap::new();
            output_config.insert("effort".to_owned(), SmithyDocument::from(effort.as_str()));
            root.insert(
                "output_config".to_owned(),
                SmithyDocument::from(output_config),
            );
        }
        ReasoningConfig::Manual { budget_tokens } => {
            let mut thinking = HashMap::new();
            thinking.insert("type".to_owned(), SmithyDocument::from("enabled"));
            thinking.insert(
                "budget_tokens".to_owned(),
                SmithyDocument::from(u64::from(budget_tokens)),
            );
            root.insert("thinking".to_owned(), SmithyDocument::from(thinking));
        }
    }
    Some(SmithyDocument::from(root))
}

pub(super) fn build_output_config(
    format: &ResponseFormat,
) -> Result<Option<OutputConfig>, LanguageModelError> {
    // Bedrock has only `json_schema`; no schema-less JSON mode. Map our
    // `JsonObject` to a permissive `{"type":"object"}` so callers asking
    // for "any JSON" still get structured output. The schema field on the
    // SDK is a JSON-encoded string, not a Value.
    let definition = match format {
        ResponseFormat::Text => return Ok(None),
        ResponseFormat::JsonObject => JsonSchemaDefinition::builder()
            .schema(serde_json::json!({"type": "object"}).to_string())
            .description("any JSON object")
            .build()
            .map_err(|e| LanguageModelError::provider(format!("build json_schema: {e}")))?,
        ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        } => {
            let mut builder = JsonSchemaDefinition::builder()
                .schema(serde_json::to_string(schema).map_err(|e| {
                    LanguageModelError::provider(format!("serialize json_schema: {e}"))
                })?)
                .name(name);
            if *strict {
                builder = builder.description("strict-mode schema");
            }
            builder
                .build()
                .map_err(|e| LanguageModelError::provider(format!("build json_schema: {e}")))?
        }
    };
    let format = OutputFormat::builder()
        .r#type(OutputFormatType::JsonSchema)
        .structure(OutputFormatStructure::JsonSchema(definition))
        .build()
        .map_err(|e| LanguageModelError::provider(format!("build output_format: {e}")))?;
    Ok(Some(OutputConfig::builder().text_format(format).build()))
}

pub(super) fn strip_region_prefix(id: &str) -> &str {
    // Bedrock cross-region inference-profile prefixes. The geographic
    // set (`us.`, `eu.`, `apac.`, `jp.`, `au.`, `us-gov.`) routes within
    // a fixed region pool; `global.` routes to any commercial region
    // and is the only profile available in regions where the foundation
    // model isn't directly hosted (e.g. ca-central-1). Every callsite
    // here (`bedrock_supports_prompt_caching`, model-catalog lookup)
    // wants the bare foundation id, so any missing prefix silently
    // mis-routes — keep this list in sync with AWS' Geo profile list.
    for prefix in ["us.", "eu.", "apac.", "jp.", "au.", "us-gov.", "global."] {
        if let Some(rest) = id.strip_prefix(prefix) {
            return rest;
        }
    }
    id
}

/// Foundation-model id an application profile's wrapped-model ARN points at.
///
/// The ARN tail after the last `/` is either a bare foundation id
/// (`…:foundation-model/anthropic.claude-…`) or a cross-region profile id
/// (`…:inference-profile/us.anthropic.claude-…`); `strip_region_prefix`
/// normalizes the latter. Returns an owned `String` since the caller stores
/// it in the `resolved_arns` map.
pub(super) fn foundation_id_from_model_arn(model_arn: &str) -> String {
    let tail = model_arn.rsplit('/').next().unwrap_or(model_arn);
    strip_region_prefix(tail).to_owned()
}

/// Whether a Converse error came from Bedrock rejecting a `cachePoint`
/// block. Bedrock returns these rejections through at least two channels:
/// `ValidationException` for malformed-request-style errors, and
/// `AccessDeniedException` with messages like "your request did not allow
/// prompt caching" for unsupported model+caching combinations. Both
/// variants are matched on the message containing "cach" (covers "cache"
/// and "caching") so the trial-family fail-safe in `execute_converse` /
/// `execute_converse_stream` can recover transparently. Genuine
/// access-denied / validation errors unrelated to caching don't contain
/// the substring and still surface; a false positive only costs one
/// wasted no-cache retry that then fails with the real error.
pub(super) fn is_cachepoint_rejection(err: &SdkError<ConverseError>) -> bool {
    let SdkError::ServiceError(svc) = err else {
        return false;
    };
    let message = match svc.err() {
        ConverseError::ValidationException(e) => e.message(),
        ConverseError::AccessDeniedException(e) => e.message(),
        _ => return false,
    };
    message.is_some_and(|m| m.to_ascii_lowercase().contains("cach"))
}

/// Streaming counterpart of [`is_cachepoint_rejection`].
pub(super) fn is_cachepoint_stream_rejection(err: &SdkError<ConverseStreamError>) -> bool {
    let SdkError::ServiceError(svc) = err else {
        return false;
    };
    let message = match svc.err() {
        ConverseStreamError::ValidationException(e) => e.message(),
        ConverseStreamError::AccessDeniedException(e) => e.message(),
        _ => return false,
    };
    message.is_some_and(|m| m.to_ascii_lowercase().contains("cach"))
}
