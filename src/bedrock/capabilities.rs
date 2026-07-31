// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::{BTreeMap, HashMap},
    sync::LazyLock,
};

use enumset::{EnumSet, EnumSetType, enum_set};

use crate::{
    capabilities::{
        MediaKind, MediaSupport, ModelCapabilities, ReasoningCapability, ReasoningMode,
        ReasoningParamConflicts,
    },
    config::ReasoningEffort,
    media::SourceKind,
};

/// Boolean capability flags for a Bedrock chat model, collapsed into an
/// `EnumSet` so the table entry stays a tidy value object instead of a row
/// of parallel `bool` fields (and so new flags don't re-trip
/// `clippy::struct_excessive_bools`).
///
/// `Streaming` / `JsonSchema` describe the response surface; `LatencyOptimized`
/// and `ExtendedCacheTtl` are inference-tier eligibility, set per-model by the
/// maintained allow-lists at the tail of `MODEL_CAPABILITIES`.
#[derive(EnumSetType, Debug)]
pub(super) enum ChatFeature {
    /// `generate_stream` is expected to work.
    Streaming,
    /// Accepts Converse `OutputConfig` json_schema / json_object.
    JsonSchema,
    /// AWS latency-optimized ("accelerated") inference (allow-listed).
    LatencyOptimized,
    /// Extended (1-hour) prompt-cache TTL — Anthropic Claude 4.5+ only
    /// (allow-listed). Basic 5-minute caching is gated separately and more
    /// broadly by [`caching_family_supported`](super::request::caching_family_supported).
    ExtendedCacheTtl,
}

#[derive(Debug, Clone)]
pub(super) struct ChatModelCapabilities {
    pub(super) context_window: u32,
    pub(super) features: EnumSet<ChatFeature>,
    pub(super) reasoning: Option<ReasoningCapability>,
}

pub(super) const fn bedrock_reasoning_conflicts() -> ReasoningParamConflicts {
    ReasoningParamConflicts {
        temperature_forbidden: true,
        top_k_forbidden: true,
        top_p_allowed_range: Some(0.95..=1.0),
    }
}

pub(super) const fn bedrock_claude_opus_4_7_reasoning() -> ReasoningCapability {
    ReasoningCapability {
        supported_modes: enum_set!(ReasoningMode::Adaptive),
        supported_efforts: enum_set!(
            ReasoningEffort::Low
                | ReasoningEffort::Medium
                | ReasoningEffort::High
                | ReasoningEffort::XHigh
                | ReasoningEffort::Max
        ),
        manual_budget_range: None,
        conflicts: bedrock_reasoning_conflicts(),
        sampling_params_removed: true,
    }
}

pub(super) const fn bedrock_claude_sonnet_4_6_reasoning() -> ReasoningCapability {
    ReasoningCapability {
        supported_modes: enum_set!(ReasoningMode::Adaptive | ReasoningMode::Manual),
        supported_efforts: enum_set!(
            ReasoningEffort::Low
                | ReasoningEffort::Medium
                | ReasoningEffort::High
                | ReasoningEffort::Max
        ),
        manual_budget_range: Some(1024..=63_000),
        conflicts: bedrock_reasoning_conflicts(),
        sampling_params_removed: false,
    }
}

/// Sonnet 5 — adaptive-only. Manual `budget_tokens` is rejected with a 400
/// (Sonnet 4.6's transitional manual surface is gone), mirroring the Opus
/// 4.7/4.8 shape; Sonnet 5 is the first Sonnet-tier model to accept the
/// `xhigh` effort level. Kept distinct from `bedrock_claude_opus_4_7_reasoning`
/// so the two models' surfaces can diverge independently rather than aliasing.
pub(super) const fn bedrock_claude_sonnet_5_reasoning() -> ReasoningCapability {
    ReasoningCapability {
        supported_modes: enum_set!(ReasoningMode::Adaptive),
        supported_efforts: enum_set!(
            ReasoningEffort::Low
                | ReasoningEffort::Medium
                | ReasoningEffort::High
                | ReasoningEffort::XHigh
                | ReasoningEffort::Max
        ),
        manual_budget_range: None,
        conflicts: bedrock_reasoning_conflicts(),
        sampling_params_removed: true,
    }
}

pub(super) const fn bedrock_claude_opus_4_6_reasoning() -> ReasoningCapability {
    ReasoningCapability {
        supported_modes: enum_set!(ReasoningMode::Adaptive | ReasoningMode::Manual),
        supported_efforts: enum_set!(
            ReasoningEffort::Low
                | ReasoningEffort::Medium
                | ReasoningEffort::High
                | ReasoningEffort::Max
        ),
        manual_budget_range: Some(1024..=127_000),
        conflicts: bedrock_reasoning_conflicts(),
        sampling_params_removed: false,
    }
}

pub(super) const fn bedrock_claude_manual_only_reasoning(
    max_output_tokens: u32,
) -> ReasoningCapability {
    let upper = max_output_tokens.saturating_sub(1024);
    ReasoningCapability {
        supported_modes: enum_set!(ReasoningMode::Manual),
        supported_efforts: enum_set!(
            ReasoningEffort::Low | ReasoningEffort::Medium | ReasoningEffort::High
        ),
        manual_budget_range: Some(1024..=upper),
        conflicts: bedrock_reasoning_conflicts(),
        sampling_params_removed: false,
    }
}

/// Reasoning surface for OpenAI gpt-oss on Bedrock. gpt-oss always reasons;
/// the only control is OpenAI's `reasoning_effort` (low/medium/high), which
/// we model as adaptive-effort. There is no `none` (can't disable), no
/// manual token budget, and — unlike Anthropic thinking — no temperature /
/// top_p restrictions.
pub(super) const fn gpt_oss_reasoning() -> ReasoningCapability {
    ReasoningCapability {
        supported_modes: enum_set!(ReasoningMode::Adaptive),
        supported_efforts: enum_set!(
            ReasoningEffort::Low | ReasoningEffort::Medium | ReasoningEffort::High
        ),
        manual_budget_range: None,
        conflicts: ReasoningParamConflicts {
            temperature_forbidden: false,
            top_k_forbidden: false,
            top_p_allowed_range: None,
        },
        sampling_params_removed: false,
    }
}

/// Keys are foundation-model ids (no regional prefix); inference-profile ids
/// are normalized via [`strip_region_prefix`](super::request::strip_region_prefix) before lookup.
/// Non-Anthropic entries get added as warn-on-miss surfaces it in application logs.
pub(super) static MODEL_CAPABILITIES: LazyLock<HashMap<&'static str, ChatModelCapabilities>> =
    LazyLock::new(|| {
        let mut m = HashMap::new();

        // Anthropic Claude 4.x — 200k context, OutputConfig (json_schema)
        // verified working. Only Bedrock family that accepts json_schema
        // today. Reasoning surface mirrors native Anthropic per the
        // Bedrock model cards reviewed during planning.
        // Streaming + json_schema by default; LatencyOptimized stays unset
        // (AWS lists Claude 3.5 Haiku, which this table doesn't carry) and
        // ExtendedCacheTtl is set per-model by the allow-list below.
        let claude_base = |reasoning: Option<ReasoningCapability>| ChatModelCapabilities {
            context_window: 200_000,
            features: ChatFeature::Streaming | ChatFeature::JsonSchema,
            reasoning,
        };
        // Sonnet 4.5 / Opus 4.1 / Opus 4.5 — pre-adaptive Claude 4 models,
        // manual-only thinking surface (per native Anthropic docs).
        for id in [
            "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "anthropic.claude-opus-4-1-20250805-v1:0",
            "anthropic.claude-opus-4-5-20251101-v1:0",
        ] {
            m.insert(
                id,
                claude_base(Some(bedrock_claude_manual_only_reasoning(64_000))),
            );
        }
        // Haiku 4.5 — manual-only.
        m.insert(
            "anthropic.claude-haiku-4-5-20251001-v1:0",
            claude_base(Some(bedrock_claude_manual_only_reasoning(64_000))),
        );
        // Sonnet 4.6 — adaptive + manual.
        m.insert(
            "anthropic.claude-sonnet-4-6",
            claude_base(Some(bedrock_claude_sonnet_4_6_reasoning())),
        );
        // Sonnet 5 — adaptive only; manual hard-rejected with 400 (like Opus
        // 4.7/4.8). Bare foundation id with no date/version suffix, matching how
        // AWS lists it (`anthropic.claude-sonnet-5`); cross-region profiles
        // (`us.`/`eu.`/`global.`) are normalized by `strip_region_prefix`
        // before this lookup, so one entry covers every regional profile.
        m.insert(
            "anthropic.claude-sonnet-5",
            claude_base(Some(bedrock_claude_sonnet_5_reasoning())),
        );
        // Opus 4.6 — adaptive + manual (128K max output).
        m.insert(
            "anthropic.claude-opus-4-6-v1",
            claude_base(Some(bedrock_claude_opus_4_6_reasoning())),
        );
        // Opus 4.7 — adaptive only; manual hard-rejected with 400.
        m.insert(
            "anthropic.claude-opus-4-7",
            claude_base(Some(bedrock_claude_opus_4_7_reasoning())),
        );

        // Amazon Nova — OutputConfig rejected by every Nova variant tested
        // (returns ValidationException). Sizes per AWS docs.
        m.insert("amazon.nova-pro-v1:0", chat(300_000));
        m.insert("amazon.nova-pro-v1:0:24k", chat(24_000));
        m.insert("amazon.nova-pro-v1:0:300k", chat(300_000));
        m.insert("amazon.nova-lite-v1:0", chat(300_000));
        m.insert("amazon.nova-lite-v1:0:24k", chat(24_000));
        m.insert("amazon.nova-lite-v1:0:300k", chat(300_000));
        m.insert("amazon.nova-micro-v1:0", chat(128_000));
        m.insert("amazon.nova-micro-v1:0:24k", chat(24_000));
        m.insert("amazon.nova-micro-v1:0:128k", chat(128_000));
        m.insert("amazon.nova-2-lite-v1:0", chat(256_000));
        m.insert("amazon.nova-2-lite-v1:0:256k", chat(256_000));

        // Meta Llama 3.1 / 3.3 / 4.x — 128k context. Llama 3.0 was 8k but
        // is LEGACY in our filter, so it never reaches this table.
        for id in [
            "meta.llama3-1-8b-instruct-v1:0",
            "meta.llama3-1-70b-instruct-v1:0",
            "meta.llama3-1-405b-instruct-v1:0",
            "meta.llama3-3-70b-instruct-v1:0",
            "meta.llama4-scout-17b-instruct-v1:0",
            "meta.llama4-maverick-17b-instruct-v1:0",
        ] {
            m.insert(id, chat(128_000));
        }
        // Llama 3 originals — 8k context. Surface honestly though they're
        // legacy candidates Meta hasn't deprecated yet.
        for id in [
            "meta.llama3-8b-instruct-v1:0",
            "meta.llama3-70b-instruct-v1:0",
        ] {
            m.insert(id, chat(8_000));
        }

        // Mistral — older 7B/Mixtral/large-2402/small-2402 are 32k; the
        // 2025 generation (Voxtral, Pixtral, Devstral, Magistral, Ministral,
        // Mistral Large 3) is 128k.
        for id in [
            "mistral.mistral-7b-instruct-v0:2",
            "mistral.mixtral-8x7b-instruct-v0:1",
            "mistral.mistral-large-2402-v1:0",
            "mistral.mistral-small-2402-v1:0",
        ] {
            m.insert(id, chat(32_000));
        }
        for id in [
            "mistral.voxtral-mini-3b-2507",
            "mistral.voxtral-small-24b-2507",
            "mistral.pixtral-large-2502-v1:0",
            "mistral.devstral-2-123b",
            "mistral.magistral-small-2509",
            "mistral.ministral-3-3b-instruct",
            "mistral.ministral-3-8b-instruct",
            "mistral.ministral-3-14b-instruct",
            "mistral.mistral-large-3-675b-instruct",
        ] {
            m.insert(id, chat(128_000));
        }

        // DeepSeek — R1 / V3.x at 128k.
        m.insert("deepseek.r1-v1:0", chat(128_000));
        m.insert("deepseek.v3.2", chat(128_000));

        // AI21 Jamba 1.5 — 256k context.
        m.insert("ai21.jamba-1-5-large-v1:0", chat(256_000));
        m.insert("ai21.jamba-1-5-mini-v1:0", chat(256_000));

        // OpenAI gpt-oss — 128k. (Open-weight; Bedrock doesn't run them
        // through the OutputConfig path either.) These always reason via the
        // Harmony format; `gpt_oss()` carries the reasoning_effort surface.
        for id in [
            "openai.gpt-oss-20b-1:0",
            "openai.gpt-oss-120b-1:0",
            "openai.gpt-oss-safeguard-20b",
            "openai.gpt-oss-safeguard-120b",
        ] {
            m.insert(id, gpt_oss(128_000));
        }

        // NVIDIA Nemotron — 128k.
        for id in [
            "nvidia.nemotron-nano-9b-v2",
            "nvidia.nemotron-nano-12b-v2",
            "nvidia.nemotron-nano-3-30b",
            "nvidia.nemotron-super-3-120b",
        ] {
            m.insert(id, chat(128_000));
        }

        // Qwen3 — 128k typical; the 235B VL variant goes higher per Alibaba's
        // model card, but Bedrock surfaces it at 128k.
        for id in [
            "qwen.qwen3-32b-v1:0",
            "qwen.qwen3-coder-30b-a3b-v1:0",
            "qwen.qwen3-coder-next",
            "qwen.qwen3-next-80b-a3b",
            "qwen.qwen3-vl-235b-a22b",
            "qwen.qwen3-235b-a22b-2507-v1:0",
        ] {
            m.insert(id, chat(128_000));
        }

        // Google Gemma 3 — 128k context.
        for id in [
            "google.gemma-3-4b-it",
            "google.gemma-3-12b-it",
            "google.gemma-3-27b-it",
        ] {
            m.insert(id, chat(128_000));
        }

        // Writer Palmyra — 128k.
        for id in [
            "writer.palmyra-x4-v1:0",
            "writer.palmyra-x5-v1:0",
            "writer.palmyra-vision-7b",
        ] {
            m.insert(id, chat(128_000));
        }

        // Z.AI GLM — 128k.
        for id in ["zai.glm-4.7", "zai.glm-4.7-flash", "zai.glm-5"] {
            m.insert(id, chat(128_000));
        }

        // MiniMax M2.x — 128k.
        for id in [
            "minimax.minimax-m2",
            "minimax.minimax-m2.1",
            "minimax.minimax-m2.5",
        ] {
            m.insert(id, chat(128_000));
        }

        // Moonshot Kimi K2 — 256k.
        for id in ["moonshot.kimi-k2-thinking", "moonshotai.kimi-k2.5"] {
            m.insert(id, chat(256_000));
        }

        // TwelveLabs Pegasus — video-only, but `inputModalities` includes
        // TEXT so it passes our text filter. 32k.
        m.insert("twelvelabs.pegasus-1-2-v1:0", chat(32_000));

        // Latency-optimized ("accelerated") inference allow-list, verified
        // against the Bedrock Price List API (`aws pricing get-attribute-values
        // --service-code AmazonBedrock --attribute-name model`): the only
        // models carrying a distinct "* Latency Optimized" pricing SKU are
        // Nova Pro, Llama 3.1 70B, and Llama 3.1 405B — no Claude SKU exists,
        // contrary to some third-party docs. Latency-optimized is reachable
        // only via cross-region profiles and is region-specific, so a wrong
        // entry only costs a missed optimization (AWS auto-falls-back to
        // standard), never correctness. Re-verify with that query when AWS
        // changes the set.
        for id in [
            "amazon.nova-pro-v1:0",
            "amazon.nova-pro-v1:0:24k",
            "amazon.nova-pro-v1:0:300k",
            "meta.llama3-1-70b-instruct-v1:0",
            "meta.llama3-1-405b-instruct-v1:0",
        ] {
            if let Some(caps) = m.get_mut(id) {
                caps.features.insert(ChatFeature::LatencyOptimized);
            }
        }

        // Extended (1-hour) prompt-cache TTL allow-list. AWS GA'd the 1-hour
        // TTL for the Claude 4.5 family (and the newer 4.6/4.7 and Sonnet 5);
        // basic 5-minute caching is broader (see `caching_family_supported`)
        // but the 1-hour tier is Anthropic-only and not the older Opus 4.1.
        // Maintained list —
        // see <https://aws.amazon.com/about-aws/whats-new/2026/01/amazon-bedrock-one-hour-duration-prompt-caching/>.
        for id in [
            "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "anthropic.claude-opus-4-5-20251101-v1:0",
            "anthropic.claude-haiku-4-5-20251001-v1:0",
            "anthropic.claude-sonnet-4-6",
            "anthropic.claude-sonnet-5",
            "anthropic.claude-opus-4-6-v1",
            "anthropic.claude-opus-4-7",
        ] {
            if let Some(caps) = m.get_mut(id) {
                caps.features.insert(ChatFeature::ExtendedCacheTtl);
            }
        }

        m
    });

const fn chat(context_window: u32) -> ChatModelCapabilities {
    // Streaming only; no json_schema. The latency-optimized allow-list at the
    // tail of MODEL_CAPABILITIES adds LatencyOptimized to the eligible
    // foundations; extended TTL is Anthropic-only so no `chat()` model gets it.
    ChatModelCapabilities {
        context_window,
        features: enum_set!(ChatFeature::Streaming),
        reasoning: None,
    }
}

/// Streaming chat model that carries the gpt-oss reasoning surface
/// ([`gpt_oss_reasoning`]). gpt-oss reasons by default; a step minimizes it
/// with `reasoning: { mode: adaptive, effort: low }`, which the provider
/// maps to `reasoning_effort` (see [`super::request::bedrock_additional_request_fields`]).
const fn gpt_oss(context_window: u32) -> ChatModelCapabilities {
    ChatModelCapabilities {
        context_window,
        features: enum_set!(ChatFeature::Streaming),
        reasoning: Some(gpt_oss_reasoning()),
    }
}

/// Derive the media-support table for a Bedrock foundation-model id.
///
/// Bedrock per-family rules are heterogenous: Anthropic Claude on
/// Bedrock accepts image + document via `InlineBytes` / `S3` (no URLs
/// — Bedrock can't fetch); Amazon Nova Pro/Lite adds video; Voxtral
/// adds audio. Models not listed return an empty map (text-only).
///
/// `Url` is intentionally excluded from every entry — Bedrock's
/// `ImageBlock`/`DocumentBlock`/`VideoBlock` `source` field accepts
/// raw bytes or an S3 URI only. Callers carrying a URL must
/// pre-materialise.
pub(super) fn bedrock_media_support(foundation_id: &str) -> BTreeMap<MediaKind, MediaSupport> {
    let bytes_or_s3 = enum_set!(SourceKind::InlineBytes | SourceKind::S3);
    let mut m = BTreeMap::new();

    // Anthropic Claude on Bedrock: image (4 formats, 3.75 MB cap) +
    // document (9 formats, 4.5 MB cap, 100 page PDF cap).
    if foundation_id.starts_with("anthropic.claude-") {
        m.insert(
            MediaKind::Image,
            MediaSupport {
                sources: bytes_or_s3,
                formats: &["png", "jpeg", "gif", "webp"],
                max_bytes: Some(3_932_160), // 3.75 MB
                max_count_per_message: None,
            },
        );
        m.insert(
            MediaKind::Document,
            MediaSupport {
                sources: bytes_or_s3,
                formats: &[
                    "pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md",
                ],
                max_bytes: Some(4_718_592), // 4.5 MB
                max_count_per_message: None,
            },
        );
        return m;
    }

    // Amazon Nova Pro / Lite / 2-Lite: image + document + video.
    // Nova Micro is text-only — handled by falling through.
    if matches!(
        foundation_id,
        "amazon.nova-pro-v1:0"
            | "amazon.nova-pro-v1:0:24k"
            | "amazon.nova-pro-v1:0:300k"
            | "amazon.nova-lite-v1:0"
            | "amazon.nova-lite-v1:0:24k"
            | "amazon.nova-lite-v1:0:300k"
            | "amazon.nova-2-lite-v1:0"
            | "amazon.nova-2-lite-v1:0:256k"
    ) {
        m.insert(
            MediaKind::Image,
            MediaSupport {
                sources: bytes_or_s3,
                formats: &["png", "jpeg", "gif", "webp"],
                max_bytes: Some(3_932_160),
                max_count_per_message: None,
            },
        );
        m.insert(
            MediaKind::Document,
            MediaSupport {
                sources: bytes_or_s3,
                formats: &[
                    "pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md",
                ],
                max_bytes: Some(4_718_592),
                max_count_per_message: None,
            },
        );
        m.insert(
            MediaKind::Video,
            MediaSupport {
                sources: bytes_or_s3,
                formats: &[
                    "mp4", "mov", "mkv", "webm", "flv", "mpeg", "mpg", "wmv", "three_gp",
                ],
                // Inline bytes cap at 25 MB; S3 supports up to 1 GB.
                // Use the inline cap here since `max_bytes` only
                // governs InlineBytes (S3 size isn't locally checked).
                max_bytes: Some(25 * 1024 * 1024),
                max_count_per_message: None,
            },
        );
        return m;
    }

    // Mistral Voxtral: audio.
    if foundation_id.starts_with("mistral.voxtral-") {
        m.insert(
            MediaKind::Audio,
            MediaSupport {
                sources: bytes_or_s3,
                // Voxtral accepts standard audio formats; per
                // AWS docs the model handles wav/mp3/flac/ogg. Until
                // explicit confirmation, list the common subset.
                formats: &["wav", "mpeg", "mp3", "flac"],
                max_bytes: Some(25 * 1024 * 1024),
                max_count_per_message: None,
            },
        );
        return m;
    }

    // Vision-only families (image, no document, no video):
    if matches!(
        foundation_id,
        "mistral.pixtral-large-2502-v1:0"
            | "meta.llama4-scout-17b-instruct-v1:0"
            | "meta.llama4-maverick-17b-instruct-v1:0"
            | "qwen.qwen3-vl-235b-a22b"
            | "writer.palmyra-vision-7b"
            | "google.gemma-3-4b-it"
            | "google.gemma-3-12b-it"
            | "google.gemma-3-27b-it"
            | "nvidia.nemotron-nano-12b-v2"
    ) {
        m.insert(
            MediaKind::Image,
            MediaSupport {
                sources: bytes_or_s3,
                formats: &["png", "jpeg", "gif", "webp"],
                max_bytes: Some(3_932_160),
                max_count_per_message: None,
            },
        );
        return m;
    }

    // Everything else (Llama 3.x text-only, DeepSeek, Cohere, Jamba,
    // Mistral Large/Mixtral/etc., Nova Micro, gpt-oss, Nemotron text,
    // GLM, MiniMax, Kimi, Pegasus-on-Converse=no): text-only.
    m
}

/// Assemble a [`ModelCapabilities`] from a Bedrock model id, its resolved
/// foundation id, and the static table entry. `model_id` is the id as
/// invoked (foundation, cross-region profile, or application-profile ARN);
/// `foundation_id` is what it routes to and keys the media-support table.
pub(super) fn bedrock_model_capabilities(
    model_id: &str,
    foundation_id: &str,
    caps: &ChatModelCapabilities,
) -> ModelCapabilities {
    ModelCapabilities {
        model_id: model_id.to_owned(),
        media_support: bedrock_media_support(foundation_id),
        reasoning: caps.reasoning.clone(),
        latency_optimized_supported: caps.features.contains(ChatFeature::LatencyOptimized),
        extended_cache_ttl_supported: caps.features.contains(ChatFeature::ExtendedCacheTtl),
    }
}
