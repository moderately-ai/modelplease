// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::{BTreeMap, HashMap},
    sync::LazyLock,
};

use enumset::enum_set;

use crate::{
    anthropic_wire::{mantle_anthropic_adaptive_reasoning, mantle_anthropic_haiku_reasoning},
    capabilities::{
        MediaKind, MediaSupport, ReasoningCapability, ReasoningMode, ReasoningParamConflicts,
    },
    config::ReasoningEffort,
    media::SourceKind,
};

/// Per-model capability data for the Mantle catalog.
///
/// Wraps the same fields as the other providers' tables, populated from each model's AWS
/// model-card page. Entries are conservative on reasoning + media (declared only when documented),
/// because [`LanguageModelProvider::validate_request`] uses them to reject ill-formed requests
/// *before* they hit the wire. Wrong-but-too-permissive entries surface as Mantle 400s; wrong-but-
/// too-restrictive entries reject calls that would have succeeded. When uncertain we lean
/// permissive on context_window / json_schema (informational) and restrictive on reasoning /
/// media (enforced).
#[derive(Debug, Clone)]
pub(super) struct MantleModelCapabilities {
    pub(super) context_window: u32,
    pub(super) supports_streaming: bool,
    pub(super) supports_json_schema: bool,
    pub(super) reasoning: Option<ReasoningCapability>,
    pub(super) media_support: BTreeMap<MediaKind, MediaSupport>,
}

// --- Media support helpers per modality ---

/// Image support shared by every vision-capable Chat Completions model on Mantle.
///
/// Same sources OpenAI Chat Completions accepts (`Url` + `InlineBytes` as a `data:` URI), the
/// same subtype list, and the same 20 MB local size cap — those are the values
/// [`crate::openai::translate_part_for_openai`] emits, and Mantle's surface accepts the same
/// shapes since it speaks the OpenAI wire format. If a specific Mantle vision model publishes a
/// tighter constraint we'll split into per-family helpers; for now this is the conservative
/// default.
const fn mantle_image_support() -> MediaSupport {
    MediaSupport {
        sources: enum_set!(SourceKind::Url | SourceKind::InlineBytes),
        formats: &["png", "jpeg", "webp", "gif"],
        max_bytes: Some(20 * 1024 * 1024),
        max_count_per_message: None,
    }
}

/// Audio support for the Voxtral mini / small models on Mantle. Per AWS Voxtral docs the model
/// accepts `audio/wav`, `audio/mpeg`, `audio/mp3`, `audio/flac` with a 25 MB inline-bytes cap.
/// `Url` is not exposed at the Chat Completions content-part layer for audio, so InlineBytes is
/// the only source kind here.
const fn mantle_voxtral_audio_support() -> MediaSupport {
    MediaSupport {
        sources: enum_set!(SourceKind::InlineBytes),
        formats: &["wav", "mpeg", "mp3", "flac"],
        max_bytes: Some(25 * 1024 * 1024),
        max_count_per_message: None,
    }
}

// --- Reasoning capability constructors per family ---

/// OpenAI reasoning models on Mantle (gpt-oss-* via Chat Completions, gpt-5.* via Responses) all
/// reject `temperature` while reasoning is active. `top_p` has no documented restriction.
const fn openai_reasoning_conflicts() -> ReasoningParamConflicts {
    ReasoningParamConflicts {
        temperature_forbidden: true,
        top_k_forbidden: false,
        top_p_allowed_range: None,
    }
}

/// `ReasoningCapability` for `openai.gpt-5.{4,5}` — full 5-value effort enum (none / low /
/// medium / high / xhigh) per OpenAI's published spec. Routed via the Responses surface (phase 6).
const fn mantle_gpt5_reasoning() -> ReasoningCapability {
    ReasoningCapability {
        supported_modes: enum_set!(ReasoningMode::Adaptive),
        supported_efforts: enum_set!(
            ReasoningEffort::None
                | ReasoningEffort::Low
                | ReasoningEffort::Medium
                | ReasoningEffort::High
                | ReasoningEffort::XHigh
        ),
        manual_budget_range: None,
        conflicts: openai_reasoning_conflicts(),
        sampling_params_removed: false,
    }
}

/// `ReasoningCapability` for OpenAI's open-weight `gpt-oss-*` family on Mantle — `reasoning_effort`
/// in `{low, medium, high}` only (no `xhigh`, no `none`-as-effort — `Off` covers that path).
const fn mantle_gpt_oss_reasoning() -> ReasoningCapability {
    ReasoningCapability {
        supported_modes: enum_set!(ReasoningMode::Adaptive),
        supported_efforts: enum_set!(
            ReasoningEffort::Low | ReasoningEffort::Medium | ReasoningEffort::High
        ),
        manual_budget_range: None,
        conflicts: openai_reasoning_conflicts(),
        sampling_params_removed: false,
    }
}

/// `ReasoningCapability` for the generic open-weight thinking-mode families — DeepSeek-R1-style,
/// Qwen3 thinking, MiniMax M2.x, Kimi K2 thinking, Nemotron, Magistral. Accept the OpenAI-shape
/// `reasoning_effort` field with `{low, medium, high}`. Conflicts are conservative (forbid
/// `temperature` when reasoning is on, matching the OpenAI-shape Chat Completions surface).
const fn mantle_open_thinking_reasoning() -> ReasoningCapability {
    ReasoningCapability {
        supported_modes: enum_set!(ReasoningMode::Adaptive),
        supported_efforts: enum_set!(
            ReasoningEffort::Low | ReasoningEffort::Medium | ReasoningEffort::High
        ),
        manual_budget_range: None,
        conflicts: openai_reasoning_conflicts(),
        sampling_params_removed: false,
    }
}

/// Static capability table for the Mantle catalog. Populated from each model's AWS model-card
/// page; a live `GET /v1/models` on 2026-06-04 returned 41–42 models per
/// region — we ship a row for each. Unknown IDs returned by upstream fire a one-time
/// `tracing::warn!` per cache fill (see [`fetch_and_merge_models`](Self::fetch_and_merge_models)).
///
/// Phase 3 ships text-only entries; phase 4 adds the per-model `media_support` rows for the
/// vision-capable families (Qwen3-VL, Palmyra-Vision-7B, Gemma 3, Nemotron-12B-VL) and audio
/// for the Voxtral family.
pub(super) static MODEL_CAPABILITIES: LazyLock<HashMap<&'static str, MantleModelCapabilities>> =
    LazyLock::new(|| {
        let mut m = HashMap::new();

        // --- OpenAI frontier (Responses surface, phase 6) ---
        for id in ["openai.gpt-5.5", "openai.gpt-5.4"] {
            m.insert(
                id,
                MantleModelCapabilities {
                    context_window: 1_000_000,
                    supports_streaming: true,
                    supports_json_schema: true,
                    reasoning: Some(mantle_gpt5_reasoning()),
                    media_support: BTreeMap::new(),
                },
            );
        }
        // Dated variants — same caps as their base id.
        for id in ["openai.gpt-5.5-2026-04-23", "openai.gpt-5.4-2026-03-05"] {
            m.insert(
                id,
                MantleModelCapabilities {
                    context_window: 1_000_000,
                    supports_streaming: true,
                    supports_json_schema: true,
                    reasoning: Some(mantle_gpt5_reasoning()),
                    media_support: BTreeMap::new(),
                },
            );
        }

        // --- OpenAI open-weight (Chat Completions) ---
        for id in ["openai.gpt-oss-120b", "openai.gpt-oss-20b"] {
            m.insert(
                id,
                MantleModelCapabilities {
                    context_window: 128_000,
                    supports_streaming: true,
                    supports_json_schema: true,
                    reasoning: Some(mantle_gpt_oss_reasoning()),
                    media_support: BTreeMap::new(),
                },
            );
        }
        // Safeguard is a classifier — no reasoning surface.
        for id in [
            "openai.gpt-oss-safeguard-120b",
            "openai.gpt-oss-safeguard-20b",
        ] {
            m.insert(
                id,
                MantleModelCapabilities {
                    context_window: 128_000,
                    supports_streaming: true,
                    supports_json_schema: true,
                    reasoning: None,
                    media_support: BTreeMap::new(),
                },
            );
        }

        // --- Anthropic Claude on Mantle (Messages surface, phase 5) ---
        // Per AWS Mythos docs, Mantle's Messages path does NOT support prompt caching (callers
        // wanting caching go through bedrock-runtime / Converse). Media support lives in phase 4
        // with the rest of the vision/document content-part wiring.
        m.insert(
            "anthropic.claude-haiku-4-5",
            MantleModelCapabilities {
                context_window: 200_000,
                supports_streaming: true,
                supports_json_schema: true,
                reasoning: Some(mantle_anthropic_haiku_reasoning()),
                media_support: BTreeMap::new(),
            },
        );
        m.insert(
            "anthropic.claude-opus-4-7",
            MantleModelCapabilities {
                context_window: 1_000_000,
                supports_streaming: true,
                supports_json_schema: true,
                reasoning: Some(mantle_anthropic_adaptive_reasoning()),
                media_support: BTreeMap::new(),
            },
        );
        m.insert(
            "anthropic.claude-opus-4-8",
            MantleModelCapabilities {
                context_window: 1_000_000,
                supports_streaming: true,
                supports_json_schema: true,
                reasoning: Some(mantle_anthropic_adaptive_reasoning()),
                media_support: BTreeMap::new(),
            },
        );
        m.insert(
            "anthropic.claude-mythos-preview",
            MantleModelCapabilities {
                context_window: 1_000_000,
                supports_streaming: true,
                supports_json_schema: true,
                reasoning: Some(mantle_anthropic_adaptive_reasoning()),
                media_support: BTreeMap::new(),
            },
        );

        // --- DeepSeek ---
        // V3.1 is a hybrid reasoning model (R1-style on demand). V3.2 is a non-reasoning variant.
        m.insert(
            "deepseek.v3.1",
            MantleModelCapabilities {
                context_window: 128_000,
                supports_streaming: true,
                supports_json_schema: true,
                reasoning: Some(mantle_open_thinking_reasoning()),
                media_support: BTreeMap::new(),
            },
        );
        m.insert(
            "deepseek.v3.2",
            MantleModelCapabilities {
                context_window: 128_000,
                supports_streaming: true,
                supports_json_schema: true,
                reasoning: None,
                media_support: BTreeMap::new(),
            },
        );

        // --- Mistral ---
        // Devstral (coding), Mistral-Large-3, Ministral 3-* (text), Voxtral mini/small (text +
        // audio — audio support lands in phase 4). Magistral-small is a reasoning model.
        m.insert(
            "mistral.devstral-2-123b",
            MantleModelCapabilities {
                context_window: 256_000,
                supports_streaming: true,
                supports_json_schema: true,
                reasoning: None,
                media_support: BTreeMap::new(),
            },
        );
        m.insert(
            "mistral.magistral-small-2509",
            MantleModelCapabilities {
                context_window: 128_000,
                supports_streaming: true,
                supports_json_schema: true,
                reasoning: Some(mantle_open_thinking_reasoning()),
                media_support: BTreeMap::new(),
            },
        );
        m.insert(
            "mistral.mistral-large-3-675b-instruct",
            MantleModelCapabilities {
                context_window: 128_000,
                supports_streaming: true,
                supports_json_schema: true,
                reasoning: None,
                media_support: BTreeMap::new(),
            },
        );
        for id in [
            "mistral.ministral-3-3b-instruct",
            "mistral.ministral-3-8b-instruct",
            "mistral.ministral-3-14b-instruct",
        ] {
            m.insert(
                id,
                MantleModelCapabilities {
                    context_window: 128_000,
                    supports_streaming: true,
                    supports_json_schema: true,
                    reasoning: None,
                    media_support: BTreeMap::new(),
                },
            );
        }
        // Voxtral — audio-input chat models, 25 MB inline-bytes cap on wav/mp3/mpeg/flac.
        for id in [
            "mistral.voxtral-mini-3b-2507",
            "mistral.voxtral-small-24b-2507",
        ] {
            let mut media = BTreeMap::new();
            media.insert(MediaKind::Audio, mantle_voxtral_audio_support());
            m.insert(
                id,
                MantleModelCapabilities {
                    context_window: 32_000,
                    supports_streaming: true,
                    supports_json_schema: true,
                    reasoning: None,
                    media_support: media,
                },
            );
        }

        // --- Google Gemma 3 (vision-capable text+image chat models) ---
        for id in [
            "google.gemma-3-4b-it",
            "google.gemma-3-12b-it",
            "google.gemma-3-27b-it",
        ] {
            let mut media = BTreeMap::new();
            media.insert(MediaKind::Image, mantle_image_support());
            m.insert(
                id,
                MantleModelCapabilities {
                    context_window: 128_000,
                    supports_streaming: true,
                    supports_json_schema: true,
                    reasoning: None,
                    media_support: media,
                },
            );
        }

        // --- Qwen3 ---
        // Text-only Qwen3 variants share text-only media. qwen3-vl-* is the vision-capable
        // variant — declares Image media_support.
        for id in [
            "qwen.qwen3-32b",
            "qwen.qwen3-235b-a22b-2507",
            "qwen.qwen3-next-80b-a3b-instruct",
            "qwen.qwen3-coder-30b-a3b-instruct",
            "qwen.qwen3-coder-480b-a35b-instruct",
            "qwen.qwen3-coder-next",
        ] {
            m.insert(
                id,
                MantleModelCapabilities {
                    context_window: 128_000,
                    supports_streaming: true,
                    supports_json_schema: true,
                    reasoning: Some(mantle_open_thinking_reasoning()),
                    media_support: BTreeMap::new(),
                },
            );
        }
        {
            let mut media = BTreeMap::new();
            media.insert(MediaKind::Image, mantle_image_support());
            m.insert(
                "qwen.qwen3-vl-235b-a22b-instruct",
                MantleModelCapabilities {
                    context_window: 128_000,
                    supports_streaming: true,
                    supports_json_schema: true,
                    reasoning: Some(mantle_open_thinking_reasoning()),
                    media_support: media,
                },
            );
        }

        // --- NVIDIA Nemotron ---
        // Text-only nano-9b / nano-3-30b / super-3-120b. nano-12b is the vision-capable (VL)
        // variant.
        for id in [
            "nvidia.nemotron-nano-9b-v2",
            "nvidia.nemotron-nano-3-30b",
            "nvidia.nemotron-super-3-120b",
        ] {
            m.insert(
                id,
                MantleModelCapabilities {
                    context_window: 128_000,
                    supports_streaming: true,
                    supports_json_schema: true,
                    reasoning: Some(mantle_open_thinking_reasoning()),
                    media_support: BTreeMap::new(),
                },
            );
        }
        {
            let mut media = BTreeMap::new();
            media.insert(MediaKind::Image, mantle_image_support());
            m.insert(
                "nvidia.nemotron-nano-12b-v2",
                MantleModelCapabilities {
                    context_window: 128_000,
                    supports_streaming: true,
                    supports_json_schema: true,
                    reasoning: Some(mantle_open_thinking_reasoning()),
                    media_support: media,
                },
            );
        }

        // --- MiniMax M2 family ---
        for id in [
            "minimax.minimax-m2",
            "minimax.minimax-m2.1",
            "minimax.minimax-m2.5",
        ] {
            m.insert(
                id,
                MantleModelCapabilities {
                    context_window: 200_000,
                    supports_streaming: true,
                    supports_json_schema: true,
                    reasoning: Some(mantle_open_thinking_reasoning()),
                    media_support: BTreeMap::new(),
                },
            );
        }

        // --- Moonshot Kimi ---
        // K2-thinking is the reasoning-mode variant; K2.5 is the standard chat model with toggle.
        m.insert(
            "moonshotai.kimi-k2-thinking",
            MantleModelCapabilities {
                context_window: 200_000,
                supports_streaming: true,
                supports_json_schema: true,
                reasoning: Some(mantle_open_thinking_reasoning()),
                media_support: BTreeMap::new(),
            },
        );
        m.insert(
            "moonshotai.kimi-k2.5",
            MantleModelCapabilities {
                context_window: 256_000,
                supports_streaming: true,
                supports_json_schema: true,
                reasoning: Some(mantle_open_thinking_reasoning()),
                media_support: BTreeMap::new(),
            },
        );

        // --- Z.AI GLM ---
        // Reasoning support not documented at parity with the open-thinking family yet — leave
        // `reasoning: None` until verified. Wrong-too-permissive here is worse than wrong-too-
        // restrictive (we'd reject valid Adaptive requests vs. the model just ignoring the field).
        for id in [
            "zai.glm-4.6",
            "zai.glm-4.7",
            "zai.glm-4.7-flash",
            "zai.glm-5",
        ] {
            m.insert(
                id,
                MantleModelCapabilities {
                    context_window: 128_000,
                    supports_streaming: true,
                    supports_json_schema: true,
                    reasoning: None,
                    media_support: BTreeMap::new(),
                },
            );
        }

        // --- Writer Palmyra-Vision ---
        // Vision-capable text+image chat model. No reasoning surface.
        {
            let mut media = BTreeMap::new();
            media.insert(MediaKind::Image, mantle_image_support());
            m.insert(
                "writer.palmyra-vision-7b",
                MantleModelCapabilities {
                    context_window: 128_000,
                    supports_streaming: true,
                    supports_json_schema: true,
                    reasoning: None,
                    media_support: media,
                },
            );
        }

        m
    });
