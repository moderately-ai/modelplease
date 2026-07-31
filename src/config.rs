// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Language model generation configuration.

use enumset::EnumSetType;

/// Reasoning effort hint for thinking models.
///
/// Six qualitative levels covering the union of every supported provider's
/// effort vocabulary. Per-model `ReasoningCapability` (see
/// [`crate::capabilities::ReasoningCapability`]) declares which subset a
/// given model accepts; calls with an unsupported effort fail validation
/// before reaching the wire.
///
/// `EnumSetType` auto-derives `Copy + Clone + PartialEq + Eq` plus the
/// bitset bookkeeping the `enumset` crate needs to use the values inside
/// an `EnumSet`. `Debug` and `Hash` are derived separately.
#[derive(EnumSetType, Debug, Hash)]
pub enum ReasoningEffort {
    /// No reasoning tokens. Wire-meaningful on OpenAI (`reasoning_effort:
    /// "none"`) and Ollama; on Anthropic the same semantics are expressed
    /// by [`ReasoningConfig::Off`] (omit the `thinking` field entirely).
    None,
    /// Shallow reasoning — fast but less thorough.
    Low,
    /// Balanced reasoning — the common default.
    Medium,
    /// Deep reasoning — slowest and most thorough.
    High,
    /// Deeper than `High`. OpenAI gpt-5 family + Anthropic Opus 4.7 only.
    XHigh,
    /// Anthropic adaptive-thinking's open-budget level. Anthropic-family
    /// models (native + Bedrock-Claude) only.
    Max,
}

impl ReasoningEffort {
    /// Wire value the provider API expects.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }
}

impl std::fmt::Display for ReasoningEffort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Error returned when parsing a `ReasoningEffort` from a string that
/// doesn't match one of the six accepted names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidReasoningEffort(pub String);

impl std::fmt::Display for InvalidReasoningEffort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "invalid reasoning effort '{}'; expected one of none, low, medium, high, xhigh, max",
            self.0
        )
    }
}

impl std::error::Error for InvalidReasoningEffort {}

impl std::str::FromStr for ReasoningEffort {
    type Err = InvalidReasoningEffort;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" => Ok(Self::None),
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "xhigh" => Ok(Self::XHigh),
            "max" => Ok(Self::Max),
            _ => Err(InvalidReasoningEffort(s.to_string())),
        }
    }
}

/// Resolved reasoning intent passed to providers via
/// [`LanguageModelConfig::reasoning`].
///
/// Three variants — none, adaptive (qualitative effort), manual (explicit
/// budget). The caller resolves the user-facing "auto" mode against
/// the per-model `ReasoningCapability` before constructing this; providers
/// never see "auto".
///
/// Adaptive on OpenAI / Ollama maps to the `reasoning_effort` string; on
/// Anthropic-family it maps to `thinking: {type: "adaptive", effort: ...}`.
/// Manual is Anthropic-family only — `thinking: {type: "enabled",
/// budget_tokens: N}` natively, and `additionalModelRequestFields.thinking`
/// on Bedrock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReasoningConfig {
    /// Reasoning disabled. Omit the wire field (Anthropic) or send
    /// `reasoning_effort: "none"` (OpenAI/Ollama) depending on the
    /// provider's idiom. Providers decide per-implementation.
    Off,
    /// Adaptive / qualitative-effort mode.
    Adaptive { effort: ReasoningEffort },
    /// Manual fixed-budget mode (Anthropic-family only). `budget_tokens`
    /// must satisfy the per-model `manual_budget_range` from
    /// [`crate::capabilities::ReasoningCapability`] (validated upstream).
    Manual { budget_tokens: u32 },
}

/// Prompt-cache time-to-live for cache breakpoints.
///
/// `FiveMin` is the default ephemeral cache (no special handling on
/// either provider). `OneHour` selects the extended TTL — on Bedrock
/// via `CacheTtl::OneHour`, on Anthropic via `cache_control.ttl = "1h"`
/// behind the `extended-cache-ttl-2025-04-11` beta header. The extended
/// TTL costs ~2× on a cache write but the same ~0.1× on reads, so it
/// wins whenever a byte-stable prefix is reused beyond the 5-minute
/// window. Applies to every breakpoint in the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CacheTtl {
    /// Default 5-minute ephemeral cache.
    #[default]
    FiveMin,
    /// Extended 1-hour cache.
    OneHour,
}

/// Latency tier for the generation call.
///
/// `Optimized` requests AWS Bedrock's latency-optimized ("accelerated")
/// inference — faster decode for a short, region-specific set of models,
/// available only through a cross-region inference profile. Only the
/// Bedrock provider honors this; other providers ignore it. The caller
/// layer gates `Optimized` against the per-model
/// [`ModelCapabilities::latency_optimized_supported`](crate::capabilities::ModelCapabilities::latency_optimized_supported)
/// flag and fails loud before the wire, so a model that can't accept it
/// never reaches Bedrock with the field set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LatencyMode {
    /// Standard latency (default).
    #[default]
    Standard,
    /// Latency-optimized inference (Bedrock only; capability-gated).
    Optimized,
}

/// Prompt-caching mode for the generation call.
///
/// `Auto` (default) lets the provider engage caching whenever the model is
/// caching-capable and a `CacheBreakpoint` is present. `Off` suppresses
/// caching entirely — no `cachePoint` / `cache_control` is emitted even
/// when a breakpoint marker is present. Useful for latency-critical calls
/// with tiny prompts where caching adds a round-trip for near-zero benefit.
///
/// There is deliberately no "force" mode: forcing caching past the
/// per-model capability gate is what produces the runtime rejections this
/// mode exists to avoid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PromptCaching {
    /// Capability-gated caching (default).
    #[default]
    Auto,
    /// Never emit cache breakpoints for this request.
    Off,
}

/// Configuration for a language model generation call.
///
/// All fields are optional — callers set only what they need. Use
/// `LanguageModelConfig::default()` for provider defaults.
#[derive(Debug, Clone, Default)]
pub struct LanguageModelConfig {
    /// Sampling temperature (0.0 = deterministic, higher = more random).
    pub temperature: Option<f64>,
    /// Maximum number of tokens to generate.
    pub max_tokens: Option<u32>,
    /// Nucleus sampling threshold.
    pub top_p: Option<f64>,
    /// Stop sequences — generation halts when any of these are produced.
    pub stop: Vec<String>,
    /// Reasoning / extended-thinking intent. `None` (default) means the
    /// caller hasn't expressed an intent; providers send no reasoning
    /// fields at all. `Some(_)` carries fully-resolved mode + effort,
    /// already validated against the model's `ReasoningCapability` by
    /// the caller.
    pub reasoning: Option<ReasoningConfig>,
    /// Constrain the output format (JSON, schema-conformant JSON, etc.).
    ///
    /// Defaults to [`ResponseFormat::Text`] (unconstrained).
    pub response_format: ResponseFormat,
    /// TTL applied to any prompt-cache breakpoints emitted for this
    /// request. Defaults to [`CacheTtl::FiveMin`]; only meaningful when
    /// caching is engaged (a caching-capable model + a `CacheBreakpoint`
    /// in the messages).
    pub cache_ttl: CacheTtl,
    /// Latency tier. Defaults to [`LatencyMode::Standard`]. Only the
    /// Bedrock provider honors [`LatencyMode::Optimized`], and only for
    /// capability-flagged models reached via a cross-region inference
    /// profile; other providers ignore it.
    pub latency: LatencyMode,
    /// Prompt-caching mode. Defaults to [`PromptCaching::Auto`]
    /// (capability-gated). [`PromptCaching::Off`] suppresses cache
    /// breakpoints entirely for this request.
    pub prompt_caching: PromptCaching,
}

/// Constrains the output format of a language model response.
///
/// Providers that don't support a given format return
/// [`LanguageModelError::Provider`](crate::LanguageModelError::Provider).
///
/// # Provider support
///
/// | Format | OpenAI | Anthropic | Ollama |
/// |--------|--------|-----------|--------|
/// | `Text` | All models | All models | All models |
/// | `JsonObject` | Most models | Not supported | Supported |
/// | `JsonSchema` | Newer models | Beta (sonnet-4-5+, opus-4-1+) | Supported |
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub enum ResponseFormat {
    /// No constraint — free-form text (default).
    #[default]
    Text,
    /// Must produce valid JSON (no schema constraint).
    ///
    /// `OpenAI` requires that you also instruct the model to produce JSON in
    /// your system or user messages — setting this alone is not sufficient.
    JsonObject,
    /// Must produce JSON conforming to the given schema.
    JsonSchema {
        /// A name for this schema (used by `OpenAI`, max 64 chars, `[a-zA-Z0-9_-]`).
        name: String,
        /// The JSON Schema definition.
        schema: serde_json::Value,
        /// Whether to enforce strict schema adherence.
        strict: bool,
    },
}
