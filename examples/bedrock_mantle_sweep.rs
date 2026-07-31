// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    reason = "examples exist to demo the API and emit results to the terminal; the workspace \
              bans on print_*/expect/casts target production code, and a benchmark harness is \
              legitimately one long top-to-bottom flow"
)]

//! Runtime sweep across the open-weight Bedrock Mantle catalog.
//!
//! Sends a ~2k-token prompt (LongBench-style "given this document, write an
//! analysis" framing wrapped around the public-domain Declaration of
//! Independence) to each model N times with bounded concurrency, captures
//! token / latency / cost / stop-reason per sample, and writes a JSON
//! artifact with per-model aggregates.
//!
//! Every request streams (`generate_stream`) so we can measure time-to-first-
//! token separately from end-to-end latency. Caching is disabled
//! (`PromptCaching::Off`) so cold-call numbers are comparable across runs.
//!
//! Usage (with AWS credentials):
//!   cargo run --release \
//!     -p modelplease --example bedrock_mantle_sweep -- \
//!     --samples-per-model 5 --concurrency 8 \
//!     --output /tmp/bedrock-mantle-sweep.json
//!
//! Env overrides (same defaults as `bedrock_mantle_test`):
//!   BEDROCK_MANTLE_DEFAULT_REGION       (default us-west-2)
//!   BEDROCK_MANTLE_OPENAI_GPT5_REGION   (default us-east-2)
//!   BEDROCK_MANTLE_ANTHROPIC_REGION     (default us-east-1)
//!   BEDROCK_MANTLE_API_KEY              if set, Bearer auth instead of SigV4

use std::{sync::Arc, time::Instant};

use aws_config::BehaviorVersion;
mod common;

use chrono::{DateTime, Utc};
use clap::{Parser, ValueEnum};
use common::build_http_client;
use futures::StreamExt;
use modelplease::{
    BedrockMantleAuth, BedrockMantleProvider, BedrockMantleProviderConfig,
    BedrockMantleProviderDeps, BedrockProvider, BedrockProviderConfig, BedrockProviderDeps,
    GenerateRequest, LanguageModelConfig, LanguageModelProvider, Message, ModelId, PromptCaching,
    RetryConfig, StopReason,
};
use serde::Serialize;

/// Public-domain US founding document used as the input payload.
///
/// Source text fetched from `archives.gov/founding-docs/declaration-transcript`
/// (NARA transcription). ~1,330 words / ~1,700 tokens by GPT-style BPE — fits
/// the LongBench-style "2k-token document + 300-token instruction" mold.
const SAMPLE_DOCUMENT: &str = include_str!("data/declaration_of_independence.txt");

/// System message — frames the task. Counts toward input tokens.
const SYSTEM_PROMPT: &str = "You are a careful historian and political-philosophy reviewer. \
You write rigorous, well-structured analyses grounded in the primary text. You always cite \
specific phrases from the document when you make a claim about it.";

/// User instruction wrapped around `SAMPLE_DOCUMENT`. Asks for a long-enough
/// answer to push past the 500-output-token floor without runaway verbosity.
/// Stays identical across every model in the sweep so results are comparable
/// — never branch on model id, even when a model's house-style guide
/// would prefer a different framing.
const USER_INSTRUCTION: &str = "Read the historical document delimited below and produce a \
single 700-word analytical essay. Structure the essay as five themed paragraphs of running \
prose — no bullet lists, no section headers, no markdown — and make sure to address every \
one of the following analytical dimensions:\n\
\n\
  1. Historical context. Identify the author of the document, the body that ratified it, the \
     immediate political and military situation it responds to, and the dates relevant to \
     its drafting, signing, and public circulation. Place the document in its broader \
     trans-Atlantic context (the wider eighteenth-century revolutionary moment, ongoing wars \
     of empire, and the colonial dispute over taxation and representation).\n\
\n\
  2. Rhetorical structure. Identify the classical oratorical sections (exordium, narration, \
     partition, confirmation, refutation, peroration) and show how each section advances the \
     overall argument. Explain why this particular ordering is rhetorically effective for the \
     document's intended dual audience (the colonial population and the international powers \
     whose recognition the new state would need).\n\
\n\
  3. Philosophical lineage of the natural-rights claims. Trace the document's central \
     premises (equality, unalienable rights, government by consent, the right of revolution) \
     to their philosophical antecedents in John Locke's Second Treatise, the Scottish \
     Enlightenment (Hutcheson, Reid, Hume), and English common-law constitutionalism. \
     Distinguish where the document follows Locke faithfully from where it departs from him.\n\
\n\
  4. The specific grievances. Group the enumerated charges against the Crown into thematic \
     clusters (legislative obstruction, judicial subversion, military overreach, economic \
     coercion, denial of self-government, encouragement of internal and external violence). \
     For each cluster, explain what specific colonial-era policies the grievance refers to \
     and why each one is structured as a violation of a stated natural-rights principle.\n\
\n\
  5. Influence on subsequent legal and political traditions. Trace the document's influence \
     on the United States Constitution and Bill of Rights, the French Declaration of the \
     Rights of Man and of the Citizen (1789), nineteenth-century abolitionist and women's \
     suffrage rhetoric (especially the Seneca Falls Declaration of Sentiments), twentieth- \
     and twenty-first-century anti-colonial declarations of independence, and the Universal \
     Declaration of Human Rights. Identify at least two later documents that explicitly \
     mirror its structure or borrow its language, and one significant criticism of the \
     document raised by later commentators (e.g. its silence on slavery, its treatment of \
     Indigenous nations, or its rhetorical inconsistencies).\n\
\n\
Requirements:\n\
  - Quote at least six distinct phrases from the document verbatim, in double quotation \
    marks, and weave each quotation into the analysis rather than presenting it as a stand- \
    alone snippet.\n\
  - Integrate evidence across the five dimensions. Do not summarize the document section by \
    section.\n\
  - Stay close to 700 words. A response significantly shorter than 600 words or longer than \
    900 words fails the brief. Do not pad with filler, restated claims, or repeated \
    quotations to reach length; tighter is better than longer.\n\
  - Stop writing as soon as the five dimensions have been substantively addressed. Do not \
    append a conclusion that merely restates the introduction, a meta-commentary on your \
    own essay, a self-assessment, or any reflection on the task itself.\n\
  - Do not include a title, byline, table of contents, headers, sub-headers, footnotes, or \
    citations — only the running prose.\n\
\n\
=== DOCUMENT ===\n\
{document}\n\
=== END DOCUMENT ===";

/// One row per open-weight model. The Mantle id is the canonical key (groups
/// samples in the artifact and prices in `PRICING`); `runtime_id` is the
/// matching `bedrock-runtime` (Converse) foundation-model id, or `None` when
/// the model exists only on Mantle.
///
/// Bedrock-runtime uses `-vN:0` version suffixes on some families
/// (`openai.gpt-oss-120b-1:0`, `qwen.qwen3-32b-v1:0`) and drops `-instruct`
/// on the Qwen3 line. The differences come straight from
/// `aws bedrock list-foundation-models --region us-west-2` (2026-06-05).
struct ModelEntry {
    mantle_id: &'static str,
    runtime_id: Option<&'static str>,
}

const OPEN_WEIGHT_MODELS: &[ModelEntry] = &[
    // Closed-weight Anthropic — included so cost / latency / intelligence comparisons against the
    // open-weight catalog have a frontier-class reference point. Routes through the Anthropic
    // Messages surface on Mantle and Converse on bedrock-runtime (US cross-region inference
    // profile required on runtime for on-demand quota).
    ModelEntry {
        mantle_id: "anthropic.claude-haiku-4-5",
        runtime_id: Some("us.anthropic.claude-haiku-4-5-20251001-v1:0"),
    },
    // Closed-weight OpenAI — routes through the Responses surface on Mantle (Mantle-only;
    // bedrock-runtime does not host gpt-5.x). Pricing is not in the live AWS Pricing API
    // mapping, so cost_usd is `None` for these rows; latency and stop_reason are the load-
    // bearing measurements here.
    ModelEntry {
        mantle_id: "openai.gpt-5.5",
        runtime_id: None,
    },
    ModelEntry {
        mantle_id: "openai.gpt-oss-120b",
        runtime_id: Some("openai.gpt-oss-120b-1:0"),
    },
    ModelEntry {
        mantle_id: "openai.gpt-oss-20b",
        runtime_id: Some("openai.gpt-oss-20b-1:0"),
    },
    // deepseek.v3.1 is mantle-only; bedrock-runtime only hosts v3 (=3.0) and v3.2.
    ModelEntry {
        mantle_id: "deepseek.v3.1",
        runtime_id: None,
    },
    ModelEntry {
        mantle_id: "deepseek.v3.2",
        runtime_id: Some("deepseek.v3.2"),
    },
    ModelEntry {
        mantle_id: "mistral.devstral-2-123b",
        runtime_id: Some("mistral.devstral-2-123b"),
    },
    ModelEntry {
        mantle_id: "mistral.magistral-small-2509",
        runtime_id: Some("mistral.magistral-small-2509"),
    },
    ModelEntry {
        mantle_id: "mistral.ministral-3-3b-instruct",
        runtime_id: Some("mistral.ministral-3-3b-instruct"),
    },
    ModelEntry {
        mantle_id: "mistral.ministral-3-8b-instruct",
        runtime_id: Some("mistral.ministral-3-8b-instruct"),
    },
    ModelEntry {
        mantle_id: "mistral.ministral-3-14b-instruct",
        runtime_id: Some("mistral.ministral-3-14b-instruct"),
    },
    ModelEntry {
        mantle_id: "mistral.mistral-large-3-675b-instruct",
        runtime_id: Some("mistral.mistral-large-3-675b-instruct"),
    },
    ModelEntry {
        mantle_id: "qwen.qwen3-32b",
        runtime_id: Some("qwen.qwen3-32b-v1:0"),
    },
    ModelEntry {
        mantle_id: "qwen.qwen3-235b-a22b-2507",
        runtime_id: Some("qwen.qwen3-235b-a22b-2507-v1:0"),
    },
    ModelEntry {
        mantle_id: "qwen.qwen3-coder-30b-a3b-instruct",
        runtime_id: Some("qwen.qwen3-coder-30b-a3b-v1:0"),
    },
    ModelEntry {
        mantle_id: "qwen.qwen3-coder-480b-a35b-instruct",
        runtime_id: Some("qwen.qwen3-coder-480b-a35b-v1:0"),
    },
    ModelEntry {
        mantle_id: "qwen.qwen3-coder-next",
        runtime_id: Some("qwen.qwen3-coder-next"),
    },
    ModelEntry {
        mantle_id: "qwen.qwen3-next-80b-a3b-instruct",
        runtime_id: Some("qwen.qwen3-next-80b-a3b"),
    },
    ModelEntry {
        mantle_id: "qwen.qwen3-vl-235b-a22b-instruct",
        runtime_id: Some("qwen.qwen3-vl-235b-a22b"),
    },
    ModelEntry {
        mantle_id: "google.gemma-3-4b-it",
        runtime_id: Some("google.gemma-3-4b-it"),
    },
    ModelEntry {
        mantle_id: "google.gemma-3-12b-it",
        runtime_id: Some("google.gemma-3-12b-it"),
    },
    ModelEntry {
        mantle_id: "google.gemma-3-27b-it",
        runtime_id: Some("google.gemma-3-27b-it"),
    },
    ModelEntry {
        mantle_id: "nvidia.nemotron-nano-9b-v2",
        runtime_id: Some("nvidia.nemotron-nano-9b-v2"),
    },
    ModelEntry {
        mantle_id: "nvidia.nemotron-nano-12b-v2",
        runtime_id: Some("nvidia.nemotron-nano-12b-v2"),
    },
    ModelEntry {
        mantle_id: "nvidia.nemotron-nano-3-30b",
        runtime_id: Some("nvidia.nemotron-nano-3-30b"),
    },
    ModelEntry {
        mantle_id: "nvidia.nemotron-super-3-120b",
        runtime_id: Some("nvidia.nemotron-super-3-120b"),
    },
    ModelEntry {
        mantle_id: "minimax.minimax-m2",
        runtime_id: Some("minimax.minimax-m2"),
    },
    ModelEntry {
        mantle_id: "minimax.minimax-m2.1",
        runtime_id: Some("minimax.minimax-m2.1"),
    },
    ModelEntry {
        mantle_id: "minimax.minimax-m2.5",
        runtime_id: Some("minimax.minimax-m2.5"),
    },
    ModelEntry {
        mantle_id: "moonshotai.kimi-k2.5",
        runtime_id: Some("moonshotai.kimi-k2.5"),
    },
    ModelEntry {
        mantle_id: "zai.glm-4.7",
        runtime_id: Some("zai.glm-4.7"),
    },
    ModelEntry {
        mantle_id: "zai.glm-4.7-flash",
        runtime_id: Some("zai.glm-4.7-flash"),
    },
    ModelEntry {
        mantle_id: "zai.glm-5",
        runtime_id: Some("zai.glm-5"),
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Platform {
    /// `bedrock-mantle.{region}.api.aws` — OpenAI / Anthropic-compatible HTTP.
    Mantle,
    /// `bedrockruntime.{region}.amazonaws.com` — Converse / ConverseStream
    /// via the typed AWS SDK.
    Runtime,
}

impl Platform {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Mantle => "mantle",
            Self::Runtime => "runtime",
        }
    }
}

impl std::fmt::Display for Platform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// On-demand $/MTok pricing per model id. Pulled live from the AWS Pricing
/// API via `scripts/bedrock-pricing.py --json` on 2026-06-05 and embedded
/// here so the harness can compute cost-per-sample without a Pricing API
/// round-trip during the sweep. Refresh when AWS adjusts pricing or new
/// open-weight models land on Mantle.
const PRICING: &[(&str, f64, f64)] = &[
    // (model_id, input_per_mtok_usd, output_per_mtok_usd)
    ("anthropic.claude-haiku-4-5", 1.00, 5.00),
    ("openai.gpt-oss-120b", 0.15, 0.60),
    ("openai.gpt-oss-20b", 0.07, 0.30),
    ("deepseek.v3.1", 0.58, 1.68),
    ("deepseek.v3.2", 0.62, 1.85),
    ("mistral.devstral-2-123b", 0.40, 2.00),
    ("mistral.magistral-small-2509", 0.50, 1.50),
    ("mistral.ministral-3-3b-instruct", 0.10, 0.10),
    ("mistral.ministral-3-8b-instruct", 0.15, 0.15),
    ("mistral.ministral-3-14b-instruct", 0.20, 0.20),
    ("mistral.mistral-large-3-675b-instruct", 0.50, 1.50),
    ("qwen.qwen3-32b", 0.15, 0.60),
    ("qwen.qwen3-235b-a22b-2507", 0.22, 0.88),
    ("qwen.qwen3-coder-30b-a3b-instruct", 0.15, 0.60),
    ("qwen.qwen3-coder-480b-a35b-instruct", 0.45, 1.80),
    ("qwen.qwen3-coder-next", 0.50, 1.20),
    ("qwen.qwen3-next-80b-a3b-instruct", 0.14, 1.20),
    ("qwen.qwen3-vl-235b-a22b-instruct", 0.53, 2.66),
    ("google.gemma-3-4b-it", 0.04, 0.08),
    ("google.gemma-3-12b-it", 0.09, 0.29),
    ("google.gemma-3-27b-it", 0.23, 0.38),
    ("nvidia.nemotron-nano-9b-v2", 0.06, 0.23),
    ("nvidia.nemotron-nano-12b-v2", 0.20, 0.60),
    ("nvidia.nemotron-nano-3-30b", 0.06, 0.24),
    ("nvidia.nemotron-super-3-120b", 0.15, 0.65),
    ("minimax.minimax-m2", 0.30, 1.20),
    ("minimax.minimax-m2.1", 0.30, 1.20),
    ("minimax.minimax-m2.5", 0.30, 1.20),
    ("moonshotai.kimi-k2.5", 0.60, 3.00),
    ("zai.glm-4.7", 0.60, 2.20),
    ("zai.glm-4.7-flash", 0.07, 0.40),
    ("zai.glm-5", 1.00, 3.20),
];

fn price_for(model_id: &str) -> Option<(f64, f64)> {
    PRICING
        .iter()
        .find(|(id, _, _)| *id == model_id)
        .map(|(_, i, o)| (*i, *o))
}

#[derive(Parser, Debug)]
#[command(name = "bedrock-mantle-sweep")]
#[command(
    about = "Latency / cost / token-usage sweep across open-weight Bedrock models, on \
                   bedrock-mantle and/or bedrock-runtime"
)]
struct Cli {
    /// Number of samples to issue per (model, platform) pair.
    #[arg(long, default_value_t = 3)]
    samples_per_model: usize,
    /// Maximum concurrent in-flight requests across the whole sweep.
    #[arg(long, default_value_t = 8)]
    concurrency: usize,
    /// Output path for the JSON artifact. Stdout if "-".
    #[arg(long, default_value = "bedrock-mantle-sweep.json")]
    output: String,
    /// Comma-separated Mantle-id filter. Defaults to all 31 open-weight models.
    #[arg(long)]
    models: Option<String>,
    /// Which platforms to exercise. Pass `mantle,runtime` to compare both
    /// endpoints for every model in the open-weight intersection. Mantle-only
    /// models are silently skipped on `runtime`.
    #[arg(long, value_delimiter = ',', default_values_t = vec![Platform::Mantle])]
    platforms: Vec<Platform>,
    /// max_tokens for each request. 2048 leaves comfortable headroom over the
    /// 700-word (~900-token) target so models can finish on `EndTurn` rather
    /// than truncating; the prompt itself bounds verbosity. Raise further if
    /// you keep seeing `stop_reasons: MaxTokens` in the artifact.
    #[arg(long, default_value_t = 2048)]
    max_tokens: u32,
    /// Sampling temperature. Default 0.7 matches the AA benchmark posture.
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,
    /// Optional per-request timeout in seconds. 0 disables.
    #[arg(long, default_value_t = 120)]
    request_timeout_secs: u64,
}

#[derive(Serialize, Clone)]
struct SampleRecord {
    platform: Platform,
    sample_idx: usize,
    /// Tokens served from Mantle's implicit prompt cache. Non-zero values
    /// here on `mantle` reveal automatic server-side caching even when the
    /// caller sets `PromptCaching::Off` (which only suppresses our own
    /// CacheBreakpoint emissions, not the upstream's internal cache).
    /// Always `0` on `runtime` since Converse doesn't surface this counter.
    /// To recover the total prompt token count: `input_tokens + cache_read_input_tokens`.
    cache_read_input_tokens: u64,
    ok: bool,
    error: Option<String>,
    ttft_ms: Option<f64>,
    total_ms: f64,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    decode_tps: Option<f64>,
    e2e_tps: Option<f64>,
    stop_reason: Option<String>,
    cost_usd: Option<f64>,
}

#[derive(Serialize)]
struct DistStats {
    n: usize,
    mean: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    min: f64,
    max: f64,
}

impl DistStats {
    fn from_values(values: &[f64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        let mut sorted: Vec<f64> = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        let sum: f64 = sorted.iter().sum();
        Some(Self {
            n,
            mean: sum / n as f64,
            p50: percentile(&sorted, 0.50),
            p95: percentile(&sorted, 0.95),
            p99: percentile(&sorted, 0.99),
            min: sorted[0],
            max: sorted[n - 1],
        })
    }
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    // Nearest-rank (Type 7 / linear interpolation), matches numpy default.
    let h = q * (sorted.len() - 1) as f64;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = h - lo as f64;
        (sorted[hi] - sorted[lo]).mul_add(frac, sorted[lo])
    }
}

#[derive(Serialize)]
struct ModelStats {
    n_samples: usize,
    n_success: usize,
    n_error: usize,
    ttft_ms: Option<DistStats>,
    total_ms: Option<DistStats>,
    decode_tps: Option<DistStats>,
    e2e_tps: Option<DistStats>,
    /// Uncached portion of the prompt as reported by the provider.
    input_tokens: Option<DistStats>,
    /// Tokens served from upstream's implicit prompt cache. Non-zero only on
    /// Mantle today; populated even when the caller set `PromptCaching::Off`
    /// because that flag suppresses our `CacheBreakpoint` markers but not
    /// the server's automatic caching.
    cache_read_input_tokens: Option<DistStats>,
    /// `input_tokens + cache_read_input_tokens` per sample — the
    /// apples-to-apples prompt size you can compare across platforms.
    prompt_tokens_total: Option<DistStats>,
    /// Fraction of samples for which any portion of the prompt was served
    /// from cache (i.e. `cache_read_input_tokens > 0`).
    cache_hit_rate: Option<f64>,
    output_tokens: Option<DistStats>,
    cost_usd_total: f64,
    cost_usd_mean: Option<f64>,
    stop_reasons: std::collections::BTreeMap<String, usize>,
}

#[derive(Serialize)]
struct ModelReport {
    /// Canonical Mantle id used to group samples and look up pricing.
    model_id: String,
    /// Which API surface this report covers (mantle vs. runtime). One model
    /// can yield up to two reports when both platforms are exercised.
    platform: Platform,
    /// Provider-specific id actually invoked. Differs from `model_id` on
    /// bedrock-runtime for the families with `-vN:0` or `-instruct`-dropped
    /// id shapes.
    invoked_id: String,
    pricing_input_per_mtok_usd: Option<f64>,
    pricing_output_per_mtok_usd: Option<f64>,
    samples: Vec<SampleRecord>,
    stats: ModelStats,
}

#[derive(Serialize)]
struct SweepReport {
    started_at: DateTime<Utc>,
    completed_at: DateTime<Utc>,
    wall_clock_ms: f64,
    config: SweepConfig,
    prompt: PromptInfo,
    models: Vec<ModelReport>,
    summary: SweepSummary,
}

#[derive(Serialize)]
struct SweepConfig {
    samples_per_model: usize,
    concurrency: usize,
    max_tokens: u32,
    temperature: f64,
    request_timeout_secs: u64,
    default_region: String,
    openai_gpt5_region: String,
    anthropic_region: String,
    platforms: Vec<Platform>,
    models: Vec<String>,
}

#[derive(Serialize)]
struct PromptInfo {
    system_chars: usize,
    user_template_chars: usize,
    document_chars: usize,
    document_words: usize,
    approx_total_input_tokens: usize,
}

#[derive(Serialize)]
struct SweepSummary {
    total_samples: usize,
    total_successes: usize,
    total_errors: usize,
    total_cost_usd: f64,
    models_with_zero_success: Vec<String>,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let default_region = std::env::var("BEDROCK_MANTLE_DEFAULT_REGION")
        .unwrap_or_else(|_| BedrockMantleProviderConfig::DEFAULT_REGION.to_owned());
    let openai_gpt5_region = std::env::var("BEDROCK_MANTLE_OPENAI_GPT5_REGION")
        .unwrap_or_else(|_| BedrockMantleProviderConfig::DEFAULT_OPENAI_GPT5_REGION.to_owned());
    let anthropic_region = std::env::var("BEDROCK_MANTLE_ANTHROPIC_REGION")
        .unwrap_or_else(|_| BedrockMantleProviderConfig::DEFAULT_ANTHROPIC_REGION.to_owned());

    // SDK config is shared between both providers — both use the AWS default
    // credential chain, both default to the same region.
    let sdk_config = aws_config::defaults(BehaviorVersion::latest())
        .region(aws_config::Region::new(default_region.clone()))
        .load()
        .await;

    let auth = std::env::var("BEDROCK_MANTLE_API_KEY").map_or_else(
        |_| {
            eprintln!("mantle auth: AWS SigV4 (default credential chain)");
            let credentials_provider = sdk_config.credentials_provider().expect(
                "AWS credential chain returned no credentials_provider; set \
                 BEDROCK_MANTLE_API_KEY or run under aws-vault",
            );
            BedrockMantleAuth::Sigv4 {
                credentials_provider,
            }
        },
        |key| {
            eprintln!("mantle auth: Bedrock API key (Bearer)");
            BedrockMantleAuth::ApiKey(key)
        },
    );

    let http_client = Arc::new(build_http_client().expect("build reqwest client"));
    let mantle_provider: Arc<dyn LanguageModelProvider> = Arc::new(BedrockMantleProvider::new(
        BedrockMantleProviderDeps {
            client: Arc::clone(&http_client),
            auth,
        },
        BedrockMantleProviderConfig {
            default_region: default_region.clone(),
            openai_gpt5_region: openai_gpt5_region.clone(),
            anthropic_region: anthropic_region.clone(),
            retry_config: RetryConfig::default(),
        },
    ));

    // Only build the runtime provider when needed — it spins up two AWS SDK
    // clients (`aws-sdk-bedrockruntime` for inference, `aws-sdk-bedrock` for
    // the control-plane catalog) and we don't want that overhead on
    // mantle-only sweeps.
    let want_runtime = cli.platforms.iter().any(|p| matches!(p, Platform::Runtime));
    let runtime_provider: Option<Arc<dyn LanguageModelProvider>> = if want_runtime {
        eprintln!("runtime auth: AWS SigV4 (default credential chain, region={default_region})");
        Some(Arc::new(BedrockProvider::new(
            BedrockProviderDeps {
                runtime_client: aws_sdk_bedrockruntime::Client::new(&sdk_config),
                control_client: aws_sdk_bedrock::Client::new(&sdk_config),
            },
            BedrockProviderConfig {
                region: Some(default_region.clone()),
                retry_config: RetryConfig::default(),
            },
        )))
    } else {
        None
    };

    let selected_models: Vec<String> = cli.models.as_deref().map_or_else(
        || {
            OPEN_WEIGHT_MODELS
                .iter()
                .map(|e| e.mantle_id.to_owned())
                .collect()
        },
        |s| {
            s.split(',')
                .map(|t| t.trim().to_owned())
                .filter(|t| !t.is_empty())
                .collect()
        },
    );

    let user_prompt = USER_INSTRUCTION.replace("{document}", SAMPLE_DOCUMENT);
    let prompt_info = PromptInfo {
        system_chars: SYSTEM_PROMPT.len(),
        user_template_chars: USER_INSTRUCTION.len(),
        document_chars: SAMPLE_DOCUMENT.len(),
        document_words: SAMPLE_DOCUMENT.split_whitespace().count(),
        // Rough Latin-script estimate: ~4 chars/token for GPT-style BPE.
        approx_total_input_tokens: (SYSTEM_PROMPT.len() + user_prompt.len()) / 4,
    };

    let platforms: Vec<Platform> = {
        let mut seen = std::collections::HashSet::new();
        cli.platforms
            .iter()
            .copied()
            .filter(|p| seen.insert(*p))
            .collect()
    };
    eprintln!(
        "sweep: models={} platforms=[{}] samples/(model,platform)={} concurrency={} max_tokens={} \
         temp={} timeout={}s",
        selected_models.len(),
        platforms
            .iter()
            .map(|p| p.as_str())
            .collect::<Vec<_>>()
            .join(","),
        cli.samples_per_model,
        cli.concurrency,
        cli.max_tokens,
        cli.temperature,
        cli.request_timeout_secs,
    );
    eprintln!(
        "prompt: doc={} words ({} chars) ≈ {} input tokens (system + user_template + doc)",
        prompt_info.document_words,
        prompt_info.document_chars,
        prompt_info.approx_total_input_tokens,
    );

    let messages = Arc::new(vec![
        Message::system(SYSTEM_PROMPT),
        Message::user(&user_prompt),
    ]);

    // Caching off — comparable cold-call numbers across providers / runs.
    let config = Arc::new(LanguageModelConfig {
        temperature: Some(cli.temperature),
        max_tokens: Some(cli.max_tokens),
        prompt_caching: PromptCaching::Off,
        ..Default::default()
    });

    let request_timeout = (cli.request_timeout_secs > 0)
        .then_some(std::time::Duration::from_secs(cli.request_timeout_secs));

    // Build the work-list. For each (platform, mantle_id) pair we look up the
    // provider-specific invocation id from the model table. Runtime-only-missing
    // models drop out cleanly here rather than failing at request time.
    let entry_by_mantle: std::collections::HashMap<&str, &ModelEntry> = OPEN_WEIGHT_MODELS
        .iter()
        .map(|e| (e.mantle_id, e))
        .collect();
    let mut work: Vec<WorkItem> =
        Vec::with_capacity(selected_models.len() * platforms.len() * cli.samples_per_model);
    let mut runtime_only_skips: Vec<String> = Vec::new();
    let mut unknown_models: Vec<String> = Vec::new();
    for mantle_id in &selected_models {
        let Some(entry) = entry_by_mantle.get(mantle_id.as_str()) else {
            unknown_models.push(mantle_id.clone());
            continue;
        };
        for plat in &platforms {
            let invoked_id: String = match plat {
                Platform::Mantle => entry.mantle_id.to_owned(),
                Platform::Runtime => {
                    if let Some(id) = entry.runtime_id {
                        id.to_owned()
                    } else {
                        runtime_only_skips.push(mantle_id.clone());
                        continue;
                    }
                }
            };
            for i in 0..cli.samples_per_model {
                work.push(WorkItem {
                    platform: *plat,
                    mantle_id: mantle_id.clone(),
                    invoked_id: invoked_id.clone(),
                    sample_idx: i,
                });
            }
        }
    }
    if !unknown_models.is_empty() {
        eprintln!(
            "warning: {} requested model(s) not in OPEN_WEIGHT_MODELS table, skipping: {}",
            unknown_models.len(),
            unknown_models.join(", "),
        );
    }
    if !runtime_only_skips.is_empty() {
        let unique: std::collections::BTreeSet<_> = runtime_only_skips.iter().collect();
        eprintln!(
            "note: {} mantle-only model(s) skipped on `runtime` platform: {}",
            unique.len(),
            unique
                .iter()
                .copied()
                .cloned()
                .collect::<Vec<_>>()
                .join(", "),
        );
    }

    let started_at = Utc::now();
    let started_instant = Instant::now();

    let total_work = work.len();
    eprintln!("dispatching {total_work} requests...");

    let progress = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let results: Vec<((String, Platform), SampleRecord)> =
        futures::stream::iter(work.into_iter().map(|item| {
            let mantle_provider = Arc::clone(&mantle_provider);
            let runtime_provider = runtime_provider.as_ref().map(Arc::clone);
            let messages = Arc::clone(&messages);
            let config = Arc::clone(&config);
            let progress = Arc::clone(&progress);
            async move {
                let provider: &dyn LanguageModelProvider = match item.platform {
                    Platform::Mantle => mantle_provider.as_ref(),
                    Platform::Runtime => runtime_provider
                        .as_deref()
                        .expect("runtime provider must exist when Runtime platform is in work"),
                };
                let sample = run_sample(SampleArgs {
                    provider,
                    platform: item.platform,
                    pricing_id: &item.mantle_id,
                    invoked_id: &item.invoked_id,
                    sample_idx: item.sample_idx,
                    messages: &messages,
                    config: &config,
                    timeout: request_timeout,
                })
                .await;
                let done = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                let detail = sample.error.as_deref().map_or_else(
                    || {
                        let ttft = sample
                            .ttft_ms
                            .map_or_else(|| "n/a".to_owned(), |v| format!("{v:.0}"));
                        format!(
                            "ttft={}ms total={:.0}ms out_tok={:?}",
                            ttft, sample.total_ms, sample.output_tokens,
                        )
                    },
                    |e| format!("err={e}"),
                );
                eprintln!(
                    "  [{done:>4}/{total_work}] {platform:<8} {model_id:<48} ok={ok} {detail}",
                    platform = item.platform.as_str(),
                    model_id = item.mantle_id,
                    ok = sample.ok,
                );
                ((item.mantle_id, item.platform), sample)
            }
        }))
        .buffer_unordered(cli.concurrency)
        .collect()
        .await;

    let wall_clock_ms = started_instant.elapsed().as_secs_f64() * 1000.0;
    let completed_at = Utc::now();

    // Group samples by (mantle_id, platform).
    let mut by_pair: std::collections::BTreeMap<(String, Platform), Vec<SampleRecord>> =
        std::collections::BTreeMap::new();
    for (key, sample) in results {
        by_pair.entry(key).or_default().push(sample);
    }
    for samples in by_pair.values_mut() {
        samples.sort_by_key(|s| s.sample_idx);
    }

    let mut model_reports = Vec::with_capacity(selected_models.len() * platforms.len());
    let mut zero_success: Vec<String> = Vec::new();
    let mut total_samples = 0usize;
    let mut total_successes = 0usize;
    let mut total_errors = 0usize;
    let mut total_cost_usd = 0.0_f64;

    for mantle_id in &selected_models {
        let entry = entry_by_mantle.get(mantle_id.as_str()).copied();
        for plat in &platforms {
            let invoked_id: String = match (plat, entry) {
                (Platform::Mantle, _) => mantle_id.clone(),
                (Platform::Runtime, Some(e)) => match e.runtime_id {
                    Some(id) => id.to_owned(),
                    None => continue,
                },
                (Platform::Runtime, None) => continue,
            };

            let samples = by_pair
                .remove(&(mantle_id.clone(), *plat))
                .unwrap_or_default();
            let (pin, pout) =
                price_for(mantle_id).map_or((None, None), |(i, o)| (Some(i), Some(o)));
            let n_samples = samples.len();
            let oks: Vec<&SampleRecord> = samples.iter().filter(|s| s.ok).collect();
            let n_success = oks.len();
            let n_error = n_samples - n_success;
            total_samples += n_samples;
            total_successes += n_success;
            total_errors += n_error;

            let cost_total: f64 = samples.iter().filter_map(|s| s.cost_usd).sum();
            total_cost_usd += cost_total;
            let cost_mean = (n_success > 0).then(|| cost_total / n_success as f64);

            let ttft_values: Vec<f64> = oks.iter().filter_map(|s| s.ttft_ms).collect();
            let total_ms_values: Vec<f64> = oks.iter().map(|s| s.total_ms).collect();
            let decode_tps_values: Vec<f64> = oks.iter().filter_map(|s| s.decode_tps).collect();
            let e2e_tps_values: Vec<f64> = oks.iter().filter_map(|s| s.e2e_tps).collect();
            let input_tok_values: Vec<f64> = oks
                .iter()
                .filter_map(|s| s.input_tokens.map(|t| t as f64))
                .collect();
            let cache_read_values: Vec<f64> = oks
                .iter()
                .map(|s| s.cache_read_input_tokens as f64)
                .collect();
            let prompt_total_values: Vec<f64> = oks
                .iter()
                .filter_map(|s| {
                    s.input_tokens
                        .map(|t| t as f64 + s.cache_read_input_tokens as f64)
                })
                .collect();
            let n_with_cache = oks.iter().filter(|s| s.cache_read_input_tokens > 0).count();
            let cache_hit_rate = (n_success > 0).then(|| n_with_cache as f64 / n_success as f64);
            let output_tok_values: Vec<f64> = oks
                .iter()
                .filter_map(|s| s.output_tokens.map(|t| t as f64))
                .collect();

            let mut stop_reasons: std::collections::BTreeMap<String, usize> =
                std::collections::BTreeMap::new();
            for s in &samples {
                if let Some(sr) = &s.stop_reason {
                    *stop_reasons.entry(sr.clone()).or_default() += 1;
                }
            }

            let stats = ModelStats {
                n_samples,
                n_success,
                n_error,
                ttft_ms: DistStats::from_values(&ttft_values),
                total_ms: DistStats::from_values(&total_ms_values),
                decode_tps: DistStats::from_values(&decode_tps_values),
                e2e_tps: DistStats::from_values(&e2e_tps_values),
                input_tokens: DistStats::from_values(&input_tok_values),
                cache_read_input_tokens: DistStats::from_values(&cache_read_values),
                prompt_tokens_total: DistStats::from_values(&prompt_total_values),
                cache_hit_rate,
                output_tokens: DistStats::from_values(&output_tok_values),
                cost_usd_total: cost_total,
                cost_usd_mean: cost_mean,
                stop_reasons,
            };

            if n_samples > 0 && n_success == 0 {
                zero_success.push(format!("{}/{}", plat.as_str(), mantle_id));
            }

            model_reports.push(ModelReport {
                model_id: mantle_id.clone(),
                platform: *plat,
                invoked_id,
                pricing_input_per_mtok_usd: pin,
                pricing_output_per_mtok_usd: pout,
                samples,
                stats,
            });
        }
    }

    let report = SweepReport {
        started_at,
        completed_at,
        wall_clock_ms,
        config: SweepConfig {
            samples_per_model: cli.samples_per_model,
            concurrency: cli.concurrency,
            max_tokens: cli.max_tokens,
            temperature: cli.temperature,
            request_timeout_secs: cli.request_timeout_secs,
            default_region,
            openai_gpt5_region,
            anthropic_region,
            platforms: platforms.clone(),
            models: selected_models,
        },
        prompt: prompt_info,
        models: model_reports,
        summary: SweepSummary {
            total_samples,
            total_successes,
            total_errors,
            total_cost_usd,
            models_with_zero_success: zero_success,
        },
    };

    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    if cli.output == "-" {
        println!("{json}");
    } else {
        std::fs::write(&cli.output, &json).expect("write JSON artifact");
        eprintln!("\nwrote {} bytes to {}", json.len(), cli.output);
    }

    print_console_summary(&report);
}

struct SampleArgs<'a> {
    provider: &'a dyn LanguageModelProvider,
    platform: Platform,
    /// Mantle id used for pricing lookup. Canonical key regardless of which
    /// platform's wire id we actually sent.
    pricing_id: &'a str,
    /// Provider-specific id actually used in the request (differs on
    /// bedrock-runtime for some Qwen3 / gpt-oss families).
    invoked_id: &'a str,
    sample_idx: usize,
    messages: &'a [Message],
    config: &'a LanguageModelConfig,
    timeout: Option<std::time::Duration>,
}

struct WorkItem {
    platform: Platform,
    mantle_id: String,
    invoked_id: String,
    sample_idx: usize,
}

async fn run_sample(args: SampleArgs<'_>) -> SampleRecord {
    let SampleArgs {
        provider,
        platform,
        pricing_id,
        invoked_id,
        sample_idx,
        messages,
        config,
        timeout,
    } = args;
    let model = ModelId::new(invoked_id);
    let req = GenerateRequest {
        model: &model,
        messages,
        config,
    };

    let start = Instant::now();
    let fut = provider.generate_stream(req);
    let stream = match timeout {
        Some(t) => match tokio::time::timeout(t, fut).await {
            Ok(r) => r,
            Err(_) => {
                return error_sample(ErrorSampleArgs {
                    platform,
                    sample_idx,
                    start,
                    err: "request setup timeout".to_owned(),
                    partial_usage: None,
                    _pricing_id: pricing_id,
                });
            }
        },
        None => fut.await,
    };
    let mut stream = match stream {
        Ok(s) => s,
        Err(e) => {
            return error_sample(ErrorSampleArgs {
                platform,
                sample_idx,
                start,
                err: format!("{e}"),
                partial_usage: None,
                _pricing_id: pricing_id,
            });
        }
    };

    let mut first_observable_at: Option<Instant> = None;
    let mut last_usage = None;
    let mut last_model: Option<String> = None;
    let mut stop_reason: Option<StopReason> = None;
    let mut had_error: Option<String> = None;
    let mut produced_any_observable = false;

    loop {
        let next = match timeout {
            Some(t) => tokio::time::timeout(t, stream.next())
                .await
                .unwrap_or_else(|_| {
                    had_error = Some("stream chunk timeout".to_owned());
                    None
                }),
            None => stream.next().await,
        };
        match next {
            None => break,
            Some(Err(e)) => {
                had_error = Some(format!("{e}"));
                break;
            }
            Some(Ok(delta)) => {
                // TTFT is "time to first observable model output" — count
                // reasoning deltas (`thinking`) as observable so Converse's
                // reasoning-model TTFT is comparable to Mantle's Chat
                // Completions surface, which streams reasoning as content.
                let observable_now = !delta.content.is_empty()
                    || delta.thinking.as_deref().is_some_and(|t| !t.is_empty());
                if observable_now && first_observable_at.is_none() {
                    first_observable_at = Some(Instant::now());
                    produced_any_observable = true;
                }
                if delta.usage.is_some() {
                    last_usage = delta.usage;
                }
                if delta.model.is_some() && last_model.is_none() {
                    last_model.clone_from(&delta.model);
                }
                // Wire-level stop_reason is now plumbed through every adapter
                // (OpenAI finish_reason, Anthropic message_delta.delta.stop_reason,
                // Converse MessageStop, Mantle Responses response.completed/
                // response.incomplete). Take the first one we see — every
                // adapter emits it exactly once per stream.
                if stop_reason.is_none() && delta.stop_reason.is_some() {
                    stop_reason.clone_from(&delta.stop_reason);
                }
                if delta.is_final {
                    break;
                }
            }
        }
    }

    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = first_observable_at.map(|t| (t - start).as_secs_f64() * 1000.0);

    if let Some(e) = had_error {
        // Even on stream error, capture whatever usage / partial timing we got.
        let (input_tokens, output_tokens, cache_read) = last_usage.map_or((None, None, 0), |u| {
            (
                Some(u.input_tokens),
                Some(u.output_tokens),
                u.cache_read_input_tokens,
            )
        });
        return SampleRecord {
            platform,
            sample_idx,
            cache_read_input_tokens: cache_read,
            ok: false,
            error: Some(e),
            ttft_ms,
            total_ms,
            input_tokens,
            output_tokens,
            decode_tps: None,
            e2e_tps: None,
            stop_reason: None,
            cost_usd: cost_for(pricing_id, input_tokens, cache_read, output_tokens),
        };
    }

    // A stream that finished without any content or thinking delta is only a
    // genuine flake if the wire didn't tell us WHY it stopped. Reasoning
    // models can legitimately burn their entire `max_tokens` budget on hidden
    // CoT and emit zero content (the gpt-5.5 / `max_tokens=32` case from the
    // audit's Mantle Responses validation). A `MaxTokens` (or `ContentFilter`)
    // stop_reason on that stream is the model's own report that it
    // intentionally truncated — surface it as `ok=true` with empty content,
    // not an error.
    if !produced_any_observable && stop_reason.is_none() {
        return error_sample(ErrorSampleArgs {
            platform,
            sample_idx,
            start,
            err: "stream completed without content or thinking delta".to_owned(),
            partial_usage: last_usage.map(|u| (u.input_tokens, u.output_tokens)),
            _pricing_id: pricing_id,
        });
    }

    let usage = last_usage;
    let input_tokens = usage.map(|u| u.input_tokens);
    let output_tokens = usage.map(|u| u.output_tokens);
    let cache_read = usage.map_or(0, |u| u.cache_read_input_tokens);

    // If the stream never surfaced an explicit stop_reason (older OpenAI-
    // compatible servers that don't emit finish_reason — not observed on
    // Mantle or Converse in practice, but defensive), fall back to the
    // output-token-vs-budget heuristic so the artifact still has *something*
    // to report.
    if stop_reason.is_none()
        && let (Some(out), Some(max)) = (output_tokens, config.max_tokens)
    {
        stop_reason = Some(if out >= u64::from(max) {
            StopReason::MaxTokens
        } else {
            StopReason::EndTurn
        });
    }

    let decode_tps = match (output_tokens, ttft_ms) {
        (Some(out), Some(ttft)) if total_ms > ttft && out > 0 => {
            Some(out as f64 / ((total_ms - ttft) / 1000.0))
        }
        _ => None,
    };
    let e2e_tps =
        output_tokens.and_then(|out| (total_ms > 0.0).then(|| out as f64 / (total_ms / 1000.0)));

    SampleRecord {
        platform,
        sample_idx,
        cache_read_input_tokens: cache_read,
        ok: true,
        error: None,
        ttft_ms,
        total_ms,
        input_tokens,
        output_tokens,
        decode_tps,
        e2e_tps,
        stop_reason: stop_reason.map(|r| format!("{r:?}")),
        cost_usd: cost_for(pricing_id, input_tokens, cache_read, output_tokens),
    }
}

/// Discount Bedrock applies to prompt-cache reads on Mantle. The Mantle wire
/// surfaces `cached_tokens` in OpenAI-style usage; AWS prices those at 10% of
/// the model's input rate (matching the Anthropic Bedrock convention — see
/// docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html).
/// Runtime/Converse doesn't expose cached_tokens at all, so the cache_read
/// argument is `0` on that platform and this scaling is a no-op.
const CACHE_READ_PRICE_FRACTION: f64 = 0.10;

fn cost_for(
    model_id: &str,
    input_tokens: Option<u64>,
    cache_read_input_tokens: u64,
    output_tokens: Option<u64>,
) -> Option<f64> {
    let (pin, pout) = price_for(model_id)?;
    let inp = input_tokens? as f64;
    let cache_read = cache_read_input_tokens as f64;
    let out = output_tokens? as f64;
    let uncached_cost = inp / 1_000_000.0 * pin;
    let cache_cost = cache_read / 1_000_000.0 * pin * CACHE_READ_PRICE_FRACTION;
    let output_cost = out / 1_000_000.0 * pout;
    Some(uncached_cost + cache_cost + output_cost)
}

struct ErrorSampleArgs<'a> {
    platform: Platform,
    sample_idx: usize,
    start: Instant,
    err: String,
    partial_usage: Option<(u64, u64)>,
    _pricing_id: &'a str,
}

fn error_sample(args: ErrorSampleArgs<'_>) -> SampleRecord {
    let ErrorSampleArgs {
        platform,
        sample_idx,
        start,
        err,
        partial_usage,
        _pricing_id,
    } = args;
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let (input_tokens, output_tokens) =
        partial_usage.map_or((None, None), |(i, o)| (Some(i), Some(o)));
    SampleRecord {
        platform,
        sample_idx,
        cache_read_input_tokens: 0,
        ok: false,
        error: Some(err),
        ttft_ms: None,
        total_ms,
        input_tokens,
        output_tokens,
        decode_tps: None,
        e2e_tps: None,
        stop_reason: None,
        cost_usd: None,
    }
}

fn print_console_summary(report: &SweepReport) {
    eprintln!("\n=== sweep complete ===");
    eprintln!(
        "wall: {:.1}s  samples: {} ok / {} err / {} total  cost: ${:.4}",
        report.wall_clock_ms / 1000.0,
        report.summary.total_successes,
        report.summary.total_errors,
        report.summary.total_samples,
        report.summary.total_cost_usd,
    );
    if !report.summary.models_with_zero_success.is_empty() {
        eprintln!(
            "models with zero successes ({}): {}",
            report.summary.models_with_zero_success.len(),
            report.summary.models_with_zero_success.join(", "),
        );
    }
    eprintln!();
    eprintln!(
        "{:<48} {:<8} {:>4} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "model", "platform", "ok", "ttft p50", "total p50", "decode tps", "out tok", "$/sample",
    );
    let dash = || "—".to_owned();
    for m in &report.models {
        let ttft = m
            .stats
            .ttft_ms
            .as_ref()
            .map_or_else(dash, |d| format!("{:.0}", d.p50));
        let total = m
            .stats
            .total_ms
            .as_ref()
            .map_or_else(dash, |d| format!("{:.0}", d.p50));
        let decode = m
            .stats
            .decode_tps
            .as_ref()
            .map_or_else(dash, |d| format!("{:.0}", d.mean));
        let out_tok = m
            .stats
            .output_tokens
            .as_ref()
            .map_or_else(dash, |d| format!("{:.0}", d.mean));
        let dollar = m
            .stats
            .cost_usd_mean
            .map_or_else(dash, |v| format!("{v:.5}"));
        eprintln!(
            "{:<48} {:<8} {:>4} {:>8} {:>10} {:>10} {:>10} {:>10}",
            m.model_id,
            m.platform.as_str(),
            m.stats.n_success,
            ttft,
            total,
            decode,
            out_tok,
            dollar,
        );
    }
}
