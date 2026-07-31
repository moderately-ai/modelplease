// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `DummyLM` — a test mock for the [`LanguageModelProvider`] trait.
//!
//! Returns pre-configured raw strings. Has no knowledge of adapters,
//! delimiters, or prediction formats — it is a pure language model mock.

use std::{collections::HashMap, pin::Pin, sync::Mutex, time::Duration};

use async_trait::async_trait;
use enumset::enum_set;
use futures::stream::{self, Stream, StreamExt};

use crate::{
    capabilities::{
        ModelCapabilities, ReasoningCapability, ReasoningMode, ReasoningParamConflicts,
    },
    config::ReasoningEffort,
    error::LanguageModelError,
    identifiers::ModelId,
    message::Message,
    provider::{ChatModelInfo, GenerateRequest, LanguageModelProvider, ResponseFormatKind},
    response::{LanguageModelResponse, StreamDelta, Usage},
};

/// Synthetic reasoning capability for the Dummy provider's `test` model.
///
/// Exposes `Adaptive` + `Manual` modes with the full enum so capability
/// validation tests can exercise every fail / pass path without hitting
/// a real provider. The conflicts table mirrors Anthropic's so the
/// integration tests cover the temperature-stripping logic too.
const fn dummy_reasoning_capability() -> ReasoningCapability {
    ReasoningCapability {
        supported_modes: enum_set!(ReasoningMode::Adaptive | ReasoningMode::Manual),
        supported_efforts: enum_set!(
            ReasoningEffort::None
                | ReasoningEffort::Low
                | ReasoningEffort::Medium
                | ReasoningEffort::High
                | ReasoningEffort::XHigh
                | ReasoningEffort::Max
        ),
        manual_budget_range: Some(1024..=32_000),
        conflicts: ReasoningParamConflicts {
            temperature_forbidden: true,
            top_k_forbidden: true,
            top_p_allowed_range: Some(0.95..=1.0),
        },
        sampling_params_removed: false,
    }
}

/// Internal mode for the [`DummyLM`].
#[derive(Debug, Clone)]
enum DummyMode {
    /// Returns answers in order. Errors when exhausted.
    Sequential { answers: Vec<String> },
    /// Matches last user message content against keys, returns corresponding answer.
    Lookup { entries: HashMap<String, String> },
    /// Token-by-token scripted streaming. Each outer `Vec<String>` is
    /// one full generation; each inner `String` is one
    /// `StreamDelta.content` chunk. Used by streaming-codec tests that
    /// need deterministic chunk boundaries to drive `ChatStreamParser`
    /// through provider-realistic split points.
    ///
    /// `chunk_delay`, when set, sleeps for that duration BEFORE each
    /// delta is yielded. Required by cancellation tests that need a
    /// yield point between deltas so a cancel signal can fire
    /// mid-stream (without a delay the stream emits all deltas
    /// synchronously and the consumer never has a chance to observe
    /// cancellation between them). `None` preserves the legacy
    /// synchronous behavior.
    Scripted {
        generations: Vec<Vec<String>>,
        chunk_delay: Option<Duration>,
        emit_usage: bool,
    },
}

/// A test mock that implements [`LanguageModelProvider`].
///
/// Operates in two modes:
/// - **Sequential**: returns the next answer from a list on each call
/// - **Lookup**: matches the last user message against keys and returns the value
///
/// # Examples
///
/// ```rust
/// use modelplease::DummyLM;
///
/// // Sequential: returns "Paris" on first call, "Berlin" on second
/// let lm = DummyLM::sequential(vec!["Paris".into(), "Berlin".into()]);
///
/// // Lookup: returns "4" when the last user message contains "2+2"
/// use std::collections::HashMap;
/// let lm = DummyLM::lookup(HashMap::from([("2+2".into(), "4".into())]));
/// ```
#[derive(Debug)]
pub struct DummyLM {
    mode: DummyMode,
    index: Mutex<usize>,
}

impl DummyLM {
    /// Create a `DummyLM` that returns answers in order.
    ///
    /// Each call to `generate()` returns the next answer. Returns
    /// [`LanguageModelError::EmptyResponse`] when all answers are exhausted.
    #[must_use]
    pub const fn sequential(answers: Vec<String>) -> Self {
        Self {
            mode: DummyMode::Sequential { answers },
            index: Mutex::new(0),
        }
    }

    /// Create a `DummyLM` that matches the last user message against keys.
    ///
    /// The last message with `Role::User` is checked — if its text content
    /// contains any key, the corresponding value is returned. Returns
    /// [`LanguageModelError::Provider`] if no key matches.
    #[must_use]
    pub const fn lookup(entries: HashMap<String, String>) -> Self {
        Self {
            mode: DummyMode::Lookup { entries },
            index: Mutex::new(0),
        }
    }

    /// Create a `DummyLM` that emits scripted token-by-token streams.
    ///
    /// Each outer `Vec<String>` is one full generation; the inner
    /// `String`s are the per-chunk `StreamDelta.content` values for
    /// that generation. The internal index advances per
    /// `generate_stream` call (mirroring [`Self::sequential`]); the
    /// last delta of each generation carries `is_final = true` and a
    /// `Usage` whose `output_tokens` is the total character count of
    /// the generation, so streaming tests can assert usage
    /// propagation.
    ///
    /// `generate` (non-streaming) on a scripted instance returns the
    /// joined chunks of the current generation as one
    /// `LanguageModelResponse`, so the same fixture can drive both
    /// the buffered and streaming code paths.
    ///
    /// Deltas emit synchronously back-to-back. For tests that need a
    /// yield point between deltas (e.g. cancellation-mid-stream)
    /// use [`Self::scripted_stream_with_delay`] instead.
    #[must_use]
    pub const fn scripted_stream(generations: Vec<Vec<String>>) -> Self {
        Self {
            mode: DummyMode::Scripted {
                generations,
                chunk_delay: None,
                emit_usage: true,
            },
            index: Mutex::new(0),
        }
    }

    /// Same as [`Self::scripted_stream`] but the final delta carries
    /// `usage = None`, modelling provider streams that don't surface
    /// usage at end-of-stream (observed on Mantle Chat Completions for
    /// `glm-4.7-flash` despite `stream_options.include_usage` being set).
    /// Used to exercise the streaming consumer's `usageSource = "missing"`
    /// fallback path.
    #[must_use]
    pub const fn scripted_stream_without_usage(generations: Vec<Vec<String>>) -> Self {
        Self {
            mode: DummyMode::Scripted {
                generations,
                chunk_delay: None,
                emit_usage: false,
            },
            index: Mutex::new(0),
        }
    }

    /// Same as [`Self::scripted_stream`] but sleeps `chunk_delay`
    /// before each delta is yielded. Used by streaming tests
    /// that need to interleave a control action (cancellation,
    /// shutdown) between deltas — without the delay the stream emits
    /// all deltas synchronously in one tokio task tick and the
    /// control action never has a chance to fire mid-stream.
    ///
    /// Recommended delay: 30–100 ms per chunk. Smaller risks the
    /// test runner not actually yielding; larger needlessly slows
    /// the test.
    #[must_use]
    pub const fn scripted_stream_with_delay(
        generations: Vec<Vec<String>>,
        chunk_delay: Duration,
    ) -> Self {
        Self {
            mode: DummyMode::Scripted {
                generations,
                chunk_delay: Some(chunk_delay),
                emit_usage: true,
            },
            index: Mutex::new(0),
        }
    }
}

impl DummyLM {
    /// Pull the next canned response (sequential) or the matching lookup
    /// entry. `request.model` is ignored — DummyLM serves any model name
    /// in its catalog.
    fn answer(&self, messages: &[Message]) -> Result<LanguageModelResponse, LanguageModelError> {
        let content = match &self.mode {
            DummyMode::Sequential { answers } => {
                let mut idx = self
                    .index
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if *idx >= answers.len() {
                    return Err(LanguageModelError::EmptyResponse);
                }
                let answer = answers[*idx].clone();
                *idx += 1;
                answer
            }
            DummyMode::Lookup { entries } => {
                let last_user = messages
                    .iter()
                    .rev()
                    .find(|m| m.role == crate::message::Role::User)
                    .map(Message::text)
                    .unwrap_or_default();

                entries
                    .iter()
                    .find(|(key, _)| last_user.contains(key.as_str()))
                    .map(|(_, value)| value)
                    .cloned()
                    .ok_or_else(|| {
                        LanguageModelError::provider(format!(
                            "DummyLM lookup: no key matched message: {last_user}"
                        ))
                    })?
            }
            DummyMode::Scripted {
                generations,
                chunk_delay: _,
                emit_usage: _,
            } => {
                // `chunk_delay` only affects the streaming path; the
                // buffered `answer()` returns the concatenated chunks
                // immediately regardless.
                let chunks = {
                    let mut idx = self
                        .index
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if *idx >= generations.len() {
                        return Err(LanguageModelError::EmptyResponse);
                    }
                    let chunks = generations[*idx].clone();
                    *idx += 1;
                    chunks
                };
                chunks.concat()
            }
        };

        Ok(LanguageModelResponse {
            content,
            thinking: None,
            usage: Some(Usage::default()),
            model: Some("dummy".to_owned()),
            stop_reason: Some(crate::StopReason::EndTurn),
        })
    }
}

#[async_trait]
impl LanguageModelProvider for DummyLM {
    fn name(&self) -> &'static str {
        "dummy"
    }

    /// Static catalog — DummyLM is a pure test fixture; no upstream
    /// to query. Reports a single `dummy/test` entry with full
    /// capability flags so consumers can route to it without any
    /// special-case handling.
    async fn list_models(&self) -> Result<Vec<ChatModelInfo>, LanguageModelError> {
        Ok(vec![ChatModelInfo {
            id: ModelId::new("test"),
            display_name: Some("Dummy Test Model".to_owned()),
            context_window: None,
            supports_streaming: true,
            supported_response_formats: vec![
                ResponseFormatKind::Text,
                ResponseFormatKind::JsonObject,
                ResponseFormatKind::JsonSchema,
            ],
            media_support: std::collections::BTreeMap::new(),
            reasoning: Some(dummy_reasoning_capability()),
        }])
    }

    /// DummyLM is text-only — no media accepted. Returning `Some(empty)`
    /// rather than `None` lets the default `validate_request` impl
    /// short-circuit on `ModalityUnsupported` instead of `UnknownModel`.
    /// Reasoning capability is fully populated so capability-validation
    /// tests can exercise every pass/fail path.
    fn capabilities(&self, model: &ModelId) -> Option<ModelCapabilities> {
        Some(ModelCapabilities {
            model_id: model.as_str().to_owned(),
            media_support: std::collections::BTreeMap::new(),
            reasoning: Some(dummy_reasoning_capability()),
            // Latency-optimized inference is a Bedrock-only tier.
            latency_optimized_supported: false,
            // Extended-TTL prompt caching is a Bedrock/Anthropic tier.
            extended_cache_ttl_supported: false,
        })
    }

    async fn generate(
        &self,
        request: GenerateRequest<'_>,
    ) -> Result<LanguageModelResponse, LanguageModelError> {
        self.answer(request.messages)
    }

    /// Emit one [`StreamDelta`] per scripted chunk for
    /// scripted mode; fall back to a single-item stream
    /// wrapping the buffered answer for the other modes (which have no
    /// scripted chunk boundaries).
    ///
    /// On scripted generations: each inner chunk becomes one
    /// `StreamDelta` with `is_final = false`, EXCEPT the last chunk
    /// of the generation which carries `is_final = true` and a
    /// non-`None` `Usage` whose `output_tokens` is the total character
    /// count of the generation. This shape lets streaming
    /// tests assert both the per-chunk event sequence and the usage
    /// roll-up.
    async fn generate_stream(
        &self,
        request: GenerateRequest<'_>,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamDelta, LanguageModelError>> + Send>>,
        LanguageModelError,
    > {
        if let DummyMode::Scripted {
            generations,
            chunk_delay,
            emit_usage,
        } = &self.mode
        {
            let chunk_delay = *chunk_delay;
            let emit_usage = *emit_usage;
            let chunks = {
                let mut idx = self
                    .index
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if *idx >= generations.len() {
                    return Err(LanguageModelError::EmptyResponse);
                }
                let chunks = generations[*idx].clone();
                *idx += 1;
                chunks
            };

            let total_chars: usize = chunks.iter().map(String::len).sum();
            // usize → u64 is always lossless on 64-bit targets; on
            // 32-bit it saturates rather than wraps. Test fixtures
            // never approach either limit.
            let output_tokens = u64::try_from(total_chars).unwrap_or(u64::MAX);
            let last_idx = chunks.len().saturating_sub(1);

            let deltas: Vec<Result<StreamDelta, LanguageModelError>> = chunks
                .into_iter()
                .enumerate()
                .map(|(i, content)| {
                    let is_final = i == last_idx;
                    Ok(StreamDelta {
                        content,
                        thinking: None,
                        usage: if is_final && emit_usage {
                            Some(Usage {
                                input_tokens: 0,
                                output_tokens,
                                ..Usage::default()
                            })
                        } else {
                            None
                        },
                        model: Some("dummy".to_owned()),
                        stop_reason: None,
                        is_final,
                    })
                })
                .collect();
            let base = stream::iter(deltas);
            return if let Some(delay) = chunk_delay {
                // `.then(|d| async { sleep().await; d })` introduces
                // the yield point each delta needs so cancellation
                // signals can fire BETWEEN chunks.
                Ok(Box::pin(base.then(move |delta| async move {
                    tokio::time::sleep(delay).await;
                    delta
                })))
            } else {
                Ok(Box::pin(base))
            };
        }

        // Fallback: single-item stream wrapping the buffered answer.
        let response = self.answer(request.messages)?;
        let delta = StreamDelta {
            content: response.content,
            thinking: response.thinking,
            usage: response.usage,
            model: response.model,
            stop_reason: response.stop_reason,
            is_final: true,
        };
        Ok(Box::pin(stream::iter([Ok(delta)])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LanguageModelConfig;

    fn req<'a>(
        model: &'a ModelId,
        messages: &'a [Message],
        config: &'a LanguageModelConfig,
    ) -> GenerateRequest<'a> {
        GenerateRequest {
            model,
            messages,
            config,
        }
    }

    #[tokio::test]
    async fn sequential_returns_in_order() {
        let lm = DummyLM::sequential(vec!["first".into(), "second".into()]);
        let config = LanguageModelConfig::default();
        let model = ModelId::new("test");

        let r1 = lm.generate(req(&model, &[], &config)).await.unwrap();
        assert_eq!(r1.content, "first");

        let r2 = lm.generate(req(&model, &[], &config)).await.unwrap();
        assert_eq!(r2.content, "second");
    }

    #[tokio::test]
    async fn sequential_errors_on_exhaustion() {
        let lm = DummyLM::sequential(vec!["only".into()]);
        let config = LanguageModelConfig::default();
        let model = ModelId::new("test");

        lm.generate(req(&model, &[], &config)).await.unwrap();
        let err = lm.generate(req(&model, &[], &config)).await.unwrap_err();
        assert!(matches!(err, LanguageModelError::EmptyResponse));
    }

    #[tokio::test]
    async fn lookup_matches_content() {
        let lm = DummyLM::lookup(HashMap::from([
            ("capital of France".into(), "Paris".into()),
            ("2+2".into(), "4".into()),
        ]));
        let config = LanguageModelConfig::default();
        let model = ModelId::new("test");
        let messages = vec![Message::user("What is the capital of France?")];

        let response = lm.generate(req(&model, &messages, &config)).await.unwrap();
        assert_eq!(response.content, "Paris");
    }

    #[tokio::test]
    async fn lookup_errors_on_no_match() {
        let lm = DummyLM::lookup(HashMap::from([("hello".into(), "world".into())]));
        let config = LanguageModelConfig::default();
        let model = ModelId::new("test");
        let messages = vec![Message::user("something else")];

        let err = lm
            .generate(req(&model, &messages, &config))
            .await
            .unwrap_err();
        assert!(matches!(err, LanguageModelError::Provider { .. }));
    }

    #[tokio::test]
    async fn response_has_model_name() {
        let lm = DummyLM::sequential(vec!["test".into()]);
        let config = LanguageModelConfig::default();
        let model = ModelId::new("test");
        let response = lm.generate(req(&model, &[], &config)).await.unwrap();
        assert_eq!(response.model.as_deref(), Some("dummy"));
    }

    #[test]
    fn is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DummyLM>();
    }

    #[tokio::test]
    async fn scripted_stream_emits_one_delta_per_chunk() {
        use futures::StreamExt;

        let lm = DummyLM::scripted_stream(vec![vec!["hello".into(), " ".into(), "world".into()]]);
        let config = LanguageModelConfig::default();
        let model = ModelId::new("test");

        let mut stream = lm.generate_stream(req(&model, &[], &config)).await.unwrap();
        let mut deltas = Vec::new();
        while let Some(delta) = stream.next().await {
            deltas.push(delta.unwrap());
        }

        assert_eq!(deltas.len(), 3);
        assert_eq!(deltas[0].content, "hello");
        assert_eq!(deltas[1].content, " ");
        assert_eq!(deltas[2].content, "world");

        // Only the last delta is_final + carries usage.
        assert!(!deltas[0].is_final);
        assert!(!deltas[1].is_final);
        assert!(deltas[2].is_final);
        assert!(deltas[0].usage.is_none());
        assert!(deltas[1].usage.is_none());
        let usage = deltas[2].usage.expect("final delta must carry usage");
        assert_eq!(usage.output_tokens, 11); // "hello" + " " + "world" = 11 bytes
    }

    #[tokio::test]
    async fn scripted_stream_without_usage_omits_usage_on_final_delta() {
        use futures::StreamExt;

        let lm = DummyLM::scripted_stream_without_usage(vec![vec![
            "hello".into(),
            " ".into(),
            "world".into(),
        ]]);
        let config = LanguageModelConfig::default();
        let model = ModelId::new("test");

        let mut stream = lm.generate_stream(req(&model, &[], &config)).await.unwrap();
        let mut deltas = Vec::new();
        while let Some(delta) = stream.next().await {
            deltas.push(delta.unwrap());
        }

        assert_eq!(deltas.len(), 3);
        assert!(deltas[2].is_final);
        assert!(
            deltas[0].usage.is_none() && deltas[1].usage.is_none() && deltas[2].usage.is_none(),
            "scripted_stream_without_usage must not surface usage on ANY delta"
        );
    }

    #[tokio::test]
    async fn scripted_stream_advances_per_call() {
        use futures::StreamExt;

        let lm = DummyLM::scripted_stream(vec![
            vec!["first".into()],
            vec!["second-a".into(), "second-b".into()],
        ]);
        let config = LanguageModelConfig::default();
        let model = ModelId::new("test");

        let mut s1 = lm.generate_stream(req(&model, &[], &config)).await.unwrap();
        let mut s1_contents = Vec::new();
        while let Some(d) = s1.next().await {
            s1_contents.push(d.unwrap().content);
        }
        assert_eq!(s1_contents, vec!["first"]);

        let mut s2 = lm.generate_stream(req(&model, &[], &config)).await.unwrap();
        let mut s2_contents = Vec::new();
        while let Some(d) = s2.next().await {
            s2_contents.push(d.unwrap().content);
        }
        assert_eq!(s2_contents, vec!["second-a", "second-b"]);

        // Third call exhausts the script. `generate_stream`'s Ok
        // variant (`Pin<Box<dyn Stream + Send>>`) doesn't impl Debug,
        // so we can't `.unwrap_err()` — match explicitly instead.
        match lm.generate_stream(req(&model, &[], &config)).await {
            Err(LanguageModelError::EmptyResponse) => {}
            Err(other) => panic!("expected EmptyResponse, got {other:?}"),
            Ok(_) => panic!("expected EmptyResponse error, got Ok"),
        }
    }

    #[tokio::test]
    async fn scripted_generate_returns_joined_chunks() {
        // Non-streaming path on a scripted instance: the concatenated
        // chunks come back as one buffered response. Lets the same
        // fixture drive both code paths.
        let lm = DummyLM::scripted_stream(vec![vec!["a".into(), "bc".into(), "def".into()]]);
        let config = LanguageModelConfig::default();
        let model = ModelId::new("test");
        let response = lm.generate(req(&model, &[], &config)).await.unwrap();
        assert_eq!(response.content, "abcdef");
    }

    #[tokio::test]
    async fn scripted_stream_with_delay_actually_yields_between_chunks() {
        use futures::StreamExt;

        // Cancellation-test prerequisite: with a chunk_delay set, the
        // total stream time must be at least chunks * delay. Without
        // a yield point between chunks, the stream would emit all
        // deltas in one tokio task tick and a cancellation signal
        // could never fire mid-stream. We use tokio::time::pause +
        // tokio::time::advance to avoid real-time waiting in the test.
        tokio::time::pause();

        let lm = DummyLM::scripted_stream_with_delay(
            vec![vec!["a".into(), "b".into(), "c".into()]],
            Duration::from_millis(50),
        );
        let config = LanguageModelConfig::default();
        let model = ModelId::new("test");

        let mut stream = lm.generate_stream(req(&model, &[], &config)).await.unwrap();
        let mut received = Vec::new();
        // Advance virtual time enough to cover all 3 sleeps (50ms each).
        // Each chunk's sleep blocks the next .next() poll.
        for expected_content in ["a", "b", "c"] {
            tokio::time::advance(Duration::from_millis(50)).await;
            let delta = stream
                .next()
                .await
                .expect("stream must yield a delta")
                .expect("Ok delta");
            assert_eq!(delta.content, expected_content);
            received.push(delta.content);
        }
        assert_eq!(received, vec!["a", "b", "c"]);
        // Stream is now exhausted.
        assert!(stream.next().await.is_none());
    }
}
