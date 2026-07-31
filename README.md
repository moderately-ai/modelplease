# modelplease

Provider-neutral language-model clients for Rust. `modelplease` supplies one request, response,
streaming, media, capability, and error vocabulary across Anthropic, OpenAI-compatible APIs,
Ollama, AWS Bedrock, and Bedrock Mantle.

No network provider is enabled by default.

```bash
cargo add modelplease
cargo add modelplease --features openai
cargo add modelplease --features anthropic
cargo add modelplease --features ollama
cargo add modelplease --features bedrock
cargo add modelplease --features bedrock-mantle
cargo add modelplease --features all-providers
```

## Provider features

| Feature | Integration | Streaming | Media and reasoning |
|---|---|---|---|
| `anthropic` | Anthropic Messages API | Yes | Capability-checked by model |
| `openai` | OpenAI and compatible endpoints such as vLLM or MLX | Yes | Capability-checked by model |
| `ollama` | Local Ollama daemon | Yes | Capability-checked by model |
| `bedrock` | AWS Bedrock Converse and ConverseStream | Yes | Capability-checked by model |
| `bedrock-mantle` | Bedrock Mantle OpenAI/Anthropic-compatible surfaces | Yes | Capability-checked by model |

## Shared API

The core-only build includes messages, media, provider traits, requests, responses, capability
validation, retry policy, SSE parsing, and `DummyLM` for deterministic tests.

```rust
use std::sync::Arc;

use modelplease::{
    DummyLM, GenerateRequest, LanguageModelConfig, LanguageModelProvider, Message, ModelId,
};

# async fn run() -> Result<(), modelplease::LanguageModelError> {
let provider = DummyLM::sequential(vec!["Paris".into()]);
let model = ModelId::new("test-model");
let messages = [Message::user("What is the capital of France?")];
let config = LanguageModelConfig::default();
let response = provider
    .generate(GenerateRequest { model: &model, messages: &messages, config: &config })
    .await?;
assert_eq!(response.content, "Paris");
# Ok(())
# }
```

## OpenAI and compatible servers

```rust,no_run
use std::sync::Arc;

use modelplease::{ApiKey, OpenAiConfig, OpenAiDeps, OpenAiLanguageModel, RetryConfig};

# fn build() -> Result<OpenAiLanguageModel, modelplease::ApiKeyError> {
let provider = OpenAiLanguageModel::new(
    OpenAiDeps { client: Arc::new(reqwest::Client::new()) },
    OpenAiConfig {
        api_key: ApiKey::parse(std::env::var("OPENAI_API_KEY").unwrap_or_default())?,
        base_url: OpenAiConfig::DEFAULT_BASE_URL.to_owned(),
        retry_config: RetryConfig::default(),
    },
);
# Ok(provider)
# }
```

Set `OpenAiConfig::base_url` to an OpenAI-compatible `/v1` endpoint for vLLM, MLX, or another
compatible server.

## Media

Messages support validated HTTPS URLs, inline bytes, provider file identifiers, and S3 URIs.
Each provider advertises and validates its accepted modalities and source kinds before a request
reaches the wire.

```rust
use modelplease::{ContentPart, HttpsUrl, MediaSource, Message, Role};

let message = Message::with_parts(
    Role::User,
    vec![
        ContentPart::text("Describe this image"),
        ContentPart::image(MediaSource::Url {
            url: HttpsUrl::parse("https://example.com/image.png").unwrap(),
        }),
    ],
);
```

## Streaming

`LanguageModelProvider::generate_stream` returns provider-neutral `StreamDelta` values. The public
`parse_sse_stream` adapter is also available for integrations that need WHATWG-compatible SSE
parsing directly.

```rust
use futures::StreamExt;
use modelplease::{
    DummyLM, GenerateRequest, LanguageModelConfig, LanguageModelProvider, Message, ModelId,
};

# async fn run() -> Result<(), modelplease::LanguageModelError> {
let provider = DummyLM::scripted_stream(vec![vec!["Par".into(), "is".into()]]);
let model = ModelId::new("test-model");
let messages = [Message::user("What is the capital of France?")];
let config = LanguageModelConfig::default();
let mut stream = provider
    .generate_stream(GenerateRequest { model: &model, messages: &messages, config: &config })
    .await?;
let mut text = String::new();
while let Some(delta) = stream.next().await {
    text.push_str(&delta?.content);
}
assert_eq!(text, "Paris");
# Ok(())
# }
```

The streaming example also needs `futures = "0.3"` in the consuming application.

## Credentials and live tests

`ApiKey` validates input and redacts both `Debug` and `Display`. Never commit credentials.
Credential-backed Bedrock tests are ignored by default:

```bash
cargo test --all-features
cargo test --features bedrock --test bedrock_live -- --ignored
cargo test --features bedrock-mantle --test bedrock_mantle_live -- --ignored
```

## Compatibility and license

The MSRV is Rust 1.94.1, required by the current AWS SDK dependency graph. Licensed under either
MIT or Apache-2.0 at your option.
