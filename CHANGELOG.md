# Changelog

All notable changes are documented here. This project follows Semantic Versioning while allowing
normal pre-1.0 evolution.

## [Unreleased]

## [0.1.1] - 2026-07-31

- Restore the documented Rust 1.88 minimum by pinning the compatible AWS SDK family.
- Make core-only Rustdoc builds independent of provider-only exports.

## [0.1.0] - 2026-07-31

- Initial provider-neutral language-model client API.
- Optional Anthropic, OpenAI, Ollama, AWS Bedrock, and Bedrock Mantle integrations.
- Buffered and streaming generation, media inputs, capability validation, retry support, and a
  deterministic dummy provider for tests.

[Unreleased]: https://github.com/moderately-ai/modelplease/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/moderately-ai/modelplease/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/moderately-ai/modelplease/releases/tag/v0.1.0
