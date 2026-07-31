# Contributing

Use Rust 1.94.1 or newer. Before opening a pull request, run:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo nextest run --all-targets --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
cargo deny check --all-features
```

Provider integrations must also compile independently with only their corresponding feature.
Never commit provider credentials or enable live credential-backed tests in ordinary CI.
