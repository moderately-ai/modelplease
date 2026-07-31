// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared helpers for HTTP error-path handling across providers.

use reqwest::Response;

/// Read a non-success response body as a string, logging at `warn` when
/// the read itself fails.
///
/// Callers want the provider's failure text (e.g. "rate limited, retry
/// in 60s" vs "quota exhausted") to surface through the typed error
/// returned by `map_*_error`. If the body read itself fails (network
/// cut mid-response, decoder error) we can't make up a body — but we
/// can make the read failure loud in the logs so operators aren't left
/// debugging an opaque status-only error.
pub async fn read_error_body_or_warn(
    response: Response,
    provider: &'static str,
    status: u16,
) -> String {
    match response.text().await {
        Ok(body) => body,
        Err(e) => {
            tracing::warn!(
                provider,
                status,
                error = %e,
                "failed to read error response body; surfacing status-only error"
            );
            String::new()
        }
    }
}
