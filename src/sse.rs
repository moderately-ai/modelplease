// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Spec-compliant Server-Sent Events (SSE) parser.
//!
//! Implements the WHATWG SSE parsing algorithm from
//! <https://html.spec.whatwg.org/multipage/server-sent-events.html>.
//!
//! Handles all three line delimiters (CR, LF, CRLF), multi-line data fields,
//! BOM stripping, comment lines, and the full field processing rules.

use std::{
    pin::Pin,
    task::{Context, Poll},
};

use bytes::Bytes;
use futures::stream::Stream;

use crate::error::LanguageModelError;

/// A parsed Server-Sent Event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// The event type (`event:` field value, or `"message"` if not set).
    pub event_type: String,
    /// The event data (concatenated `data:` field values, joined with `\n`).
    pub data: String,
    /// The event ID (`id:` field value, empty string if not set).
    pub id: String,
}

/// Parse a byte stream (e.g. from `reqwest::Response::bytes_stream()`) into
/// a stream of [`SseEvent`]s following the WHATWG SSE specification.
pub fn parse_sse_stream<S, E>(byte_stream: S) -> SseStream<S>
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
    E: std::fmt::Display,
{
    SseStream {
        inner: byte_stream,
        line_buf: Vec::new(),
        last_was_cr: bool::default(),
        event_buf: EventBuffer::default(),
        last_event_id: String::new(),
        pending_events: Vec::new(),
        bom_stripped: false,
        done: false,
    }
}

/// Stream adapter that transforms a byte stream into SSE events.
pub struct SseStream<S> {
    inner: S,
    line_buf: Vec<u8>,
    last_was_cr: bool,
    event_buf: EventBuffer,
    last_event_id: String,
    pending_events: Vec<SseEvent>,
    bom_stripped: bool,
    done: bool,
}

#[derive(Default)]
struct EventBuffer {
    data: String,
    event_type: String,
    id_buffer: String,
}

impl EventBuffer {
    fn dispatch(&mut self, last_event_id: &str) -> Option<SseEvent> {
        if self.data.is_empty() {
            self.event_type.clear();
            self.id_buffer.clear();
            return None;
        }

        // Strip trailing \n added by the last data: line
        if self.data.ends_with('\n') {
            self.data.pop();
        }

        let event = SseEvent {
            event_type: if self.event_type.is_empty() {
                "message".to_owned()
            } else {
                std::mem::take(&mut self.event_type)
            },
            data: std::mem::take(&mut self.data),
            id: if self.id_buffer.is_empty() {
                last_event_id.to_owned()
            } else {
                std::mem::take(&mut self.id_buffer)
            },
        };

        self.event_type.clear();
        self.id_buffer.clear();

        Some(event)
    }

    fn process_line(&mut self, line: &str, last_event_id: &mut String) {
        // Comment
        if line.starts_with(':') {
            return;
        }

        // Split at first ':'. Absent → field name is the whole line, value empty.
        let (name, value) = line.find(':').map_or((line, ""), |colon_pos| {
            let name = &line[..colon_pos];
            let raw_value = &line[colon_pos + 1..];
            // Strip exactly one leading space
            let value = raw_value.strip_prefix(' ').unwrap_or(raw_value);
            (name, value)
        });

        match name {
            "data" => {
                self.data.push_str(value);
                self.data.push('\n');
            }
            "event" => {
                value.clone_into(&mut self.event_type);
            }
            "id" if !value.contains('\0') => {
                value.clone_into(&mut self.id_buffer);
                value.clone_into(last_event_id);
            }
            // "retry" is spec-defined but we don't implement reconnection;
            // unknown fields are ignored per the SSE spec.
            _ => {}
        }
    }
}

impl<S, E> SseStream<S>
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
    E: std::fmt::Display,
{
    /// Process a single byte, emitting lines and dispatching events.
    fn process_byte(&mut self, b: u8) {
        if self.last_was_cr {
            self.last_was_cr = false;
            if b == b'\n' {
                // Second byte of CRLF, already emitted line on CR
                return;
            }
        }

        if b == b'\r' {
            self.last_was_cr = true;
            self.emit_line();
        } else if b == b'\n' {
            self.emit_line();
        } else {
            self.line_buf.push(b);
        }
    }

    /// Convert `line_buf` to a string, process it, and clear the buffer.
    fn emit_line(&mut self) {
        // BOM stripping: only on the very first line, at the start
        if !self.bom_stripped {
            self.bom_stripped = true;
            if self.line_buf.starts_with(&[0xEF, 0xBB, 0xBF]) {
                self.line_buf.drain(..3);
            }
        }

        let line = String::from_utf8_lossy(&self.line_buf).into_owned();
        self.line_buf.clear();

        if line.is_empty() {
            // Blank line: dispatch event
            if let Some(event) = self.event_buf.dispatch(&self.last_event_id) {
                self.pending_events.push(event);
            }
        } else {
            self.event_buf.process_line(&line, &mut self.last_event_id);
        }
    }

    /// Flush any remaining data at end of stream.
    fn flush(&mut self) {
        if !self.line_buf.is_empty() {
            self.emit_line();
        }
        // Dispatch any pending event without a trailing blank line
        if let Some(event) = self.event_buf.dispatch(&self.last_event_id) {
            self.pending_events.push(event);
        }
    }
}

impl<S, E> Stream for SseStream<S>
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
    E: std::fmt::Display,
{
    type Item = Result<SseEvent, LanguageModelError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        loop {
            // Return pending events first
            if !this.pending_events.is_empty() {
                return Poll::Ready(Some(Ok(this.pending_events.remove(0))));
            }

            if this.done {
                return Poll::Ready(None);
            }

            // Poll the inner byte stream
            match Pin::new(&mut this.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    for b in &bytes {
                        this.process_byte(*b);
                    }
                    // Continue loop to check for pending events
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(LanguageModelError::provider(e.to_string()))));
                }
                Poll::Ready(None) => {
                    // Stream ended — flush remaining
                    this.done = true;
                    this.flush();
                    // Continue loop to drain pending events
                }
                Poll::Pending => {
                    return Poll::Pending;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use futures::stream::{self, StreamExt};

    use super::*;

    /// Helper: parse a string as an SSE byte stream and collect all events.
    async fn parse_str(input: &str) -> Vec<SseEvent> {
        let bytes = Bytes::from(input.to_owned());
        let byte_stream = stream::iter(vec![Ok::<_, std::io::Error>(bytes)]);
        let mut sse_stream = parse_sse_stream(byte_stream);
        let mut events = Vec::new();
        while let Some(result) = sse_stream.next().await {
            events.push(result.unwrap());
        }
        events
    }

    #[tokio::test]
    async fn basic_data_event() {
        let events = parse_str("data: hello\n\n").await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "message");
        assert_eq!(events[0].data, "hello");
    }

    #[tokio::test]
    async fn multi_line_data() {
        let events = parse_str("data: line one\ndata: line two\n\n").await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "line one\nline two");
    }

    #[tokio::test]
    async fn event_type_field() {
        let events = parse_str("event: custom\ndata: payload\n\n").await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "custom");
        assert_eq!(events[0].data, "payload");
    }

    #[tokio::test]
    async fn comment_lines_ignored() {
        let events = parse_str(": this is a comment\ndata: hello\n\n").await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[tokio::test]
    async fn empty_data_not_dispatched() {
        // Blank line with no preceding data: lines → no event
        let events = parse_str("event: test\n\ndata: real\n\n").await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "real");
    }

    #[tokio::test]
    async fn crlf_line_endings() {
        let events = parse_str("data: hello\r\n\r\n").await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[tokio::test]
    async fn cr_only_line_endings() {
        let events = parse_str("data: hello\r\r").await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[tokio::test]
    async fn mixed_line_endings() {
        let events = parse_str("data: a\ndata: b\r\ndata: c\r\r").await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "a\nb\nc");
    }

    #[tokio::test]
    async fn strip_one_leading_space() {
        let events = parse_str("data: hello\n\n").await;
        assert_eq!(events[0].data, "hello");

        // No space: still works
        let events = parse_str("data:hello\n\n").await;
        assert_eq!(events[0].data, "hello");

        // Two spaces: strip one, keep one
        let events = parse_str("data:  hello\n\n").await;
        assert_eq!(events[0].data, " hello");
    }

    #[tokio::test]
    async fn bom_at_stream_start() {
        let mut input = Vec::new();
        input.extend_from_slice(&[0xEF, 0xBB, 0xBF]); // UTF-8 BOM
        input.extend_from_slice(b"data: hello\n\n");
        let bytes = Bytes::from(input);
        let byte_stream = stream::iter(vec![Ok::<_, std::io::Error>(bytes)]);
        let mut sse_stream = parse_sse_stream(byte_stream);
        let event = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event.data, "hello");
    }

    #[tokio::test]
    async fn id_field() {
        let events = parse_str("id: 42\ndata: hello\n\n").await;
        assert_eq!(events[0].id, "42");
    }

    #[tokio::test]
    async fn id_with_null_ignored() {
        let events = parse_str("id: bad\0id\ndata: hello\n\n").await;
        assert_eq!(events[0].id, ""); // NULL in id → ignored, falls back to empty
    }

    #[tokio::test]
    async fn id_persists_across_events() {
        let events = parse_str("id: 1\ndata: first\n\ndata: second\n\n").await;
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].id, "1");
        assert_eq!(events[1].id, "1"); // persists from previous event
    }

    #[tokio::test]
    async fn unknown_fields_ignored() {
        let events = parse_str("foo: bar\ndata: hello\n\n").await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[tokio::test]
    async fn multiple_events() {
        let events = parse_str("data: first\n\ndata: second\n\ndata: third\n\n").await;
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].data, "first");
        assert_eq!(events[1].data, "second");
        assert_eq!(events[2].data, "third");
    }

    #[tokio::test]
    async fn field_with_no_colon() {
        // "data" with no colon → name="data", value=""
        let events = parse_str("data\n\n").await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "");
    }

    #[tokio::test]
    async fn chunked_byte_delivery() {
        // Simulate bytes arriving in small chunks across multiple polls
        let chunks: Vec<Result<Bytes, std::io::Error>> = vec![
            Ok(Bytes::from("dat")),
            Ok(Bytes::from("a: hel")),
            Ok(Bytes::from("lo\n")),
            Ok(Bytes::from("\n")),
        ];
        let byte_stream = stream::iter(chunks);
        let mut sse_stream = parse_sse_stream(byte_stream);
        let event = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event.data, "hello");
    }

    #[tokio::test]
    async fn anthropic_format() {
        let input = "\
event: message_start\n\
data: {\"type\":\"message_start\"}\n\
\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"Hello\"}}\n\
\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\
\n";
        let events = parse_str(input).await;
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].event_type, "message_start");
        assert_eq!(events[1].event_type, "content_block_delta");
        assert_eq!(events[2].event_type, "message_stop");
    }

    #[tokio::test]
    async fn openai_format() {
        let input = "\
data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\
\n\
data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\
\n\
data: [DONE]\n\
\n";
        let events = parse_str(input).await;
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].event_type, "message"); // default type
        assert_eq!(events[2].data, "[DONE]");
    }

    #[tokio::test]
    async fn flush_without_trailing_blank_line() {
        // Stream ends without a final blank line — still dispatch
        let events = parse_str("data: hello\n").await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[tokio::test]
    async fn empty_stream() {
        let events = parse_str("").await;
        assert!(events.is_empty());
    }
}
