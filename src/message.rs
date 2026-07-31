// Copyright 2026 Thomas Santerre and Moderately AI Inc.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! LM conversation message types.
//!
//! Messages follow the OpenAI-style content parts pattern: each message has a
//! role and a list of typed [`ContentPart`]s. Beyond text, the part variants
//! cover the four multimodal modalities every backend we support exposes —
//! image, document, audio, video — with the source-kind discrimination
//! pushed into [`MediaSource`] so providers can publish per-model
//! capability tables that gate URL vs base64 vs `S3` vs provider-file
//! references uniformly.
//!
//! Per-provider translation lives in `openai.rs`, `anthropic.rs`,
//! `bedrock.rs`, and `ollama.rs`. Each one chooses which [`MediaSource`]
//! variants it natively speaks and returns
//! [`CapabilityError`](crate::CapabilityError) for the rest — the
//! validation lives on the capability tables, not here.

use serde::{Deserialize, Serialize};

use crate::{capabilities::MediaKind, media::MediaSource};

/// Message role in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System instructions.
    System,
    /// User input.
    User,
    /// Assistant response.
    Assistant,
}

/// A single content part within a message.
///
/// The four non-text variants share a [`MediaSource`] payload that
/// chooses between URL / inline-bytes / provider-file / S3 references.
/// [`ContentPart::Document`] additionally carries an optional `name`
/// for providers that surface a filename on the wire (Bedrock's
/// `DocumentBlock.name`, OpenAI's `file.filename`).
///
/// # Serde
///
/// Internally tagged on `type`:
///
/// ```json
/// {"type": "text", "text": "Hello"}
/// {"type": "image", "source": {"source": "url", "url": "..."}}
/// {"type": "document", "name": "report.pdf", "source": {...}}
/// {"type": "audio", "source": {...}}
/// {"type": "video", "source": {...}}
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// A text content part.
    Text {
        /// The text content.
        text: String,
    },
    /// An image. Source kind determined at construction time.
    Image {
        /// Where the image bytes come from.
        source: MediaSource,
    },
    /// A document (PDF and other formats per provider). Some providers
    /// surface `name` to the model as a filename hint; passing
    /// untrusted user-supplied filenames is a prompt-injection vector
    /// (Bedrock documents this; treat the same way for other providers).
    Document {
        /// Where the document bytes come from.
        source: MediaSource,
        /// Optional filename to surface to the provider.
        #[serde(skip_serializing_if = "Option::is_none", default)]
        name: Option<String>,
    },
    /// Audio input. Currently OpenAI `gpt-audio*` and Bedrock Voxtral.
    Audio {
        /// Where the audio bytes come from.
        source: MediaSource,
    },
    /// Video input. Currently Bedrock Nova Pro/Lite/2-Lite.
    Video {
        /// Where the video bytes come from.
        source: MediaSource,
    },
    /// A zero-width prompt-cache boundary marker.
    ///
    /// Carries no content — it marks the end of a cacheable prefix. The
    /// builder (`ChatAdapter`) inserts it; each provider translates it
    /// into its own mechanism (Anthropic attaches `cache_control` to the
    /// preceding block; Bedrock emits a standalone `CachePoint` block) or
    /// drops it when the target model can't cache. Placing the boundary
    /// here — in the message content, where the adapter knows what is
    /// static — keeps providers from having to guess the system/user
    /// split or which inputs are stable.
    CacheBreakpoint,
}

impl ContentPart {
    /// Create a text content part.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Create an image content part with the given source.
    #[must_use]
    pub const fn image(source: MediaSource) -> Self {
        Self::Image { source }
    }

    /// Create a document content part with the given source and
    /// optional filename hint.
    #[must_use]
    pub const fn document(source: MediaSource, name: Option<String>) -> Self {
        Self::Document { source, name }
    }

    /// Create an audio content part with the given source.
    #[must_use]
    pub const fn audio(source: MediaSource) -> Self {
        Self::Audio { source }
    }

    /// Create a video content part with the given source.
    #[must_use]
    pub const fn video(source: MediaSource) -> Self {
        Self::Video { source }
    }

    /// Create a prompt-cache boundary marker.
    #[must_use]
    pub const fn cache_breakpoint() -> Self {
        Self::CacheBreakpoint
    }

    /// Whether this is a [`CacheBreakpoint`](ContentPart::CacheBreakpoint)
    /// marker rather than real content.
    #[must_use]
    pub const fn is_cache_breakpoint(&self) -> bool {
        matches!(self, Self::CacheBreakpoint)
    }

    /// Returns the text content if this is a [`Text`](ContentPart::Text) part.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text),
            Self::Image { .. }
            | Self::Document { .. }
            | Self::Audio { .. }
            | Self::Video { .. }
            | Self::CacheBreakpoint => None,
        }
    }

    /// Returns the [`MediaKind`] for non-text parts, or `None` for text
    /// and cache markers. Used by capability validation and per-provider
    /// translation — a `None` here means the part is skipped by the
    /// capability walk, which is correct for the zero-width marker.
    #[must_use]
    pub const fn media_kind(&self) -> Option<MediaKind> {
        match self {
            Self::Text { .. } | Self::CacheBreakpoint => None,
            Self::Image { .. } => Some(MediaKind::Image),
            Self::Document { .. } => Some(MediaKind::Document),
            Self::Audio { .. } => Some(MediaKind::Audio),
            Self::Video { .. } => Some(MediaKind::Video),
        }
    }

    /// Returns the underlying [`MediaSource`] for non-text parts.
    #[must_use]
    pub const fn media_source(&self) -> Option<&MediaSource> {
        match self {
            Self::Text { .. } | Self::CacheBreakpoint => None,
            Self::Image { source }
            | Self::Document { source, .. }
            | Self::Audio { source }
            | Self::Video { source } => Some(source),
        }
    }
}

/// A single message in an LM conversation.
///
/// Content is represented as a list of [`ContentPart`]s, following the OpenAI-style
/// content parts pattern. For simple text-only messages, use the convenience
/// constructors [`Message::system`], [`Message::user`], and [`Message::assistant`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// The role of this message.
    pub role: Role,
    /// The content parts.
    pub content: Vec<ContentPart>,
}

impl Message {
    /// Create a message with the given role and content parts.
    #[must_use]
    pub const fn with_parts(role: Role, content: Vec<ContentPart>) -> Self {
        Self { role, content }
    }

    /// Create a text-only system message.
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![ContentPart::text(text)],
        }
    }

    /// Create a text-only user message.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentPart::text(text)],
        }
    }

    /// Create a text-only assistant message.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentPart::text(text)],
        }
    }

    /// Returns the concatenated text content of all [`Text`](ContentPart::Text) parts.
    ///
    /// Returns an empty string if the message has no text parts.
    #[must_use]
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(ContentPart::as_text)
            .collect::<Vec<_>>()
            .join("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::{HttpsUrl, MediaType, S3Uri};

    fn https(url: &str) -> MediaSource {
        MediaSource::Url {
            url: HttpsUrl::parse(url).unwrap(),
        }
    }

    fn png_bytes(data: Vec<u8>) -> MediaSource {
        MediaSource::InlineBytes {
            mime: MediaType::parse("image/png").unwrap(),
            data,
        }
    }

    #[test]
    fn text_constructors_set_correct_role() {
        assert_eq!(Message::system("hi").role, Role::System);
        assert_eq!(Message::user("hi").role, Role::User);
        assert_eq!(Message::assistant("hi").role, Role::Assistant);
    }

    #[test]
    fn text_constructors_create_single_text_part() {
        let msg = Message::user("hello");
        assert_eq!(msg.content.len(), 1);
        assert_eq!(msg.content[0].as_text(), Some("hello"));
    }

    #[test]
    fn with_parts_preserves_mixed_content() {
        let msg = Message::with_parts(
            Role::User,
            vec![
                ContentPart::text("Describe this:"),
                ContentPart::image(https("https://example.com/img.png")),
            ],
        );
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content.len(), 2);
        assert_eq!(msg.content[0].as_text(), Some("Describe this:"));
        assert_eq!(msg.content[1].as_text(), None);
        assert_eq!(msg.content[1].media_kind(), Some(MediaKind::Image));
    }

    #[test]
    fn text_method_concatenates_text_parts() {
        let msg = Message::with_parts(
            Role::User,
            vec![
                ContentPart::text("Hello "),
                ContentPart::image(https("https://example.com/img.png")),
                ContentPart::text("world"),
            ],
        );
        assert_eq!(msg.text(), "Hello world");
    }

    #[test]
    fn text_method_returns_empty_for_no_text_parts() {
        let msg = Message::with_parts(
            Role::User,
            vec![ContentPart::image(https("https://example.com/img.png"))],
        );
        assert_eq!(msg.text(), "");
    }

    #[test]
    fn content_part_text_as_text() {
        let part = ContentPart::text("hello");
        assert_eq!(part.as_text(), Some("hello"));
        assert_eq!(part.media_kind(), None);
        assert!(part.media_source().is_none());
    }

    #[test]
    fn content_part_image_kind_and_source() {
        let src = https("https://example.com/img.png");
        let part = ContentPart::image(src.clone());
        assert_eq!(part.as_text(), None);
        assert_eq!(part.media_kind(), Some(MediaKind::Image));
        assert_eq!(part.media_source(), Some(&src));
    }

    #[test]
    fn content_part_document_with_name() {
        let src = MediaSource::S3 {
            uri: S3Uri::parse("s3://b/k.pdf").unwrap(),
            bucket_owner: None,
        };
        let part = ContentPart::document(src, Some("report.pdf".into()));
        assert_eq!(part.media_kind(), Some(MediaKind::Document));
        match &part {
            ContentPart::Document { name, .. } => assert_eq!(name.as_deref(), Some("report.pdf")),
            other => panic!("expected Document, got {other:?}"),
        }
    }

    #[test]
    fn message_serde_round_trip() {
        let msg = Message::user("hello world");
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, deserialized);
    }

    #[test]
    fn mixed_content_serde_round_trip() {
        let msg = Message::with_parts(
            Role::User,
            vec![
                ContentPart::text("Look at this:"),
                ContentPart::image(https("https://example.com/img.png")),
            ],
        );
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, deserialized);
    }

    #[test]
    fn role_serializes_lowercase() {
        let json = serde_json::to_string(&Role::System).unwrap();
        assert_eq!(json, "\"system\"");
    }

    #[test]
    fn text_part_serializes_with_type_tag() {
        let part = ContentPart::text("hello");
        let json = serde_json::to_string(&part).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["type"], "text");
        assert_eq!(value["text"], "hello");
    }

    #[test]
    fn image_part_serializes_with_type_tag_and_source() {
        let part = ContentPart::image(https("https://example.com/img.png"));
        let json = serde_json::to_string(&part).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["type"], "image");
        assert_eq!(value["source"]["source"], "url");
        assert_eq!(value["source"]["url"], "https://example.com/img.png");
    }

    #[test]
    fn audio_video_serialize_with_correct_type_tags() {
        let audio = ContentPart::audio(png_bytes(vec![0, 1, 2]));
        let json = serde_json::to_value(&audio).unwrap();
        assert_eq!(json["type"], "audio");

        let video = ContentPart::video(png_bytes(vec![0, 1, 2]));
        let json = serde_json::to_value(&video).unwrap();
        assert_eq!(json["type"], "video");
    }

    #[test]
    fn document_omits_name_when_none() {
        let part = ContentPart::document(https("https://example.com/x.pdf"), None);
        let value = serde_json::to_value(&part).unwrap();
        assert!(value.get("name").is_none());
    }

    #[test]
    fn document_includes_name_when_some() {
        let part = ContentPart::document(
            https("https://example.com/x.pdf"),
            Some("report.pdf".into()),
        );
        let value = serde_json::to_value(&part).unwrap();
        assert_eq!(value["name"], "report.pdf");
    }
}
