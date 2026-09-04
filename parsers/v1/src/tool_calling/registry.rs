// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-overridable tool-call parser registry.
//!
//! Built-in parsers are compiled in and dispatched by [`ToolCallConfig`]. This
//! module adds an *optional* layer on top: a named [`ToolCallParser`] object
//! registered here takes precedence over the built-in of the same name, letting
//! a deployment ship a one-file parser fix or a proprietary grammar as config,
//! with zero image rebuild — and delete the file the moment upstream lands the
//! equivalent change.
//!
//! Resolution order in every dispatch entry point:
//! 1. an object registered under the config's parser name (override), else
//! 2. the built-in [`ParserConfig`] `match`.
//!
//! Overrides are process-global and intended to be installed once at startup.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use anyhow::anyhow;

use super::ToolDefinition;
use super::response::ToolCallResponse;

/// A tool-call parser supplied at runtime. Implementors own compat with the
/// grammar they emit; `ToolCallResponse` is the contract with the jail/streaming
/// layer, so an override breaks loudly at compile time if that struct changes.
pub trait ToolCallParser: Send + Sync {
    /// See [`super::parsers::detect_tool_call_start`].
    fn detect_start(&self, chunk: &str) -> bool;
    /// See [`super::parsers::find_tool_call_end_position`].
    fn find_end(&self, chunk: &str) -> Option<usize>;
    /// See [`super::parsers::try_tool_call_parse`]. Async-agnostic: overrides
    /// that need `.await` (harmony-style) should block_on internally or be
    /// wrapped; the sync signature keeps the registry object-safe and the
    /// built-in fast path allocation-free.
    fn parse(
        &self,
        message: &str,
        tools: Option<&[ToolDefinition]>,
    ) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)>;
}

static REGISTRY: RwLock<Option<HashMap<String, Arc<dyn ToolCallParser>>>> = RwLock::new(None);

/// Install `parser` under `name`. If `name` matches a built-in (e.g.
/// `"kimi_k3"`), this object shadows it for every subsequent dispatch — that is
/// the intended override behavior. Re-registering the same name replaces the
/// prior override.
pub fn register_tool_call_parser(name: impl Into<String>, parser: Arc<dyn ToolCallParser>) {
    let mut guard = REGISTRY.write().expect("parser registry poisoned");
    guard.get_or_insert_with(HashMap::new).insert(name.into(), parser);
}

/// Remove the override under `name`, restoring built-in dispatch. Returns the
/// removed parser if one was registered. Deleting the registration is the
/// "single config change" once upstream carries the fix.
pub fn unregister_tool_call_parser(name: &str) -> Option<Arc<dyn ToolCallParser>> {
    let mut guard = REGISTRY.write().expect("parser registry poisoned");
    guard.as_mut().and_then(|m| m.remove(name))
}

/// Look up an override by parser name. Returns `None` when the registry is
/// empty or has no entry under `name` — the common case, which falls through to
/// the built-in `match` with only a read-lock + hash lookup added.
pub fn get_tool_call_parser_override(name: &str) -> Option<Arc<dyn ToolCallParser>> {
    let guard = REGISTRY.read().expect("parser registry poisoned");
    guard.as_ref().and_then(|m| m.get(name).cloned())
}

/// Convenience assert for tests and startup validation: is `name` overridden?
pub fn is_overridden(name: &str) -> bool {
    let guard = REGISTRY.read().expect("parser registry poisoned");
    guard.as_ref().is_some_and(|m| m.contains_key(name))
}

/// Error helper for overrides that only implement the aggregate entry point and
/// want a uniform message for the streaming hooks they don't support.
pub fn unsupported_hook(hook: &str, name: &str) -> anyhow::Error {
    anyhow!("override parser '{name}' does not implement streaming hook '{hook}'")
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoParser;
    impl ToolCallParser for EchoParser {
        fn detect_start(&self, chunk: &str) -> bool {
            chunk.contains("<echo>")
        }
        fn find_end(&self, chunk: &str) -> Option<usize> {
            chunk.find("</echo>").map(|p| p + "</echo>".len())
        }
        fn parse(
            &self,
            message: &str,
            _tools: Option<&[ToolDefinition]>,
        ) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
            Ok((vec![], Some(message.to_string())))
        }
    }

    #[test]
    fn override_roundtrip_shadows_then_restores() {
        assert!(!is_overridden("kimi_k3"));
        register_tool_call_parser("kimi_k3", Arc::new(EchoParser));
        assert!(is_overridden("kimi_k3"));
        let removed = unregister_tool_call_parser("kimi_k3");
        assert!(removed.is_some());
        assert!(!is_overridden("kimi_k3"));
        assert!(unregister_tool_call_parser("kimi_k3").is_none());
    }
}
