//! The one declared function a portfolio judgement answers through. Each
//! caller states its answer as a JSON Schema; the request declares one
//! function with that schema as its parameters, and the answer is the
//! arguments of that one call, so prose around an answer or a field named
//! outside the schema is refused by name instead of parsed out of text.

use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::AppError;
use crate::brama::BramaCompletion;
use crate::domain::ToolDefinition;

/// The declared function's name.
const ANSWER: &str = "answer";

/// The function whose parameters are `schema`.
pub(super) fn tool(schema: Value) -> ToolDefinition {
    ToolDefinition::function(ANSWER, "The judgement, as its schema states it.", schema)
}

/// The arguments of the one `answer` call `completion` carries, read as `T`.
pub(super) fn read<T: DeserializeOwned>(
    purpose: &str,
    completion: &BramaCompletion,
) -> Result<T, AppError> {
    let [call] = completion.tool_calls.as_slice() else {
        return Err(AppError::State(format!(
            "{purpose}: the model answered {} function calls and text {:?}, not one call of {ANSWER}",
            completion.tool_calls.len(),
            completion.content
        )));
    };
    if call.function.name != ANSWER {
        return Err(AppError::State(format!(
            "{purpose}: the model called {}, which was not offered",
            call.function.name
        )));
    }
    serde_json::from_str(&call.function.arguments).map_err(|error| {
        AppError::State(format!(
            "{purpose}: the {ANSWER} arguments do not fit the judgement ({error}): {}",
            call.function.arguments
        ))
    })
}

/// A schema object whose properties are `properties`, every one required and
/// no other admitted.
pub(crate) fn object(properties: Value) -> Value {
    let required: Vec<String> = properties
        .as_object()
        .into_iter()
        .flat_map(|fields| fields.keys().cloned())
        .collect();
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })
}
