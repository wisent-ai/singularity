//! The one printer behind every report `singularity` prints: JSON by default,
//! for the machines that read it, or the same document as indented
//! `key: value` lines for a person with the global `--text`.

use serde::Serialize;
use serde_json::Value;

use crate::error::AppError;

const INDENT: &str = "  ";

/// Print one report in the form the invocation asked for.
pub fn answer<T: Serialize>(report: &T, text: bool) -> Result<(), AppError> {
    if !text {
        println!("{}", serde_json::to_string_pretty(report)?);
        return Ok(());
    }
    let value = serde_json::to_value(report)?;
    let mut out = String::new();
    render(&value, 0, &mut out);
    print!("{out}");
    Ok(())
}

fn scalar(value: &Value) -> Option<String> {
    match value {
        Value::Null => Some("null".to_string()),
        Value::Bool(flag) => Some(flag.to_string()),
        Value::Number(number) => Some(number.to_string()),
        Value::String(text) => Some(text.clone()),
        Value::Array(_) | Value::Object(_) => None,
    }
}

fn render(value: &Value, depth: usize, out: &mut String) {
    let indent = INDENT.repeat(depth);
    if let Some(line) = scalar(value) {
        out.push_str(&format!("{indent}{line}\n"));
        return;
    }
    match value {
        Value::Object(map) => {
            for (key, member) in map {
                match scalar(member) {
                    Some(line) => out.push_str(&format!("{indent}{key}: {line}\n")),
                    None => {
                        out.push_str(&format!("{indent}{key}:\n"));
                        render(member, depth + 1, out);
                    }
                }
            }
        }
        Value::Array(items) => {
            for member in items {
                match scalar(member) {
                    Some(line) => out.push_str(&format!("{indent}- {line}\n")),
                    None => {
                        out.push_str(&format!("{indent}-\n"));
                        render(member, depth + 1, out);
                    }
                }
            }
        }
        _ => {}
    }
}
