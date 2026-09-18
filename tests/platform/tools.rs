// Tool catalog stories, attached to src/platform/tools.rs.

use super::*;

#[test]
fn dynamic_catalog_preserves_every_las_tool_and_adds_being_tools() {
    let offered = ["warsztat__workspace_create", "finance__finance_execute"].map(|name| McpTool {
        name: name.into(),
        description: String::new(),
        input_schema: json!({}),
    });
    let catalog = ToolCatalog::build(&offered, false).unwrap();
    let names: Vec<_> = catalog
        .definitions()
        .iter()
        .map(|definition| definition.function.name.as_str())
        .collect();

    assert!(names.contains(&"warsztat__workspace_create"));
    assert!(names.contains(&"finance__finance_execute"));
    assert!(names.contains(&MEMORY_REMEMBER));
    assert!(names.contains(&SELF_SET_PROMPT));
    assert!(names.contains(&SPAWN_CHILD));
}
