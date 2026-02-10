"""
Supabase Skill - Database Operations

Contains run_sql, list_tables, create_table.
"""

from typing import Dict, List
from singularity.skills.base import SkillResult


async def run_sql(skill, project_id: str, query: str) -> SkillResult:
    """Run SQL query on project database"""
    if not project_id or not query:
        return SkillResult(success=False, message="project_id and query required")

    response = await skill.http.post(
        f"{skill.API_BASE}/projects/{project_id}/database/query",
        headers=skill._get_headers(),
        json={"query": query}
    )

    if response.status_code == 200:
        result = response.json()
        return SkillResult(success=True, message="Query executed", data={"result": result})
    elif response.status_code == 201:
        return SkillResult(success=True, message="Query executed (no results)", data={"result": []})
    else:
        return SkillResult(success=False, message=f"Query failed: {response.text}")


async def list_tables(skill, project_id: str, schema: str = "public") -> SkillResult:
    """List tables in database"""
    query = f"""
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = '{schema}'
        ORDER BY table_name;
    """
    return await run_sql(skill, project_id, query)


async def create_table(skill, project_id: str, table_name: str,
                       columns: List[Dict], enable_rls: bool = True) -> SkillResult:
    """Create a new table"""
    if not project_id or not table_name or not columns:
        return SkillResult(success=False, message="project_id, table_name, and columns required")

    col_defs = []
    for col in columns:
        col_def = f"{col['name']} {col['type']}"
        if col.get('primary_key'):
            col_def += " PRIMARY KEY"
        if not col.get('nullable', True):
            col_def += " NOT NULL"
        if col.get('default'):
            col_def += f" DEFAULT {col['default']}"
        col_defs.append(col_def)

    cols_str = ',\n  '.join(col_defs)
    sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n  {cols_str}\n);"

    if enable_rls:
        sql += f"\nALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY;"

    result = await run_sql(skill, project_id, sql)

    if result.success:
        return SkillResult(
            success=True,
            message=f"Created table: {table_name}",
            data={
                "table_name": table_name,
                "columns": [c['name'] for c in columns],
                "rls_enabled": enable_rls,
                "sql": sql
            }
        )
    else:
        return result
