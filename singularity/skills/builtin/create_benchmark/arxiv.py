"""
Academic paper generation and arXiv submission.
"""

import io
import json
import os
import tarfile
from collections import Counter
from typing import Dict

from singularity.skills.base import SkillResult


async def generate_paper(skill, params: Dict) -> SkillResult:
    """LLM generates a LaTeX academic paper describing the benchmark."""
    project_name = params.get("project_name")
    if not project_name:
        return SkillResult(success=False, message="Missing required parameter: project_name")

    state = skill._load_state(project_name)
    spec = state.get("spec")
    if not spec:
        return SkillResult(success=False, message=f"No spec for '{project_name}'.")

    authors = params.get("authors", "Anonymous")
    abstract_focus = params.get("abstract_focus", "")

    # Build stats summary
    stats_lines = []
    for split in ("train", "test", "val"):
        exs = state["examples"].get(split, [])
        if exs:
            cats = Counter(e.get("category", "?") for e in exs if isinstance(e, dict))
            stats_lines.append(f"{split}: {len(exs)} examples, categories={dict(cats)}")

    # Sample examples
    sample_exs = []
    for split in ("test", "train", "val"):
        exs = state["examples"].get(split, [])
        if exs:
            sample_exs = exs[:3]
            break

    prompt = f"""Write a complete LaTeX academic paper for this evaluation benchmark.

Benchmark: {spec['name']}
Description: {spec['description']}
Task type: {spec['task_type']}
Metrics: {json.dumps(spec['metrics'])}
Categories: {json.dumps(spec['categories'])}
Schema: {json.dumps(spec['example_schema'])}
Authors: {authors}
{f"Abstract focus: {abstract_focus}" if abstract_focus else ""}

Statistics:
{chr(10).join(stats_lines) if stats_lines else "No examples yet"}

Sample examples:
{json.dumps(sample_exs, indent=2) if sample_exs else "None"}

Sections: Abstract, Introduction, Benchmark Design, Dataset Description,
Example Analysis (table), Evaluation Metrics, Related Work, Conclusion.
Use standard article class with tables for stats.
Output ONLY LaTeX source starting with \\documentclass."""

    latex = await skill._generate(prompt, max_tokens=8000)

    # Strip markdown fences if present
    if latex.strip().startswith("```"):
        lines = latex.strip().split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        latex = "\n".join(lines)

    if "\\documentclass" not in latex:
        return SkillResult(success=False, message="LLM did not produce valid LaTeX",
                           data={"raw": latex[:500]})

    state["paper_latex"] = latex
    skill._save_state(project_name, state)

    tex_path = skill._workspace / project_name / "paper.tex"
    tex_path.write_text(latex)

    return SkillResult(
        success=True,
        message=f"Generated LaTeX paper ({len(latex)} chars) saved to {tex_path}",
        data={"project_name": project_name, "tex_path": str(tex_path), "length": len(latex)}
    )


async def submit_to_arxiv(skill, params: Dict) -> SkillResult:
    """Package LaTeX and submit via arXiv SWORD API (or return manual instructions)."""
    project_name = params.get("project_name")
    if not project_name:
        return SkillResult(success=False, message="Missing required parameter: project_name")

    state = skill._load_state(project_name)
    latex = state.get("paper_latex")
    if not latex:
        return SkillResult(success=False, message=f"No paper for '{project_name}'. Run generate_paper first.")

    spec = state.get("spec", {})
    categories = params.get("categories", "cs.CL")
    comments = params.get("comments", "")

    # Build tar.gz package
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        tex_bytes = latex.encode("utf-8")
        info = tarfile.TarInfo(name="paper.tex")
        info.size = len(tex_bytes)
        tar.addfile(info, io.BytesIO(tex_bytes))
    tar_data = tar_buffer.getvalue()

    pkg_path = skill._workspace / project_name / "arxiv_submission.tar.gz"
    pkg_path.write_bytes(tar_data)

    arxiv_user = skill.credentials.get("ARXIV_USERNAME") or os.environ.get("ARXIV_USERNAME")
    arxiv_pass = skill.credentials.get("ARXIV_PASSWORD") or os.environ.get("ARXIV_PASSWORD")

    if not arxiv_user or not arxiv_pass:
        title = spec.get("name", project_name).replace("_", " ").title()
        state["arxiv"] = {"status": "manual", "package": str(pkg_path)}
        skill._save_state(project_name, state)
        return SkillResult(
            success=True,
            message=f"No arXiv credentials. Package at {pkg_path}. Upload at https://arxiv.org/submit",
            data={"project_name": project_name, "package_path": str(pkg_path),
                  "instructions": f"Title: {title}, Categories: {categories}"}
        )

    # Submit via SWORD API
    import base64
    auth = base64.b64encode(f"{arxiv_user}:{arxiv_pass}".encode()).decode()
    title = spec.get("name", project_name).replace("_", " ").title()
    abstract = spec.get("description", "")

    atom = _build_atom_entry(title, abstract, categories, comments)
    resp = await skill.http.post(
        "https://arxiv.org/sword-app/",
        headers={"Authorization": f"Basic {auth}", "Content-Type": "application/atom+xml"},
        content=atom.encode(),
        
    )

    if resp.status_code in (200, 201, 202):
        state["arxiv"] = {"status": "submitted", "response": resp.text[:500]}
        skill._save_state(project_name, state)
        return SkillResult(success=True, message=f"Submitted to arXiv ({resp.status_code})",
                           data={"project_name": project_name, "status_code": resp.status_code})

    state["arxiv"] = {"status": "failed", "error": resp.text[:500], "package": str(pkg_path)}
    skill._save_state(project_name, state)
    return SkillResult(
        success=False,
        message=f"arXiv submission failed ({resp.status_code}). Package at {pkg_path}",
        data={"project_name": project_name, "error": resp.text[:300], "package_path": str(pkg_path)}
    )


async def get_status(skill, params: Dict) -> SkillResult:
    """Pipeline progress and next step suggestion."""
    project_name = params.get("project_name")
    if not project_name:
        return SkillResult(success=False, message="Missing required parameter: project_name")

    state = skill._load_state(project_name)
    spec = state.get("spec")

    steps = {
        "design_benchmark": spec is not None,
        "generate_examples": any(state["examples"].get(s) for s in ("train", "test", "val")),
        "validate_benchmark": state.get("validation") is not None,
        "publish_to_huggingface": state.get("huggingface") is not None,
        "generate_paper": state.get("paper_latex") is not None,
        "submit_to_arxiv": state.get("arxiv") is not None,
    }

    completed = [k for k, v in steps.items() if v]
    pending = [k for k, v in steps.items() if not v]
    next_step = pending[0] if pending else "all_done"
    total = sum(len(state["examples"].get(s, [])) for s in ("train", "test", "val"))
    target = spec.get("num_examples", 100) if spec else 0

    progress = {"completed_steps": completed, "pending_steps": pending,
                "next_step": next_step, "examples_generated": total, "examples_target": target}
    if state.get("huggingface"):
        progress["huggingface_url"] = state["huggingface"].get("url")
    if state.get("arxiv"):
        progress["arxiv_status"] = state["arxiv"].get("status")

    return SkillResult(
        success=True,
        message=f"{len(completed)}/6 steps done. Next: {next_step}. Examples: {total}/{target}",
        data={"project_name": project_name, "progress": progress}
    )


def _build_atom_entry(title, abstract, categories, comments):
    cat = categories.split()[0] if categories else "cs.CL"
    return f"""<?xml version="1.0" encoding="utf-8"?>
<entry xmlns="http://www.w3.org/2005/Atom">
  <title>{title}</title>
  <summary>{abstract}</summary>
  <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="{cat}"/>
  {f'<arxiv:comment xmlns:arxiv="http://arxiv.org/schemas/atom">{comments}</arxiv:comment>' if comments else ''}
</entry>"""
