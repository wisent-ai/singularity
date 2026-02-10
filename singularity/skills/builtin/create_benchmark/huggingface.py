"""
HuggingFace Hub publishing: create dataset repo, upload JSONL + dataset card.
"""

import base64
import json
import os
from typing import Dict

from singularity.skills.base import SkillResult


HF_API = "https://huggingface.co/api"


async def publish_to_huggingface(skill, params: Dict) -> SkillResult:
    """Create HF dataset repo, upload JSONL files per split, and a dataset card."""
    project_name = params.get("project_name")
    if not project_name:
        return SkillResult(success=False, message="Missing required parameter: project_name")

    state = skill._load_state(project_name)
    spec = state.get("spec")
    if not spec:
        return SkillResult(success=False, message=f"No spec for '{project_name}'. Run design_benchmark first.")

    total = sum(len(state["examples"].get(s, [])) for s in ("train", "test", "val"))
    if total == 0:
        return SkillResult(success=False, message="No examples to publish. Run generate_examples first.")

    hf_token = skill.credentials.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
    if not hf_token:
        return SkillResult(success=False, message="Missing HF_TOKEN credential")

    hf_repo = params.get("hf_repo_name", project_name)
    private = params.get("private", False)
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}

    # 1. Get username
    whoami = await skill.http.get(f"{HF_API}/whoami-v2", headers=headers)
    if whoami.status_code != 200:
        return SkillResult(success=False, message=f"HF auth failed: {whoami.text[:200]}")
    username = whoami.json().get("name", "")

    repo_id = f"{username}/{hf_repo}" if "/" not in hf_repo else hf_repo

    # 2. Create dataset repo
    create_resp = await skill.http.post(f"{HF_API}/repos/create", headers=headers, json={
        "type": "dataset", "name": hf_repo, "private": private
    })
    if create_resp.status_code not in (200, 201, 409):  # 409 = already exists
        return SkillResult(success=False, message=f"Failed to create repo: {create_resp.text[:200]}")

    # 3. Build files to upload
    files_to_upload = []

    for split in ("train", "test", "val"):
        examples = state["examples"].get(split, [])
        if not examples:
            continue
        jsonl = "\n".join(json.dumps(ex, ensure_ascii=False) for ex in examples)
        files_to_upload.append({
            "path": f"data/{split}.jsonl",
            "content": base64.b64encode(jsonl.encode()).decode()
        })

    # Generate dataset card
    card = _generate_dataset_card(spec, state, repo_id)
    files_to_upload.append({
        "path": "README.md",
        "content": base64.b64encode(card.encode()).decode()
    })

    # 4. Upload via commit API
    operations = [
        {"key": "file", "value": {"content": f["content"], "path": f["path"], "encoding": "base64"}}
        for f in files_to_upload
    ]

    commit_resp = await skill.http.post(
        f"{HF_API}/datasets/{repo_id}/commit/main",
        headers=headers,
        json={
            "summary": f"Upload {project_name} benchmark dataset",
            "operations": operations
        },
        
    )

    if commit_resp.status_code not in (200, 201):
        return SkillResult(success=False,
                           message=f"Failed to upload files: {commit_resp.status_code} {commit_resp.text[:200]}")

    dataset_url = f"https://huggingface.co/datasets/{repo_id}"
    state["huggingface"] = {"repo_id": repo_id, "url": dataset_url, "private": private}
    skill._save_state(project_name, state)

    return SkillResult(
        success=True,
        message=f"Published to HuggingFace: {dataset_url} ({total} examples, {len(files_to_upload) - 1} splits)",
        data={"project_name": project_name, "repo_id": repo_id, "url": dataset_url},
        asset_created=dataset_url
    )


def _generate_dataset_card(spec: Dict, state: Dict, repo_id: str) -> str:
    """Generate a HuggingFace dataset card (README.md)."""
    split_info = []
    for s in ("train", "test", "val"):
        count = len(state["examples"].get(s, []))
        if count:
            split_info.append(f"  - {s}: {count} examples")

    categories = ", ".join(spec.get("categories", []))
    metrics = ", ".join(spec.get("metrics", []))

    return f"""---
dataset_info:
  task_categories:
    - text-classification
  language:
    - en
  size_categories:
    - n<1K
license: mit
---

# {spec.get('name', repo_id)}

{spec.get('description', '')}

## Benchmark Details

- **Task type**: {spec.get('task_type', 'N/A')}
- **Metrics**: {metrics}
- **Categories**: {categories}

## Dataset Splits

{chr(10).join(split_info)}

## Example Schema

```json
{json.dumps(spec.get('example_schema', {}), indent=2)}
```

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
```

## Citation

If you use this benchmark, please cite it as:

```bibtex
@misc{{{spec.get('name', 'benchmark')},
  title={{{spec.get('name', 'Benchmark')}}},
  year={{2025}},
  publisher={{HuggingFace}},
  url={{https://huggingface.co/datasets/{repo_id}}}
}}
```
"""
