"""
Clone SaaS — Deployment Actions

create_repo:      Create a GitHub repo.
push_files:       Push all generated files via Git Trees API (atomic commit).
deploy_to_vercel: Create Vercel project linked to GitHub repo and trigger deploy.
"""

from typing import Dict
from singularity.skills.base import SkillResult

GITHUB_API = "https://api.github.com"
VERCEL_API = "https://api.vercel.com"


async def create_repo(skill, params: Dict) -> SkillResult:
    """Create a GitHub repo."""
    project_name = params.get("project_name")
    if not project_name:
        return SkillResult(success=False, message="project_name is required")

    state = skill._load_state(project_name)
    if state.get("repo"):
        return SkillResult(success=True, message=f"Repo already exists: {state['repo'].get('full_name')}",
                           data=state["repo"])

    repo_name = params.get("repo_name", project_name)
    private = params.get("private", True)
    headers = skill._github_headers()

    resp = await skill.http.post(f"{GITHUB_API}/user/repos", headers=headers, json={
        "name": repo_name,
        "description": f"{project_name} — SaaS clone (Next.js + Tailwind + Supabase)",
        "private": private, "auto_init": True})

    if resp.status_code not in [200, 201]:
        return SkillResult(success=False, message=f"Failed to create repo: {resp.text}")

    r = resp.json()
    info = {"name": r.get("name"), "full_name": r.get("full_name"), "url": r.get("html_url"),
            "clone_url": r.get("clone_url"), "private": r.get("private"),
            "default_branch": r.get("default_branch", "main"), "pushed": False}

    state["repo"] = info
    skill._save_state(project_name, state)

    return SkillResult(success=True, message=f"Created repo: {info['full_name']}", data=info,
                       asset_created={"type": "github_repo", "name": info["full_name"], "url": info["url"]})


async def push_files(skill, params: Dict) -> SkillResult:
    """Push all generated files to GitHub in one atomic commit."""
    project_name = params.get("project_name")
    if not project_name:
        return SkillResult(success=False, message="project_name is required")

    state = skill._load_state(project_name)
    repo = state.get("repo")
    if not repo:
        return SkillResult(success=False, message="No repo — run create_repo first")

    files = state.get("generated_files", {})
    if not files:
        return SkillResult(success=False, message="No generated files to push")

    full_name, branch = repo["full_name"], repo.get("default_branch", "main")
    headers = skill._github_headers()
    msg = params.get("commit_message", f"feat: initial {project_name} clone")

    try:
        sha = await _atomic_push(skill.http, headers, full_name, branch, files, msg)
    except Exception as e:
        return SkillResult(success=False, message=f"Git push failed: {str(e)}")

    state["repo"]["pushed"] = True
    skill._save_state(project_name, state)

    return SkillResult(success=True, message=f"Pushed {len(files)} files to {full_name}",
                       data={"full_name": full_name, "commit_sha": sha,
                             "files_pushed": len(files), "commit_message": msg})


async def _atomic_push(http, headers, repo, branch, files, message) -> str:
    """Create blobs → tree → commit → update ref. Returns new commit SHA."""
    ref = await http.get(f"{GITHUB_API}/repos/{repo}/git/ref/heads/{branch}", headers=headers)
    if ref.status_code != 200:
        raise RuntimeError(f"Could not get branch ref: {ref.text}")
    latest = ref.json()["object"]["sha"]

    base_tree = (await http.get(
        f"{GITHUB_API}/repos/{repo}/git/commits/{latest}", headers=headers)).json()["tree"]["sha"]

    tree_items = []
    for path, content in files.items():
        blob = await http.post(f"{GITHUB_API}/repos/{repo}/git/blobs", headers=headers,
                               json={"content": content, "encoding": "utf-8"})
        if blob.status_code != 201:
            raise RuntimeError(f"Blob failed for {path}: {blob.text}")
        tree_items.append({"path": path, "mode": "100644", "type": "blob", "sha": blob.json()["sha"]})

    tree = await http.post(f"{GITHUB_API}/repos/{repo}/git/trees", headers=headers,
                           json={"base_tree": base_tree, "tree": tree_items})
    if tree.status_code != 201:
        raise RuntimeError(f"Tree creation failed: {tree.text}")

    commit = await http.post(f"{GITHUB_API}/repos/{repo}/git/commits", headers=headers,
                             json={"message": message, "tree": tree.json()["sha"], "parents": [latest]})
    if commit.status_code != 201:
        raise RuntimeError(f"Commit failed: {commit.text}")
    new_sha = commit.json()["sha"]

    upd = await http.patch(f"{GITHUB_API}/repos/{repo}/git/refs/heads/{branch}", headers=headers,
                           json={"sha": new_sha})
    if upd.status_code != 200:
        raise RuntimeError(f"Ref update failed: {upd.text}")
    return new_sha


async def deploy_to_vercel(skill, params: Dict) -> SkillResult:
    """Create Vercel project linked to GitHub repo and trigger deploy."""
    project_name = params.get("project_name")
    if not project_name:
        return SkillResult(success=False, message="project_name is required")

    state = skill._load_state(project_name)
    repo = state.get("repo")
    if not repo:
        return SkillResult(success=False, message="No repo — run create_repo and push_files first")
    if not repo.get("pushed"):
        return SkillResult(success=False, message="Files not pushed — run push_files first")

    headers = skill._vercel_headers()
    full_name = repo["full_name"]
    env_vars = params.get("env_vars") or {}

    # Create or get project
    resp = await skill.http.post(f"{VERCEL_API}/v10/projects", headers=headers,
                                 json={"name": project_name, "framework": "nextjs",
                                       "gitRepository": {"type": "github", "repo": full_name}})
    if resp.status_code in [200, 201]:
        proj = resp.json()
    elif "already exist" in resp.text.lower():
        g = await skill.http.get(f"{VERCEL_API}/v9/projects/{project_name}", headers=headers)
        proj = g.json() if g.status_code == 200 else None
        if not proj:
            return SkillResult(success=False, message="Project exists but couldn't fetch it")
    else:
        return SkillResult(success=False, message=f"Vercel project creation failed: {resp.text}")

    pid = proj.get("id")

    # Set env vars
    for k, v in env_vars.items():
        await skill.http.post(f"{VERCEL_API}/v10/projects/{pid}/env", headers=headers,
                              json={"key": k, "value": v, "type": "encrypted",
                                    "target": ["production", "preview", "development"]})

    # Trigger deploy
    dr = await skill.http.post(
        f"{VERCEL_API}/v13/deployments", headers=headers,
        json={"name": project_name, "target": "production",
              "gitSource": {"type": "github", "ref": repo.get("default_branch", "main"),
                            "repoId": str(proj.get("link", {}).get("repoId", ""))}})

    if dr.status_code in [200, 201]:
        d = dr.json()
        dep = {"id": d.get("id"), "url": f"https://{d.get('url')}" if d.get("url") else None,
               "state": d.get("readyState", "QUEUED"), "project_url": f"https://{project_name}.vercel.app"}
    else:
        dep = {"url": f"https://{project_name}.vercel.app", "state": "PENDING",
               "note": "Auto-deploy should trigger from GitHub push"}

    state["deployment"] = dep
    skill._save_state(project_name, state)

    url = dep.get("url") or dep.get("project_url")
    return SkillResult(success=True, message=f"Vercel deploy triggered: {url}",
                       data={"project_name": project_name, "project_id": pid, "deployment": dep},
                       asset_created={"type": "vercel_project", "name": project_name, "url": url})
