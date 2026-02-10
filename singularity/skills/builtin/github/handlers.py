"""Handler functions for GitHubSkill actions."""

from typing import List, Optional
from singularity.skills.base import SkillResult


async def create_repo(skill, name: str, description: str = None, private: bool = False) -> SkillResult:
    if not name:
        return SkillResult(success=False, message="Repository name required")
    data = {"name": name, "description": description or "", "private": private, "auto_init": True}
    response = await skill.http.post(f"{skill.API_BASE}/user/repos", headers=skill._get_headers(), json=data)
    if response.status_code == 201:
        repo = response.json()
        return SkillResult(success=True, message=f"Created repository: {repo.get('full_name')}",
            data={"name": repo.get("name"), "full_name": repo.get("full_name"),
                  "url": repo.get("html_url"), "clone_url": repo.get("clone_url"),
                  "private": repo.get("private")},
            asset_created={"type": "github_repo", "name": repo.get("full_name"), "url": repo.get("html_url")})
    return SkillResult(success=False, message=f"Failed to create repo: {response.text}")


async def create_issue(skill, repo: str, title: str, body: str = None,
                       labels: List[str] = None) -> SkillResult:
    if not repo or not title:
        return SkillResult(success=False, message="Repo and title required")
    data = {"title": title, "body": body or "", "labels": labels or []}
    response = await skill.http.post(f"{skill.API_BASE}/repos/{repo}/issues",
        headers=skill._get_headers(), json=data)
    if response.status_code == 201:
        issue = response.json()
        return SkillResult(success=True, message=f"Created issue #{issue.get('number')}: {title}",
            data={"number": issue.get("number"), "title": issue.get("title"),
                  "url": issue.get("html_url"), "state": issue.get("state")})
    return SkillResult(success=False, message=f"Failed to create issue: {response.text}")


async def search_repos(skill, query: str, sort: str = None, limit: int = 10) -> SkillResult:
    if not query:
        return SkillResult(success=False, message="Search query required")
    params = {"q": query, "per_page": min(limit, 100)}
    if sort:
        params["sort"] = sort
    response = await skill.http.get(f"{skill.API_BASE}/search/repositories",
        headers=skill._get_headers(), params=params)
    if response.status_code == 200:
        data = response.json()
        repos = [{"name": r.get("full_name"), "description": r.get("description"),
                  "url": r.get("html_url"), "stars": r.get("stargazers_count"),
                  "forks": r.get("forks_count"), "language": r.get("language")}
                 for r in data.get("items", [])[:limit]]
        return SkillResult(success=True, message=f"Found {len(repos)} repositories",
            data={"repos": repos, "total": data.get("total_count")})
    return SkillResult(success=False, message=f"Search failed: {response.text}")


async def search_issues(skill, query: str, state: str = "open", labels: str = None) -> SkillResult:
    if not query:
        return SkillResult(success=False, message="Search query required")
    q = f"{query} is:issue state:{state}"
    if labels:
        q += f" label:{labels}"
    response = await skill.http.get(f"{skill.API_BASE}/search/issues",
        headers=skill._get_headers(), params={"q": q, "per_page": 30})
    if response.status_code == 200:
        data = response.json()
        issues = [{"title": i.get("title"), "url": i.get("html_url"),
                   "repo": i.get("repository_url", "").split("/")[-1],
                   "state": i.get("state"),
                   "labels": [l.get("name") for l in i.get("labels", [])]}
                  for i in data.get("items", [])]
        return SkillResult(success=True, message=f"Found {len(issues)} issues",
            data={"issues": issues, "total": data.get("total_count")})
    return SkillResult(success=False, message=f"Search failed: {response.text}")


async def fork_repo(skill, repo: str) -> SkillResult:
    if not repo:
        return SkillResult(success=False, message="Repository required")
    response = await skill.http.post(f"{skill.API_BASE}/repos/{repo}/forks", headers=skill._get_headers())
    if response.status_code in [200, 202]:
        fork = response.json()
        return SkillResult(success=True, message=f"Forked {repo}",
            data={"name": fork.get("full_name"), "url": fork.get("html_url"),
                  "clone_url": fork.get("clone_url")},
            asset_created={"type": "github_fork", "name": fork.get("full_name"), "url": fork.get("html_url")})
    return SkillResult(success=False, message=f"Failed to fork: {response.text}")


async def star_repo(skill, repo: str) -> SkillResult:
    if not repo:
        return SkillResult(success=False, message="Repository required")
    response = await skill.http.put(f"{skill.API_BASE}/user/starred/{repo}", headers=skill._get_headers())
    if response.status_code == 204:
        return SkillResult(success=True, message=f"Starred {repo}", data={"repo": repo, "starred": True})
    return SkillResult(success=False, message=f"Failed to star: {response.text}")


async def get_user(skill, username: str = None) -> SkillResult:
    url = f"{skill.API_BASE}/users/{username}" if username else f"{skill.API_BASE}/user"
    response = await skill.http.get(url, headers=skill._get_headers())
    if response.status_code == 200:
        user = response.json()
        return SkillResult(success=True, message=f"Got user info for {user.get('login')}",
            data={"login": user.get("login"), "name": user.get("name"), "bio": user.get("bio"),
                  "url": user.get("html_url"), "followers": user.get("followers"),
                  "following": user.get("following"), "public_repos": user.get("public_repos")})
    return SkillResult(success=False, message=f"Failed to get user: {response.text}")


async def create_gist(skill, description: str, files: dict, public: bool = True) -> SkillResult:
    if not files:
        return SkillResult(success=False, message="Files required")
    data = {"description": description or "", "public": public,
            "files": {name: {"content": content} for name, content in files.items()}}
    response = await skill.http.post(f"{skill.API_BASE}/gists", headers=skill._get_headers(), json=data)
    if response.status_code == 201:
        gist = response.json()
        return SkillResult(success=True, message=f"Created gist: {gist.get('id')}",
            data={"id": gist.get("id"), "url": gist.get("html_url"),
                  "files": list(gist.get("files", {}).keys()), "public": gist.get("public")},
            asset_created={"type": "gist", "id": gist.get("id"), "url": gist.get("html_url")})
    return SkillResult(success=False, message=f"Failed to create gist: {response.text}")
