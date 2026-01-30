#!/usr/bin/env python3
"""
GitHub Skill

Enables agents to:
- Create repositories
- Create/manage issues
- Create pull requests
- Search code/repos
- Fork repositories
"""

import httpx
from typing import Dict, List, Optional
from .base import Skill, SkillResult, SkillManifest, SkillAction


class GitHubSkill(Skill):
    """
    Skill for GitHub API interactions.

    Required credentials:
    - GITHUB_TOKEN: Personal access token with appropriate scopes
    """

    API_BASE = "https://api.github.com"

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="github",
            name="GitHub Management",
            version="1.0.0",
            category="dev",
            description="Create repos, manage issues, and interact with GitHub",
            required_credentials=["GITHUB_TOKEN"],
            install_cost=0,
            actions=[
                SkillAction(
                    name="create_repo",
                    description="Create a new GitHub repository",
                    parameters={
                        "name": {"type": "string", "required": True, "description": "Repository name"},
                        "description": {"type": "string", "required": False, "description": "Repo description"},
                        "private": {"type": "boolean", "required": False, "description": "Private repo? (default: false)"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=5,
                    success_probability=0.95
                ),
                SkillAction(
                    name="create_issue",
                    description="Create an issue in a repository",
                    parameters={
                        "repo": {"type": "string", "required": True, "description": "Repo (owner/name)"},
                        "title": {"type": "string", "required": True, "description": "Issue title"},
                        "body": {"type": "string", "required": False, "description": "Issue body"},
                        "labels": {"type": "array", "required": False, "description": "Labels to add"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=3,
                    success_probability=0.95
                ),
                SkillAction(
                    name="search_repos",
                    description="Search GitHub repositories",
                    parameters={
                        "query": {"type": "string", "required": True, "description": "Search query"},
                        "sort": {"type": "string", "required": False, "description": "Sort by (stars, forks, updated)"},
                        "limit": {"type": "integer", "required": False, "description": "Max results"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=5,
                    success_probability=0.9
                ),
                SkillAction(
                    name="search_issues",
                    description="Search GitHub issues",
                    parameters={
                        "query": {"type": "string", "required": True, "description": "Search query"},
                        "state": {"type": "string", "required": False, "description": "State (open, closed, all)"},
                        "labels": {"type": "string", "required": False, "description": "Label filter"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=5,
                    success_probability=0.9
                ),
                SkillAction(
                    name="fork_repo",
                    description="Fork a repository",
                    parameters={
                        "repo": {"type": "string", "required": True, "description": "Repo to fork (owner/name)"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=10,
                    success_probability=0.9
                ),
                SkillAction(
                    name="star_repo",
                    description="Star a repository",
                    parameters={
                        "repo": {"type": "string", "required": True, "description": "Repo to star (owner/name)"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.95
                ),
                SkillAction(
                    name="get_user",
                    description="Get GitHub user info",
                    parameters={
                        "username": {"type": "string", "required": False, "description": "Username (default: authenticated user)"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.95
                ),
                SkillAction(
                    name="create_gist",
                    description="Create a GitHub Gist",
                    parameters={
                        "description": {"type": "string", "required": False, "description": "Gist description"},
                        "files": {"type": "object", "required": True, "description": "Files {filename: content}"},
                        "public": {"type": "boolean", "required": False, "description": "Public gist? (default: true)"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=3,
                    success_probability=0.95
                )
            ]
        )

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self.http = httpx.AsyncClient(timeout=30)

    def _get_headers(self) -> Dict:
        """Get authorization headers"""
        return {
            "Authorization": f"Bearer {self.credentials.get('GITHUB_TOKEN')}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }

    async def execute(self, action: str, params: Dict) -> SkillResult:
        """Execute a GitHub action"""
        if not self.check_credentials():
            missing = self.get_missing_credentials()
            return SkillResult(
                success=False,
                message=f"Missing credentials: {missing}"
            )

        try:
            if action == "create_repo":
                return await self._create_repo(
                    params.get("name"),
                    params.get("description"),
                    params.get("private", False)
                )
            elif action == "create_issue":
                return await self._create_issue(
                    params.get("repo"),
                    params.get("title"),
                    params.get("body"),
                    params.get("labels", [])
                )
            elif action == "search_repos":
                return await self._search_repos(
                    params.get("query"),
                    params.get("sort"),
                    params.get("limit", 10)
                )
            elif action == "search_issues":
                return await self._search_issues(
                    params.get("query"),
                    params.get("state", "open"),
                    params.get("labels")
                )
            elif action == "fork_repo":
                return await self._fork_repo(params.get("repo"))
            elif action == "star_repo":
                return await self._star_repo(params.get("repo"))
            elif action == "get_user":
                return await self._get_user(params.get("username"))
            elif action == "create_gist":
                return await self._create_gist(
                    params.get("description"),
                    params.get("files"),
                    params.get("public", True)
                )
            else:
                return SkillResult(
                    success=False,
                    message=f"Unknown action: {action}"
                )
        except Exception as e:
            return SkillResult(
                success=False,
                message=f"GitHub error: {str(e)}"
            )

    async def _create_repo(
        self,
        name: str,
        description: str = None,
        private: bool = False
    ) -> SkillResult:
        """Create a new repository"""
        if not name:
            return SkillResult(success=False, message="Repository name required")

        data = {
            "name": name,
            "description": description or "",
            "private": private,
            "auto_init": True
        }

        response = await self.http.post(
            f"{self.API_BASE}/user/repos",
            headers=self._get_headers(),
            json=data
        )

        if response.status_code == 201:
            repo = response.json()
            return SkillResult(
                success=True,
                message=f"Created repository: {repo.get('full_name')}",
                data={
                    "name": repo.get("name"),
                    "full_name": repo.get("full_name"),
                    "url": repo.get("html_url"),
                    "clone_url": repo.get("clone_url"),
                    "private": repo.get("private")
                },
                asset_created={
                    "type": "github_repo",
                    "name": repo.get("full_name"),
                    "url": repo.get("html_url")
                }
            )
        else:
            return SkillResult(
                success=False,
                message=f"Failed to create repo: {response.text}"
            )

    async def _create_issue(
        self,
        repo: str,
        title: str,
        body: str = None,
        labels: List[str] = None
    ) -> SkillResult:
        """Create an issue"""
        if not repo or not title:
            return SkillResult(success=False, message="Repo and title required")

        data = {
            "title": title,
            "body": body or "",
            "labels": labels or []
        }

        response = await self.http.post(
            f"{self.API_BASE}/repos/{repo}/issues",
            headers=self._get_headers(),
            json=data
        )

        if response.status_code == 201:
            issue = response.json()
            return SkillResult(
                success=True,
                message=f"Created issue #{issue.get('number')}: {title}",
                data={
                    "number": issue.get("number"),
                    "title": issue.get("title"),
                    "url": issue.get("html_url"),
                    "state": issue.get("state")
                }
            )
        else:
            return SkillResult(
                success=False,
                message=f"Failed to create issue: {response.text}"
            )

    async def _search_repos(
        self,
        query: str,
        sort: str = None,
        limit: int = 10
    ) -> SkillResult:
        """Search repositories"""
        if not query:
            return SkillResult(success=False, message="Search query required")

        params = {"q": query, "per_page": min(limit, 100)}
        if sort:
            params["sort"] = sort

        response = await self.http.get(
            f"{self.API_BASE}/search/repositories",
            headers=self._get_headers(),
            params=params
        )

        if response.status_code == 200:
            data = response.json()
            repos = [
                {
                    "name": r.get("full_name"),
                    "description": r.get("description"),
                    "url": r.get("html_url"),
                    "stars": r.get("stargazers_count"),
                    "forks": r.get("forks_count"),
                    "language": r.get("language")
                }
                for r in data.get("items", [])[:limit]
            ]
            return SkillResult(
                success=True,
                message=f"Found {len(repos)} repositories",
                data={"repos": repos, "total": data.get("total_count")}
            )
        else:
            return SkillResult(
                success=False,
                message=f"Search failed: {response.text}"
            )

    async def _search_issues(
        self,
        query: str,
        state: str = "open",
        labels: str = None
    ) -> SkillResult:
        """Search issues"""
        if not query:
            return SkillResult(success=False, message="Search query required")

        q = f"{query} is:issue state:{state}"
        if labels:
            q += f" label:{labels}"

        response = await self.http.get(
            f"{self.API_BASE}/search/issues",
            headers=self._get_headers(),
            params={"q": q, "per_page": 30}
        )

        if response.status_code == 200:
            data = response.json()
            issues = [
                {
                    "title": i.get("title"),
                    "url": i.get("html_url"),
                    "repo": i.get("repository_url", "").split("/")[-1],
                    "state": i.get("state"),
                    "labels": [l.get("name") for l in i.get("labels", [])]
                }
                for i in data.get("items", [])
            ]
            return SkillResult(
                success=True,
                message=f"Found {len(issues)} issues",
                data={"issues": issues, "total": data.get("total_count")}
            )
        else:
            return SkillResult(
                success=False,
                message=f"Search failed: {response.text}"
            )

    async def _fork_repo(self, repo: str) -> SkillResult:
        """Fork a repository"""
        if not repo:
            return SkillResult(success=False, message="Repository required")

        response = await self.http.post(
            f"{self.API_BASE}/repos/{repo}/forks",
            headers=self._get_headers()
        )

        if response.status_code in [200, 202]:
            fork = response.json()
            return SkillResult(
                success=True,
                message=f"Forked {repo}",
                data={
                    "name": fork.get("full_name"),
                    "url": fork.get("html_url"),
                    "clone_url": fork.get("clone_url")
                },
                asset_created={
                    "type": "github_fork",
                    "name": fork.get("full_name"),
                    "url": fork.get("html_url")
                }
            )
        else:
            return SkillResult(
                success=False,
                message=f"Failed to fork: {response.text}"
            )

    async def _star_repo(self, repo: str) -> SkillResult:
        """Star a repository"""
        if not repo:
            return SkillResult(success=False, message="Repository required")

        response = await self.http.put(
            f"{self.API_BASE}/user/starred/{repo}",
            headers=self._get_headers()
        )

        if response.status_code == 204:
            return SkillResult(
                success=True,
                message=f"Starred {repo}",
                data={"repo": repo, "starred": True}
            )
        else:
            return SkillResult(
                success=False,
                message=f"Failed to star: {response.text}"
            )

    async def _get_user(self, username: str = None) -> SkillResult:
        """Get user information"""
        if username:
            url = f"{self.API_BASE}/users/{username}"
        else:
            url = f"{self.API_BASE}/user"

        response = await self.http.get(url, headers=self._get_headers())

        if response.status_code == 200:
            user = response.json()
            return SkillResult(
                success=True,
                message=f"Got user info for {user.get('login')}",
                data={
                    "login": user.get("login"),
                    "name": user.get("name"),
                    "bio": user.get("bio"),
                    "url": user.get("html_url"),
                    "followers": user.get("followers"),
                    "following": user.get("following"),
                    "public_repos": user.get("public_repos")
                }
            )
        else:
            return SkillResult(
                success=False,
                message=f"Failed to get user: {response.text}"
            )

    async def _create_gist(
        self,
        description: str,
        files: Dict[str, str],
        public: bool = True
    ) -> SkillResult:
        """Create a gist"""
        if not files:
            return SkillResult(success=False, message="Files required")

        data = {
            "description": description or "",
            "public": public,
            "files": {
                name: {"content": content}
                for name, content in files.items()
            }
        }

        response = await self.http.post(
            f"{self.API_BASE}/gists",
            headers=self._get_headers(),
            json=data
        )

        if response.status_code == 201:
            gist = response.json()
            return SkillResult(
                success=True,
                message=f"Created gist: {gist.get('id')}",
                data={
                    "id": gist.get("id"),
                    "url": gist.get("html_url"),
                    "files": list(gist.get("files", {}).keys()),
                    "public": gist.get("public")
                },
                asset_created={
                    "type": "gist",
                    "id": gist.get("id"),
                    "url": gist.get("html_url")
                }
            )
        else:
            return SkillResult(
                success=False,
                message=f"Failed to create gist: {response.text}"
            )

    async def close(self):
        """Clean up"""
        await self.http.aclose()
