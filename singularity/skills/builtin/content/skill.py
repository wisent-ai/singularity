#!/usr/bin/env python3
"""
Content Creation Skill

Enables agents to generate content using LLM:
- Blog posts / articles
- Social media posts
- Marketing copy
- Code snippets
- Documentation
"""

import json
from typing import Dict, List, Optional
from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction

try:
    from anthropic import AsyncAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class ContentCreationSkill(Skill):
    """
    Skill for AI-powered content creation.

    Uses LLM (Claude or GPT) to generate various types of content.
    Can use agent's existing LLM or initialize its own.

    Required credentials (one of, unless LLM provided):
    - ANTHROPIC_API_KEY: Anthropic API key (preferred)
    - OPENAI_API_KEY: OpenAI API key
    """

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="content_creation",
            name="AI Content Creation",
            version="1.0.0",
            category="content",
            description="Generate content using AI (articles, social posts, code, etc.)",
            required_credentials=[],  # Optional if LLM injected
            install_cost=0,
            actions=[
                SkillAction(
                    name="write_article",
                    description="Write a blog post or article",
                    parameters={
                        "topic": {"type": "string", "required": True, "description": "Article topic"},
                        "style": {"type": "string", "required": False, "description": "Writing style (professional, casual, technical)"},
                        "length": {"type": "string", "required": False, "description": "Length (short, medium, long)"},
                        "keywords": {"type": "array", "required": False, "description": "SEO keywords to include"}
                    },
                    estimated_cost=0.05,
                    estimated_duration_seconds=30,
                    success_probability=0.95
                ),
                SkillAction(
                    name="write_social_post",
                    description="Write a social media post",
                    parameters={
                        "platform": {"type": "string", "required": True, "description": "Platform (twitter, linkedin, facebook)"},
                        "topic": {"type": "string", "required": True, "description": "Post topic"},
                        "tone": {"type": "string", "required": False, "description": "Tone (professional, casual, humorous)"},
                        "include_hashtags": {"type": "boolean", "required": False, "description": "Include hashtags?"}
                    },
                    estimated_cost=0.01,
                    estimated_duration_seconds=10,
                    success_probability=0.95
                ),
                SkillAction(
                    name="write_marketing_copy",
                    description="Write marketing or sales copy",
                    parameters={
                        "product": {"type": "string", "required": True, "description": "Product/service name"},
                        "type": {"type": "string", "required": True, "description": "Type (landing_page, email, ad)"},
                        "target_audience": {"type": "string", "required": False, "description": "Target audience"},
                        "cta": {"type": "string", "required": False, "description": "Call to action"}
                    },
                    estimated_cost=0.03,
                    estimated_duration_seconds=20,
                    success_probability=0.95
                ),
                SkillAction(
                    name="write_code",
                    description="Generate code snippet",
                    parameters={
                        "description": {"type": "string", "required": True, "description": "What the code should do"},
                        "language": {"type": "string", "required": True, "description": "Programming language"},
                        "include_comments": {"type": "boolean", "required": False, "description": "Include comments?"}
                    },
                    estimated_cost=0.02,
                    estimated_duration_seconds=15,
                    success_probability=0.9
                ),
                SkillAction(
                    name="write_documentation",
                    description="Generate documentation",
                    parameters={
                        "code": {"type": "string", "required": True, "description": "Code to document"},
                        "type": {"type": "string", "required": False, "description": "Type (readme, api, tutorial)"}
                    },
                    estimated_cost=0.03,
                    estimated_duration_seconds=20,
                    success_probability=0.9
                ),
                SkillAction(
                    name="rewrite_content",
                    description="Rewrite or improve existing content",
                    parameters={
                        "content": {"type": "string", "required": True, "description": "Content to rewrite"},
                        "goal": {"type": "string", "required": False, "description": "Goal (improve_clarity, make_shorter, change_tone)"}
                    },
                    estimated_cost=0.02,
                    estimated_duration_seconds=15,
                    success_probability=0.95
                ),
                SkillAction(
                    name="summarize",
                    description="Summarize text content",
                    parameters={
                        "content": {"type": "string", "required": True, "description": "Content to summarize"},
                        "length": {"type": "string", "required": False, "description": "Summary length (brief, medium, detailed)"}
                    },
                    estimated_cost=0.01,
                    estimated_duration_seconds=10,
                    success_probability=0.95
                )
            ]
        )

    def __init__(self, credentials: Dict[str, str] = None, llm=None, llm_type: str = None, model: str = None):
        """
        Initialize content skill.

        Args:
            credentials: API credentials (optional if llm provided)
            llm: Existing LLM client to reuse (AsyncAnthropic or AsyncOpenAI)
            llm_type: Type of LLM ("anthropic" or "openai")
            model: Model name to use
        """
        super().__init__(credentials)

        # Use injected LLM or create own
        if llm:
            self.llm = llm
            self.llm_type = llm_type or "anthropic"
            self.model = model or "claude-sonnet-4-20250514"
        else:
            self._init_llm()

    def _init_llm(self):
        """Initialize own LLM client from credentials"""
        if self.credentials.get("ANTHROPIC_API_KEY") and HAS_ANTHROPIC:
            self.llm = AsyncAnthropic(api_key=self.credentials["ANTHROPIC_API_KEY"])
            self.llm_type = "anthropic"
            self.model = "claude-sonnet-4-20250514"
        elif self.credentials.get("OPENAI_API_KEY") and HAS_OPENAI:
            self.llm = openai.AsyncOpenAI(api_key=self.credentials["OPENAI_API_KEY"])
            self.llm_type = "openai"
            self.model = "gpt-4o"
        else:
            self.llm = None
            self.llm_type = "none"

    def set_llm(self, llm, llm_type: str, model: str):
        """Inject LLM after initialization"""
        self.llm = llm
        self.llm_type = llm_type
        self.model = model

    def check_credentials(self) -> bool:
        """Check if LLM is available"""
        return self.llm_type != "none"

    async def execute(self, action: str, params: Dict) -> SkillResult:
        """Execute a content creation action"""
        from . import generators

        if not self.check_credentials():
            return SkillResult(
                success=False,
                message="No LLM provider configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY."
            )

        try:
            if action == "write_article":
                return await generators.write_article(
                    self, params.get("topic"), params.get("style", "professional"),
                    params.get("length", "medium"), params.get("keywords", [])
                )
            elif action == "write_social_post":
                return await generators.write_social_post(
                    self, params.get("platform"), params.get("topic"),
                    params.get("tone", "professional"), params.get("include_hashtags", True)
                )
            elif action == "write_marketing_copy":
                return await generators.write_marketing_copy(
                    self, params.get("product"), params.get("type"),
                    params.get("target_audience"), params.get("cta")
                )
            elif action == "write_code":
                return await generators.write_code(
                    self, params.get("description"), params.get("language"),
                    params.get("include_comments", True)
                )
            elif action == "write_documentation":
                return await generators.write_documentation(
                    self, params.get("code"), params.get("type", "readme")
                )
            elif action == "rewrite_content":
                return await generators.rewrite_content(
                    self, params.get("content"), params.get("goal", "improve_clarity")
                )
            elif action == "summarize":
                return await generators.summarize(
                    self, params.get("content"), params.get("length", "brief")
                )
            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, message=f"Content creation error: {str(e)}")

    async def _generate(self, prompt: str, system: str = None) -> str:
        """Generate text using LLM"""
        if self.llm_type == "anthropic":
            response = await self.llm.messages.create(
                model=self.model,
                max_tokens=4000,
                system=system or "You are a skilled content creator.",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        elif self.llm_type == "openai":
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = await self.llm.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=4000
            )
            return response.choices[0].message.content
        else:
            raise RuntimeError("No LLM available")
