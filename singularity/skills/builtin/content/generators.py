"""
Content Creation Skill - Generator Methods

Contains the individual content generation methods for ContentCreationSkill.
"""

from typing import List
from singularity.skills.base import SkillResult


async def write_article(skill, topic: str, style: str, length: str, keywords: List[str]) -> SkillResult:
    """Write an article"""
    if not topic:
        return SkillResult(success=False, message="Topic required")

    length_guide = {
        "short": "500-800 words",
        "medium": "1000-1500 words",
        "long": "2000-3000 words"
    }

    keywords_str = ", ".join(keywords) if keywords else "none specified"

    prompt = f"""Write a {style} article about: {topic}

Length: {length_guide.get(length, '1000-1500 words')}
Keywords to include naturally: {keywords_str}

Include:
- Engaging headline
- Clear introduction
- Well-structured body with subheadings
- Conclusion with key takeaways

Write the complete article now."""

    content = await skill._generate(
        prompt,
        system="You are an expert content writer who creates engaging, well-researched articles."
    )

    return SkillResult(
        success=True,
        message=f"Article written: {topic}",
        data={
            "content": content,
            "topic": topic,
            "style": style,
            "length": length,
            "word_count": len(content.split())
        },
        cost=0.05,
        asset_created={"type": "article", "topic": topic, "content": content[:500]}
    )


async def write_social_post(skill, platform: str, topic: str, tone: str, include_hashtags: bool) -> SkillResult:
    """Write a social media post"""
    if not platform or not topic:
        return SkillResult(success=False, message="Platform and topic required")

    char_limits = {"twitter": 280, "linkedin": 3000, "facebook": 500, "instagram": 2200}
    limit = char_limits.get(platform.lower(), 500)

    prompt = f"""Write a {tone} {platform} post about: {topic}

Character limit: {limit}
{'Include relevant hashtags' if include_hashtags else 'No hashtags'}

Make it engaging and shareable. Write only the post content, nothing else."""

    content = await skill._generate(
        prompt,
        system=f"You are a social media expert who creates viral {platform} content."
    )

    if len(content) > limit:
        content = content[:limit-3] + "..."

    return SkillResult(
        success=True,
        message=f"{platform} post created",
        data={"content": content, "platform": platform, "character_count": len(content), "limit": limit},
        cost=0.01,
        asset_created={"type": "social_post", "platform": platform, "content": content}
    )


async def write_marketing_copy(skill, product: str, copy_type: str, target_audience: str, cta: str) -> SkillResult:
    """Write marketing copy"""
    if not product or not copy_type:
        return SkillResult(success=False, message="Product and type required")

    type_instructions = {
        "landing_page": "Write compelling landing page copy with headline, subheadline, benefits, and CTA",
        "email": "Write a marketing email with subject line, preview text, body, and CTA",
        "ad": "Write short, punchy ad copy with headline and description"
    }

    prompt = f"""{type_instructions.get(copy_type, 'Write marketing copy')} for:

Product/Service: {product}
Target Audience: {target_audience or 'General'}
Call to Action: {cta or 'Learn More'}

Focus on benefits, not features. Create urgency. Be persuasive."""

    content = await skill._generate(
        prompt, system="You are a world-class copywriter who writes copy that converts."
    )

    return SkillResult(
        success=True,
        message=f"Marketing copy created: {copy_type}",
        data={"content": content, "type": copy_type, "product": product},
        cost=0.03,
        asset_created={"type": "marketing_copy", "copy_type": copy_type, "product": product}
    )


async def write_code(skill, description: str, language: str, include_comments: bool) -> SkillResult:
    """Generate code"""
    if not description or not language:
        return SkillResult(success=False, message="Description and language required")

    prompt = f"""Write {language} code that does the following:

{description}

{'Include clear comments explaining the code' if include_comments else 'No comments needed'}

Provide only the code, no explanations outside of comments."""

    content = await skill._generate(
        prompt, system=f"You are an expert {language} developer who writes clean, efficient code."
    )

    return SkillResult(
        success=True,
        message=f"Code generated: {language}",
        data={"code": content, "language": language, "description": description},
        cost=0.02,
        asset_created={"type": "code", "language": language, "description": description[:100]}
    )


async def write_documentation(skill, code: str, doc_type: str) -> SkillResult:
    """Generate documentation"""
    if not code:
        return SkillResult(success=False, message="Code required")

    type_instructions = {
        "readme": "Write a comprehensive README.md",
        "api": "Write API documentation with endpoints, parameters, and examples",
        "tutorial": "Write a step-by-step tutorial"
    }

    prompt = f"""{type_instructions.get(doc_type, 'Write documentation')} for this code:

```
{code[:3000]}
```

Make it clear, comprehensive, and beginner-friendly."""

    content = await skill._generate(
        prompt, system="You are a technical writer who creates excellent documentation."
    )

    return SkillResult(
        success=True,
        message=f"Documentation created: {doc_type}",
        data={"documentation": content, "type": doc_type},
        cost=0.03,
        asset_created={"type": "documentation", "doc_type": doc_type}
    )


async def rewrite_content(skill, content: str, goal: str) -> SkillResult:
    """Rewrite content"""
    if not content:
        return SkillResult(success=False, message="Content required")

    goal_instructions = {
        "improve_clarity": "Make this clearer and easier to understand",
        "make_shorter": "Make this more concise while keeping the key points",
        "make_longer": "Expand on this with more detail and examples",
        "change_tone": "Make this more professional/casual as appropriate"
    }

    prompt = f"""{goal_instructions.get(goal, 'Improve this content')}:

{content}

Rewrite it now."""

    rewritten = await skill._generate(prompt)

    return SkillResult(
        success=True,
        message=f"Content rewritten: {goal}",
        data={"original": content[:500], "rewritten": rewritten, "goal": goal},
        cost=0.02
    )


async def summarize(skill, content: str, length: str) -> SkillResult:
    """Summarize content"""
    if not content:
        return SkillResult(success=False, message="Content required")

    length_guide = {
        "brief": "2-3 sentences",
        "medium": "1 paragraph",
        "detailed": "3-5 paragraphs with key points"
    }

    prompt = f"""Summarize this in {length_guide.get(length, '1 paragraph')}:

{content}"""

    summary = await skill._generate(prompt)

    return SkillResult(
        success=True,
        message=f"Content summarized: {length}",
        data={"summary": summary, "original_length": len(content), "summary_length": len(summary)},
        cost=0.01
    )
