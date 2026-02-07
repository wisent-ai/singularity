"""Tests for RevenueServiceSkill - concrete revenue-generating services."""

import pytest
from singularity.skills.revenue_services import RevenueServiceSkill


@pytest.fixture
def skill():
    s = RevenueServiceSkill()
    s._execution_log = []  # Fresh log
    return s


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "revenue_services"
    assert m.category == "revenue"
    actions = [a.name for a in m.actions]
    assert "code_review" in actions
    assert "summarize_text" in actions
    assert "analyze_data" in actions
    assert "seo_audit" in actions
    assert "generate_api_docs" in actions
    assert "service_stats" in actions


@pytest.mark.asyncio
async def test_code_review_python(skill):
    code = '''
def process(data=[]):
    eval(data[0])
    password = "secret123"
    for i in range(len(data)):
        print(data[i])
'''
    result = await skill.execute("code_review", {"code": code, "language": "python"})
    assert result.success
    review = result.data["review"]
    assert review["score"] < 80  # Should find issues
    assert len(review["issues"]) > 0
    assert result.revenue > 0


@pytest.mark.asyncio
async def test_code_review_empty(skill):
    result = await skill.execute("code_review", {"code": ""})
    assert not result.success


@pytest.mark.asyncio
async def test_code_review_language_detection(skill):
    js_code = "const x = () => { console.log('hello'); };"
    result = await skill.execute("code_review", {"code": js_code})
    assert result.success
    assert result.data["review"]["metrics"]["language"] == "javascript"


@pytest.mark.asyncio
async def test_summarize_text(skill):
    text = (
        "Artificial intelligence is transforming the modern world. "
        "Machine learning algorithms can now process vast amounts of data. "
        "Natural language processing enables computers to understand human speech. "
        "Computer vision allows machines to interpret images and video. "
        "Deep learning has achieved breakthroughs in many domains. "
        "The impact on healthcare includes faster diagnosis and drug discovery. "
        "In finance, AI powers fraud detection and algorithmic trading. "
        "Education is being personalized through adaptive learning systems. "
        "Transportation will be revolutionized by autonomous vehicles. "
        "The ethical implications of AI require careful consideration."
    )
    result = await skill.execute("summarize_text", {"text": text, "max_points": 3})
    assert result.success
    assert len(result.data["key_points"]) <= 3
    assert result.data["metrics"]["original_words"] > 0
    assert result.revenue > 0


@pytest.mark.asyncio
async def test_summarize_empty(skill):
    result = await skill.execute("summarize_text", {"text": ""})
    assert not result.success


@pytest.mark.asyncio
async def test_analyze_data(skill):
    data = [
        {"name": "Alice", "age": 30, "score": 95},
        {"name": "Bob", "age": 25, "score": 82},
        {"name": "Charlie", "age": 35, "score": 88},
        {"name": "Diana", "age": 28, "score": 91},
    ]
    result = await skill.execute("analyze_data", {"data": data, "question": "What is the average age?"})
    assert result.success
    assert result.data["record_count"] == 4
    assert "age" in result.data["field_analysis"]
    assert result.data["field_analysis"]["age"]["type"] == "numeric"
    assert "answer" in result.data
    assert result.revenue > 0


@pytest.mark.asyncio
async def test_analyze_data_empty(skill):
    result = await skill.execute("analyze_data", {"data": []})
    assert not result.success


@pytest.mark.asyncio
async def test_seo_audit(skill):
    text = "Python programming is great. " * 50  # ~300 words
    result = await skill.execute("seo_audit", {
        "text": text,
        "target_keywords": ["python", "programming", "nonexistent"],
    })
    assert result.success
    assert result.data["score"] > 0
    assert "python" in result.data["keyword_analysis"]
    assert result.data["keyword_analysis"]["nonexistent"]["count"] == 0
    assert result.revenue > 0


@pytest.mark.asyncio
async def test_generate_api_docs_from_code(skill):
    code = '''
@app.get("/users")
async def list_users():
    """List all users."""
    pass

@app.post("/users")
async def create_user():
    """Create a new user."""
    pass
'''
    result = await skill.execute("generate_api_docs", {"code": code, "format": "markdown"})
    assert result.success
    assert result.data["endpoints_found"] == 2
    assert "GET" in result.data["documentation"]
    assert result.revenue > 0


@pytest.mark.asyncio
async def test_generate_api_docs_openapi(skill):
    result = await skill.execute("generate_api_docs", {
        "endpoints": [{"method": "GET", "path": "/health", "description": "Health check"}],
        "format": "openapi",
    })
    assert result.success
    doc = result.data["documentation"]
    assert doc["openapi"] == "3.0.0"
    assert "/health" in doc["paths"]


@pytest.mark.asyncio
async def test_service_stats(skill):
    await skill.execute("code_review", {"code": "x = 1"})
    await skill.execute("summarize_text", {"text": "Hello world. This is a test. Another sentence here."})
    result = await skill.execute("service_stats", {})
    assert result.success
    assert result.data["total_executions"] >= 2
    assert result.data["total_revenue"] > 0


@pytest.mark.asyncio
async def test_unknown_action(skill):
    result = await skill.execute("nonexistent", {})
    assert not result.success
    assert "Unknown action" in result.message
