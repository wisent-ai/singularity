#!/usr/bin/env python3
"""
Revenue Services Skill - Concrete service implementations that produce sellable value.

While MarketplaceSkill manages the catalog/orders/pricing business layer, and
ServiceAPI exposes HTTP endpoints, THIS skill is the actual value engine. It
implements specific, useful services that external customers would pay for:

  - Code Review: Analyze code for bugs, style issues, security vulnerabilities
  - Text Summarization: Condense long documents into key points
  - Data Analysis: Extract insights from structured data (CSV/JSON)
  - SEO Audit: Analyze text for search engine optimization
  - API Documentation: Generate docs from code/endpoints

Each service is designed to run without external API keys (using pattern-based
analysis) so the agent can start earning immediately. Services that use LLM
are optional upgrades that produce higher-quality results.

Revenue flow:
  1. Agent registers these services in MarketplaceSkill catalog
  2. Customer places order via ServiceAPI
  3. MarketplaceSkill dispatches to this skill
  4. This skill executes the service and returns results
  5. MarketplaceSkill records revenue

Part of the Revenue Generation pillar: the value production layer.
"""

import json
import re
import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillResult, SkillManifest, SkillAction


# Persisted service execution log
SERVICES_LOG_FILE = Path(__file__).parent.parent / "data" / "revenue_services_log.json"
MAX_LOG_ENTRIES = 500


class RevenueServiceSkill(Skill):
    """
    Concrete revenue-generating service implementations.

    Provides actual useful services that produce value customers pay for.
    Each service works standalone (pattern-based analysis) with optional
    LLM enhancement for higher quality results.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._execution_log: List[Dict] = []
        self._load_log()

    def _load_log(self):
        try:
            if SERVICES_LOG_FILE.exists():
                with open(SERVICES_LOG_FILE, "r") as f:
                    self._execution_log = json.load(f)
        except (json.JSONDecodeError, IOError):
            self._execution_log = []

    def _save_log(self):
        SERVICES_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Trim old entries
        if len(self._execution_log) > MAX_LOG_ENTRIES:
            self._execution_log = self._execution_log[-MAX_LOG_ENTRIES:]
        try:
            with open(SERVICES_LOG_FILE, "w") as f:
                json.dump(self._execution_log, f, indent=2, default=str)
        except IOError:
            pass

    def _log_execution(self, service: str, success: bool, revenue: float, cost: float):
        self._execution_log.append({
            "service": service,
            "success": success,
            "revenue": revenue,
            "cost": cost,
            "profit": revenue - cost,
            "timestamp": datetime.now().isoformat(),
        })
        self._save_log()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="revenue_services",
            name="Revenue Services",
            version="1.0.0",
            category="revenue",
            description="Concrete service implementations that produce sellable value (code review, summarization, data analysis, etc.)",
            required_credentials=[],
            install_cost=0,
            actions=[
                SkillAction(
                    name="code_review",
                    description="Analyze code for bugs, style issues, security vulnerabilities, and improvement suggestions",
                    parameters={
                        "code": {"type": "string", "required": True, "description": "Source code to review"},
                        "language": {"type": "string", "required": False, "description": "Programming language (auto-detected if omitted)"},
                        "focus": {"type": "string", "required": False, "description": "Focus area: security, performance, style, bugs, all"},
                    },
                    estimated_cost=0.01,
                    estimated_duration_seconds=5,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="summarize_text",
                    description="Condense long text into key points and a concise summary",
                    parameters={
                        "text": {"type": "string", "required": True, "description": "Text to summarize"},
                        "max_points": {"type": "number", "required": False, "description": "Maximum number of key points (default 5)"},
                        "style": {"type": "string", "required": False, "description": "Style: bullet, paragraph, executive"},
                    },
                    estimated_cost=0.005,
                    estimated_duration_seconds=3,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="analyze_data",
                    description="Extract insights from structured data (JSON records)",
                    parameters={
                        "data": {"type": "array", "required": True, "description": "Array of data records (dicts)"},
                        "question": {"type": "string", "required": False, "description": "Specific question to answer about the data"},
                    },
                    estimated_cost=0.01,
                    estimated_duration_seconds=5,
                    success_probability=0.90,
                ),
                SkillAction(
                    name="seo_audit",
                    description="Analyze text content for SEO optimization opportunities",
                    parameters={
                        "text": {"type": "string", "required": True, "description": "Content to audit"},
                        "target_keywords": {"type": "array", "required": False, "description": "Target keywords to check for"},
                        "url": {"type": "string", "required": False, "description": "URL of the page (for meta analysis)"},
                    },
                    estimated_cost=0.005,
                    estimated_duration_seconds=3,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="generate_api_docs",
                    description="Generate API documentation from code or endpoint descriptions",
                    parameters={
                        "code": {"type": "string", "required": False, "description": "Source code containing API endpoints"},
                        "endpoints": {"type": "array", "required": False, "description": "Array of endpoint definitions"},
                        "format": {"type": "string", "required": False, "description": "Output format: markdown, openapi, html"},
                    },
                    estimated_cost=0.01,
                    estimated_duration_seconds=5,
                    success_probability=0.90,
                ),
                SkillAction(
                    name="service_stats",
                    description="Get execution statistics for all revenue services",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=1.0,
                ),
            ],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        actions = {
            "code_review": self._code_review,
            "summarize_text": self._summarize_text,
            "analyze_data": self._analyze_data,
            "seo_audit": self._seo_audit,
            "generate_api_docs": self._generate_api_docs,
            "service_stats": self._service_stats,
        }

        if action not in actions:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )

        try:
            return await actions[action](params)
        except Exception as e:
            self._log_execution(action, False, 0, 0)
            return SkillResult(success=False, message=f"Error in {action}: {str(e)}")

    async def initialize(self) -> bool:
        self.initialized = True
        return True

    # ── Code Review Service ─────────────────────────────────────────

    async def _code_review(self, params: Dict) -> SkillResult:
        """Analyze code for issues and improvements."""
        code = params.get("code", "")
        if not code or not code.strip():
            return SkillResult(success=False, message="'code' parameter is required and must be non-empty")

        language = params.get("language", self._detect_language(code))
        focus = params.get("focus", "all")

        issues = []
        suggestions = []
        metrics = {
            "lines": len(code.splitlines()),
            "characters": len(code),
            "language": language,
            "focus": focus,
        }

        # Security analysis
        if focus in ("all", "security"):
            issues.extend(self._check_security(code, language))

        # Bug pattern analysis
        if focus in ("all", "bugs"):
            issues.extend(self._check_bugs(code, language))

        # Style analysis
        if focus in ("all", "style"):
            issues.extend(self._check_style(code, language))

        # Performance analysis
        if focus in ("all", "performance"):
            issues.extend(self._check_performance(code, language))

        # General suggestions
        suggestions = self._generate_suggestions(code, language, issues)

        # Score
        severity_weights = {"critical": 10, "high": 5, "medium": 2, "low": 1}
        total_weight = sum(severity_weights.get(i.get("severity", "low"), 1) for i in issues)
        score = max(0, 100 - total_weight * 3)

        review = {
            "score": score,
            "grade": "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 40 else "F",
            "issues": issues,
            "suggestions": suggestions,
            "metrics": metrics,
            "summary": f"Found {len(issues)} issue(s). Code quality score: {score}/100 ({language})",
        }

        revenue = 0.10  # $0.10 per code review
        cost = 0.01
        self._log_execution("code_review", True, revenue, cost)

        return SkillResult(
            success=True,
            message=review["summary"],
            data={"review": review},
            revenue=revenue,
            cost=cost,
        )

    def _detect_language(self, code: str) -> str:
        """Detect programming language from code patterns."""
        indicators = {
            "python": [r"\bdef \w+\(", r"\bimport \w+", r"\bclass \w+:", r"print\(", r"self\.", r"__init__"],
            "javascript": [r"\bfunction\b", r"\bconst\b", r"\blet\b", r"=>", r"console\.", r"\brequire\("],
            "typescript": [r"\binterface\b", r":\s*(string|number|boolean)", r"\btype\b\s+\w+\s*="],
            "go": [r"\bfunc\b", r"\bpackage\b", r":=", r"\bgo\b\s+\w+"],
            "rust": [r"\bfn\b", r"\blet\s+mut\b", r"\bimpl\b", r"->", r"\buse\b\s+\w+"],
            "java": [r"\bpublic\s+class\b", r"\bprivate\b", r"\bSystem\.out\b", r"\bvoid\b"],
            "ruby": [r"\bdef\b", r"\bend\b", r"\bputs\b", r"\brequire\b", r"\battr_"],
        }
        scores = {}
        for lang, patterns in indicators.items():
            scores[lang] = sum(1 for p in patterns if re.search(p, code))
        if scores:
            best = max(scores, key=scores.get)
            if scores[best] > 0:
                return best
        return "unknown"

    def _check_security(self, code: str, language: str) -> List[Dict]:
        """Check for security vulnerabilities."""
        issues = []
        patterns = [
            (r"eval\s*\(", "Use of eval() - potential code injection", "critical"),
            (r"exec\s*\(", "Use of exec() - potential code injection", "critical"),
            (r"__import__\s*\(", "Dynamic import - potential code injection", "high"),
            (r"subprocess\.call\s*\(.*shell\s*=\s*True", "Shell injection risk with subprocess", "critical"),
            (r"os\.system\s*\(", "Use of os.system() - prefer subprocess with shell=False", "high"),
            (r"password\s*=\s*['\"][^'\"]+['\"]", "Hardcoded password detected", "critical"),
            (r"api_key\s*=\s*['\"][^'\"]+['\"]", "Hardcoded API key detected", "critical"),
            (r"secret\s*=\s*['\"][A-Za-z0-9+/=]{16,}['\"]", "Hardcoded secret detected", "critical"),
            (r"SELECT\s+.*\+\s*\w+", "Potential SQL injection (string concatenation in query)", "high"),
            (r"innerHTML\s*=", "innerHTML assignment - potential XSS", "high"),
            (r"document\.write\s*\(", "document.write() - potential XSS", "medium"),
            (r"pickle\.load", "Unsafe deserialization with pickle", "high"),
            (r"yaml\.load\s*\((?!.*Loader)", "Unsafe YAML load without explicit Loader", "high"),
            (r"chmod\s+777", "Overly permissive file permissions (777)", "medium"),
            (r"verify\s*=\s*False", "SSL verification disabled", "high"),
            (r"CORS.*\*", "Wildcard CORS - potential security issue", "medium"),
        ]
        for pattern, msg, severity in patterns:
            matches = list(re.finditer(pattern, code, re.IGNORECASE))
            for match in matches:
                line_num = code[:match.start()].count("\n") + 1
                issues.append({
                    "type": "security",
                    "severity": severity,
                    "message": msg,
                    "line": line_num,
                    "match": match.group(0)[:50],
                })
        return issues

    def _check_bugs(self, code: str, language: str) -> List[Dict]:
        """Check for common bug patterns."""
        issues = []
        patterns = [
            (r"except\s*:", "Bare except clause catches all exceptions including SystemExit/KeyboardInterrupt", "medium"),
            (r"==\s*None", "Use 'is None' instead of '== None'", "low"),
            (r"!=\s*None", "Use 'is not None' instead of '!= None'", "low"),
            (r"type\(\w+\)\s*==", "Use isinstance() instead of type() comparison", "low"),
            (r"\.append\(.*\bfor\b", "Possible list comprehension would be cleaner", "low"),
            (r"while\s+True\s*:", "Infinite loop - ensure break condition exists", "medium"),
            (r"global\s+\w+", "Global variable usage - consider passing as parameter", "medium"),
            (r"except\s+\w+\s*,\s*\w+", "Old-style except syntax (Python 2)", "medium"),
            (r"mutable.*default", "Possible mutable default argument", "medium"),
        ]

        if language == "python":
            patterns.extend([
                (r"def\s+\w+\([^)]*=\s*\[\]", "Mutable default argument (empty list)", "high"),
                (r"def\s+\w+\([^)]*=\s*\{\}", "Mutable default argument (empty dict)", "high"),
            ])

        if language in ("javascript", "typescript"):
            patterns.extend([
                (r"==(?!=)", "Loose equality (==) - prefer strict equality (===)", "medium"),
                (r"var\s+", "Use of 'var' - prefer 'const' or 'let'", "low"),
            ])

        for pattern, msg, severity in patterns:
            matches = list(re.finditer(pattern, code))
            for match in matches:
                line_num = code[:match.start()].count("\n") + 1
                issues.append({
                    "type": "bug",
                    "severity": severity,
                    "message": msg,
                    "line": line_num,
                })
        return issues

    def _check_style(self, code: str, language: str) -> List[Dict]:
        """Check code style issues."""
        issues = []
        lines = code.splitlines()

        for i, line in enumerate(lines, 1):
            # Long lines
            if len(line) > 120:
                issues.append({
                    "type": "style",
                    "severity": "low",
                    "message": f"Line too long ({len(line)} chars, max 120)",
                    "line": i,
                })
            # Trailing whitespace
            if line != line.rstrip() and line.strip():
                issues.append({
                    "type": "style",
                    "severity": "low",
                    "message": "Trailing whitespace",
                    "line": i,
                })

        # Check for consistent indentation
        indent_counts = {"tabs": 0, "spaces": 0}
        for line in lines:
            if line.startswith("\t"):
                indent_counts["tabs"] += 1
            elif line.startswith("  "):
                indent_counts["spaces"] += 1
        if indent_counts["tabs"] > 0 and indent_counts["spaces"] > 0:
            issues.append({
                "type": "style",
                "severity": "medium",
                "message": "Mixed tabs and spaces for indentation",
                "line": 0,
            })

        # Function/method length
        if language == "python":
            func_pattern = re.compile(r"^\s*def\s+(\w+)\s*\(", re.MULTILINE)
            func_starts = [(m.group(1), m.start()) for m in func_pattern.finditer(code)]
            for idx, (name, start) in enumerate(func_starts):
                end = func_starts[idx + 1][1] if idx + 1 < len(func_starts) else len(code)
                func_lines = code[start:end].count("\n")
                if func_lines > 50:
                    line_num = code[:start].count("\n") + 1
                    issues.append({
                        "type": "style",
                        "severity": "medium",
                        "message": f"Function '{name}' is {func_lines} lines long (consider splitting)",
                        "line": line_num,
                    })

        return issues

    def _check_performance(self, code: str, language: str) -> List[Dict]:
        """Check for performance anti-patterns."""
        issues = []
        patterns = [
            (r"for\s+.*\bin\s+range\s*\(\s*len\s*\(", "Use enumerate() instead of range(len())", "low"),
            (r"\+\s*=\s*['\"]", "String concatenation in loop - use join() or list", "medium"),
            (r"time\.sleep\s*\(\s*0\s*\)", "sleep(0) is a no-op, consider removing", "low"),
            (r"SELECT\s+\*", "SELECT * can be slow - specify needed columns", "medium"),
            (r"\.readlines\(\)", "readlines() loads entire file - iterate file object directly", "low"),
            (r"import\s+\*", "Wildcard import - increases memory and may shadow names", "medium"),
            (r"re\.\w+\(.*\)\s*#.*loop", "Regex in loop - consider compiling with re.compile()", "medium"),
        ]
        for pattern, msg, severity in patterns:
            matches = list(re.finditer(pattern, code, re.IGNORECASE))
            for match in matches:
                line_num = code[:match.start()].count("\n") + 1
                issues.append({
                    "type": "performance",
                    "severity": severity,
                    "message": msg,
                    "line": line_num,
                })
        return issues

    def _generate_suggestions(self, code: str, language: str, issues: List[Dict]) -> List[str]:
        """Generate improvement suggestions based on the analysis."""
        suggestions = []
        severity_counts = Counter(i.get("severity") for i in issues)

        if severity_counts.get("critical", 0) > 0:
            suggestions.append("URGENT: Address critical security issues before deploying this code")
        if severity_counts.get("high", 0) > 0:
            suggestions.append("Fix high-severity issues to prevent potential security vulnerabilities or bugs")

        lines = len(code.splitlines())
        if lines > 300:
            suggestions.append(f"Consider splitting this {lines}-line file into smaller modules")

        # Check for missing error handling
        if language == "python" and "try" not in code and lines > 20:
            suggestions.append("Consider adding try/except blocks for error handling")

        # Check for no docstrings
        if language == "python" and '"""' not in code and "'''" not in code and lines > 10:
            suggestions.append("Add docstrings to functions and classes for better documentation")

        # Check for no tests
        if "test" not in code.lower() and "assert" not in code:
            suggestions.append("Consider writing unit tests for this code")

        if not issues:
            suggestions.append("Code looks clean! Consider adding more comprehensive tests")

        return suggestions

    # ── Text Summarization Service ──────────────────────────────────

    async def _summarize_text(self, params: Dict) -> SkillResult:
        """Summarize text into key points."""
        text = params.get("text", "")
        if not text or not text.strip():
            return SkillResult(success=False, message="'text' parameter is required and must be non-empty")

        max_points = int(params.get("max_points", 5))
        style = params.get("style", "bullet")

        # Sentence extraction
        sentences = re.split(r'[.!?]+\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return SkillResult(success=False, message="Text too short or no meaningful sentences found")

        # Score sentences by importance
        word_freq = Counter()
        for s in sentences:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', s.lower())
            word_freq.update(words)

        # Remove very common words (basic stopwords)
        stopwords = {"the", "and", "for", "are", "but", "not", "you", "all", "can",
                      "has", "her", "was", "one", "our", "out", "this", "that", "with",
                      "have", "from", "they", "been", "said", "each", "which", "their",
                      "will", "way", "about", "many", "then", "them", "would", "like",
                      "more", "some", "very", "when", "what", "your", "how", "its"}
        for sw in stopwords:
            word_freq.pop(sw, None)

        def score_sentence(s):
            words = re.findall(r'\b[a-zA-Z]{3,}\b', s.lower())
            if not words:
                return 0
            word_score = sum(word_freq.get(w, 0) for w in words) / len(words)
            # Bonus for position (first/last sentences more important)
            position_bonus = 0
            idx = sentences.index(s) if s in sentences else len(sentences)
            if idx < 2:
                position_bonus = 2
            elif idx >= len(sentences) - 2:
                position_bonus = 1
            # Bonus for length (medium sentences preferred)
            length_bonus = 1 if 20 < len(s) < 200 else 0
            return word_score + position_bonus + length_bonus

        scored = [(s, score_sentence(s)) for s in sentences]
        scored.sort(key=lambda x: x[1], reverse=True)

        key_points = [s for s, _ in scored[:max_points]]

        # Build summary based on style
        word_count = len(text.split())
        summary_ratio = len(" ".join(key_points).split()) / word_count if word_count > 0 else 0

        if style == "paragraph":
            summary_text = ". ".join(key_points)
            if not summary_text.endswith("."):
                summary_text += "."
        elif style == "executive":
            summary_text = f"Executive Summary ({word_count} words analyzed):\n\n"
            summary_text += ". ".join(key_points[:3]) + "."
            if len(key_points) > 3:
                summary_text += "\n\nAdditional points:\n"
                for p in key_points[3:]:
                    summary_text += f"- {p}\n"
        else:  # bullet
            summary_text = "\n".join(f"- {p}" for p in key_points)

        result = {
            "summary": summary_text,
            "key_points": key_points,
            "metrics": {
                "original_words": word_count,
                "original_sentences": len(sentences),
                "points_extracted": len(key_points),
                "compression_ratio": round(summary_ratio, 2),
            },
        }

        revenue = 0.05
        cost = 0.005
        self._log_execution("summarize_text", True, revenue, cost)

        return SkillResult(
            success=True,
            message=f"Summarized {word_count} words into {len(key_points)} key points",
            data=result,
            revenue=revenue,
            cost=cost,
        )

    # ── Data Analysis Service ───────────────────────────────────────

    async def _analyze_data(self, params: Dict) -> SkillResult:
        """Analyze structured data and extract insights."""
        data = params.get("data", [])
        if not data or not isinstance(data, list):
            return SkillResult(success=False, message="'data' parameter must be a non-empty array of records")

        if not isinstance(data[0], dict):
            return SkillResult(success=False, message="Data records must be dictionaries/objects")

        question = params.get("question", "")

        # Basic statistics
        record_count = len(data)
        fields = set()
        for record in data:
            fields.update(record.keys())
        fields = sorted(fields)

        # Analyze each field
        field_analysis = {}
        for field in fields:
            values = [r.get(field) for r in data if field in r]
            non_null = [v for v in values if v is not None]

            analysis = {
                "count": len(values),
                "null_count": len(values) - len(non_null),
                "unique_count": len(set(str(v) for v in non_null)),
            }

            # Numeric analysis
            numeric_vals = []
            for v in non_null:
                try:
                    numeric_vals.append(float(v))
                except (ValueError, TypeError):
                    pass

            if numeric_vals and len(numeric_vals) > len(non_null) * 0.5:
                analysis["type"] = "numeric"
                analysis["min"] = min(numeric_vals)
                analysis["max"] = max(numeric_vals)
                analysis["mean"] = sum(numeric_vals) / len(numeric_vals)
                analysis["sum"] = sum(numeric_vals)
                sorted_vals = sorted(numeric_vals)
                mid = len(sorted_vals) // 2
                analysis["median"] = sorted_vals[mid] if len(sorted_vals) % 2 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
            else:
                analysis["type"] = "categorical"
                value_counts = Counter(str(v) for v in non_null)
                analysis["top_values"] = value_counts.most_common(5)

            field_analysis[field] = analysis

        # Generate insights
        insights = []

        # Find correlations between numeric fields
        numeric_fields = [f for f, a in field_analysis.items() if a.get("type") == "numeric"]
        if len(numeric_fields) >= 2:
            insights.append(f"Dataset has {len(numeric_fields)} numeric fields that may be correlated: {', '.join(numeric_fields)}")

        # Find fields with high null rates
        for field, analysis in field_analysis.items():
            null_rate = analysis["null_count"] / analysis["count"] if analysis["count"] > 0 else 0
            if null_rate > 0.2:
                insights.append(f"Field '{field}' has {null_rate:.0%} null values - data quality issue")

        # Find low-cardinality fields (potential categories)
        for field, analysis in field_analysis.items():
            if analysis.get("type") == "categorical" and analysis["unique_count"] < 10:
                top = analysis.get("top_values", [])
                if top:
                    top_str = ", ".join(f"{v} ({c})" for v, c in top[:3])
                    insights.append(f"Field '{field}' has {analysis['unique_count']} categories. Top: {top_str}")

        # Summary stats
        if numeric_fields:
            for f in numeric_fields[:3]:
                a = field_analysis[f]
                insights.append(f"'{f}': min={a['min']}, max={a['max']}, mean={a['mean']:.2f}, median={a['median']}")

        # Answer specific question if provided
        answer = None
        if question:
            answer = self._answer_data_question(question, data, field_analysis)

        result = {
            "record_count": record_count,
            "fields": fields,
            "field_analysis": field_analysis,
            "insights": insights,
        }
        if answer:
            result["answer"] = answer

        revenue = 0.10
        cost = 0.01
        self._log_execution("analyze_data", True, revenue, cost)

        return SkillResult(
            success=True,
            message=f"Analyzed {record_count} records across {len(fields)} fields. Generated {len(insights)} insights.",
            data=result,
            revenue=revenue,
            cost=cost,
        )

    def _answer_data_question(self, question: str, data: List[Dict], field_analysis: Dict) -> str:
        """Try to answer a specific question about the data."""
        q = question.lower()

        # Common question patterns
        if "how many" in q or "count" in q:
            return f"The dataset contains {len(data)} records."

        if "average" in q or "mean" in q:
            for field, analysis in field_analysis.items():
                if field.lower() in q and analysis.get("type") == "numeric":
                    return f"The average {field} is {analysis['mean']:.2f}"

        if "max" in q or "highest" in q or "largest" in q:
            for field, analysis in field_analysis.items():
                if field.lower() in q and analysis.get("type") == "numeric":
                    return f"The maximum {field} is {analysis['max']}"

        if "min" in q or "lowest" in q or "smallest" in q:
            for field, analysis in field_analysis.items():
                if field.lower() in q and analysis.get("type") == "numeric":
                    return f"The minimum {field} is {analysis['min']}"

        if "total" in q or "sum" in q:
            for field, analysis in field_analysis.items():
                if field.lower() in q and analysis.get("type") == "numeric":
                    return f"The total {field} is {analysis['sum']}"

        return f"Analyzed {len(data)} records. See field_analysis and insights for details."

    # ── SEO Audit Service ───────────────────────────────────────────

    async def _seo_audit(self, params: Dict) -> SkillResult:
        """Audit content for SEO optimization."""
        text = params.get("text", "")
        if not text or not text.strip():
            return SkillResult(success=False, message="'text' parameter is required and must be non-empty")

        target_keywords = params.get("target_keywords", [])
        url = params.get("url", "")

        words = text.lower().split()
        word_count = len(words)
        sentences = re.split(r'[.!?]+\s+', text)
        sentence_count = len([s for s in sentences if s.strip()])

        # Readability metrics
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        long_words = [w for w in words if len(w) > 6]
        reading_ease = max(0, min(100, 206.835 - 1.015 * avg_sentence_length - 84.6 * (len(long_words) / word_count if word_count > 0 else 0)))

        # Keyword analysis
        keyword_results = {}
        for kw in target_keywords:
            kw_lower = kw.lower()
            count = text.lower().count(kw_lower)
            density = (count / word_count * 100) if word_count > 0 else 0
            in_first_100 = kw_lower in " ".join(words[:100])
            keyword_results[kw] = {
                "count": count,
                "density_pct": round(density, 2),
                "in_first_100_words": in_first_100,
                "status": "good" if 1 <= density <= 3 else "low" if density < 1 else "high",
            }

        # Word frequency (for keyword suggestions)
        word_freq = Counter(w for w in words if len(w) > 4)
        top_words = word_freq.most_common(10)

        # SEO issues and recommendations
        issues = []
        recommendations = []

        if word_count < 300:
            issues.append({"severity": "high", "message": f"Content too short ({word_count} words). Aim for 1000+ words for SEO."})
        elif word_count < 1000:
            issues.append({"severity": "medium", "message": f"Content could be longer ({word_count} words). 1000-2000 words is ideal."})

        if avg_sentence_length > 25:
            issues.append({"severity": "medium", "message": f"Average sentence too long ({avg_sentence_length:.0f} words). Aim for 15-20."})

        if reading_ease < 30:
            issues.append({"severity": "medium", "message": f"Content is hard to read (score: {reading_ease:.0f}/100). Simplify language."})

        # Check headings
        heading_count = len(re.findall(r'^#+\s|<h[1-6]', text, re.MULTILINE))
        if heading_count == 0 and word_count > 200:
            issues.append({"severity": "medium", "message": "No headings found. Add H2/H3 headings for better structure."})

        # Check for keyword stuffing
        for kw, data in keyword_results.items():
            if data["density_pct"] > 3:
                issues.append({"severity": "high", "message": f"Keyword '{kw}' may be over-optimized ({data['density_pct']}% density)"})
            elif data["count"] == 0:
                recommendations.append(f"Target keyword '{kw}' not found in content. Consider adding it naturally.")
            elif not data["in_first_100_words"]:
                recommendations.append(f"Add '{kw}' to the first 100 words for better SEO signal.")

        # Score
        score = 100
        for issue in issues:
            if issue["severity"] == "high":
                score -= 15
            elif issue["severity"] == "medium":
                score -= 8
            else:
                score -= 3
        score = max(0, score)

        result = {
            "score": score,
            "grade": "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 50 else "D",
            "metrics": {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_sentence_length": round(avg_sentence_length, 1),
                "reading_ease": round(reading_ease, 1),
                "heading_count": heading_count,
            },
            "keyword_analysis": keyword_results,
            "suggested_keywords": [w for w, _ in top_words],
            "issues": issues,
            "recommendations": recommendations,
        }

        revenue = 0.05
        cost = 0.005
        self._log_execution("seo_audit", True, revenue, cost)

        return SkillResult(
            success=True,
            message=f"SEO audit complete. Score: {score}/100 ({len(issues)} issues found)",
            data=result,
            revenue=revenue,
            cost=cost,
        )

    # ── API Documentation Service ───────────────────────────────────

    async def _generate_api_docs(self, params: Dict) -> SkillResult:
        """Generate API documentation from code or endpoint definitions."""
        code = params.get("code", "")
        endpoints = params.get("endpoints", [])
        output_format = params.get("format", "markdown")

        if not code and not endpoints:
            return SkillResult(success=False, message="Either 'code' or 'endpoints' parameter is required")

        discovered_endpoints = []

        # Extract endpoints from code
        if code:
            discovered_endpoints.extend(self._extract_endpoints_from_code(code))

        # Add manually specified endpoints
        for ep in endpoints:
            if isinstance(ep, dict):
                discovered_endpoints.append({
                    "method": ep.get("method", "GET"),
                    "path": ep.get("path", "/"),
                    "description": ep.get("description", ""),
                    "parameters": ep.get("parameters", []),
                    "response": ep.get("response", ""),
                })

        if not discovered_endpoints:
            return SkillResult(
                success=False,
                message="No API endpoints found in the provided code or definitions",
            )

        # Generate documentation
        if output_format == "markdown":
            doc = self._generate_markdown_docs(discovered_endpoints)
        elif output_format == "openapi":
            doc = self._generate_openapi_docs(discovered_endpoints)
        else:
            doc = self._generate_markdown_docs(discovered_endpoints)

        result = {
            "documentation": doc,
            "endpoints_found": len(discovered_endpoints),
            "endpoints": discovered_endpoints,
            "format": output_format,
        }

        revenue = 0.10
        cost = 0.01
        self._log_execution("generate_api_docs", True, revenue, cost)

        return SkillResult(
            success=True,
            message=f"Generated {output_format} documentation for {len(discovered_endpoints)} endpoints",
            data=result,
            revenue=revenue,
            cost=cost,
        )

    def _extract_endpoints_from_code(self, code: str) -> List[Dict]:
        """Extract API endpoints from code patterns."""
        endpoints = []

        # Flask/FastAPI patterns
        patterns = [
            # @app.get("/path") or @router.get("/path") (Python decorator style)
            (r'@\w+\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']', "decorator"),
            # app.add_route or router.add_api_route
            (r'add_(?:api_)?route\s*\(\s*["\']([^"\']+)["\'].*methods?\s*=\s*\[?\s*["\'](\w+)', "add_route"),
            # Express.js: app.get('/path', function/callback) - not preceded by @ on same line
            (r'(?<![@\w])app\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']', "express"),
        ]

        seen = set()  # (method, path) deduplication
        for pattern, source in patterns:
            for match in re.finditer(pattern, code, re.IGNORECASE):
                if source == "add_route":
                    path, method = match.group(1), match.group(2)
                else:
                    method, path = match.group(1), match.group(2)

                key = (method.upper(), path)
                if key in seen:
                    continue
                seen.add(key)

                # Try to find the function name/docstring after the decorator
                after = code[match.end():match.end() + 200]
                func_match = re.search(r'(?:async\s+)?def\s+(\w+)', after)
                doc_match = re.search(r'"""([^"]+)"""', after)

                endpoints.append({
                    "method": method.upper(),
                    "path": path,
                    "description": doc_match.group(1).strip() if doc_match else (func_match.group(1).replace("_", " ").title() if func_match else ""),
                    "function_name": func_match.group(1) if func_match else "",
                    "parameters": [],
                })

        return endpoints

    def _generate_markdown_docs(self, endpoints: List[Dict]) -> str:
        """Generate Markdown API documentation."""
        lines = ["# API Documentation\n"]
        lines.append(f"*{len(endpoints)} endpoints documented*\n")

        # Group by path prefix
        for ep in endpoints:
            method = ep.get("method", "GET")
            path = ep.get("path", "/")
            desc = ep.get("description", "No description")

            lines.append(f"## `{method}` {path}\n")
            lines.append(f"{desc}\n")

            params = ep.get("parameters", [])
            if params:
                lines.append("### Parameters\n")
                lines.append("| Name | Type | Required | Description |")
                lines.append("|------|------|----------|-------------|")
                for p in params:
                    name = p.get("name", "")
                    ptype = p.get("type", "string")
                    req = "Yes" if p.get("required") else "No"
                    pdesc = p.get("description", "")
                    lines.append(f"| {name} | {ptype} | {req} | {pdesc} |")
                lines.append("")

            resp = ep.get("response", "")
            if resp:
                lines.append(f"### Response\n\n```json\n{resp}\n```\n")

            lines.append("---\n")

        return "\n".join(lines)

    def _generate_openapi_docs(self, endpoints: List[Dict]) -> Dict:
        """Generate OpenAPI 3.0 spec."""
        paths = {}
        for ep in endpoints:
            method = ep.get("method", "GET").lower()
            path = ep.get("path", "/")

            if path not in paths:
                paths[path] = {}

            operation = {
                "summary": ep.get("description", ""),
                "responses": {"200": {"description": "Success"}},
            }

            params = ep.get("parameters", [])
            if params:
                operation["parameters"] = [
                    {
                        "name": p.get("name", ""),
                        "in": "query",
                        "required": p.get("required", False),
                        "schema": {"type": p.get("type", "string")},
                        "description": p.get("description", ""),
                    }
                    for p in params
                ]

            paths[path][method] = operation

        return {
            "openapi": "3.0.0",
            "info": {"title": "API Documentation", "version": "1.0.0"},
            "paths": paths,
        }

    # ── Service Stats ───────────────────────────────────────────────

    async def _service_stats(self, params: Dict) -> SkillResult:
        """Get execution statistics for all revenue services."""
        by_service = {}
        total_revenue = 0
        total_cost = 0
        total_executions = 0

        for entry in self._execution_log:
            service = entry.get("service", "unknown")
            if service not in by_service:
                by_service[service] = {"executions": 0, "successes": 0, "revenue": 0, "cost": 0}
            by_service[service]["executions"] += 1
            if entry.get("success"):
                by_service[service]["successes"] += 1
            by_service[service]["revenue"] += entry.get("revenue", 0)
            by_service[service]["cost"] += entry.get("cost", 0)
            total_revenue += entry.get("revenue", 0)
            total_cost += entry.get("cost", 0)
            total_executions += 1

        for service_data in by_service.values():
            service_data["profit"] = service_data["revenue"] - service_data["cost"]
            service_data["success_rate"] = (
                service_data["successes"] / service_data["executions"]
                if service_data["executions"] > 0 else 0
            )

        result = {
            "total_executions": total_executions,
            "total_revenue": round(total_revenue, 4),
            "total_cost": round(total_cost, 4),
            "total_profit": round(total_revenue - total_cost, 4),
            "by_service": by_service,
        }

        return SkillResult(
            success=True,
            message=f"Revenue services: {total_executions} executions, ${total_revenue:.2f} revenue, ${total_revenue - total_cost:.2f} profit",
            data=result,
        )
