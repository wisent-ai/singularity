#!/usr/bin/env python3
"""
CodeReviewSkill - Automated code analysis, review, and security scanning.

Serves TWO pillars simultaneously:

1. Self-Improvement: The agent can review its OWN code changes before
   committing, catching bugs, anti-patterns, and regressions automatically.
   This closes a critical gap - without code review, self-modification is
   flying blind.

2. Revenue Generation: Code review is one of the most valuable services
   a developer tool can offer. This skill can be exposed via ServiceAPI
   to provide paid code review to external consumers.

Actions:
  - analyze: Static analysis of a code string or file (patterns, complexity, issues)
  - review_diff: Review a unified diff for bugs, anti-patterns, and improvements
  - security_scan: Scan code for common security vulnerabilities
  - summarize: Generate a concise summary of what code does
  - score: Produce a numeric quality score (0-100) with breakdown

No external dependencies required - uses pure Python pattern matching and AST.
"""

import ast
import re
import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from .base import Skill, SkillManifest, SkillAction, SkillResult


REVIEW_LOG = Path(__file__).parent.parent / "data" / "code_reviews.json"
MAX_REVIEWS = 500


# --- Pattern Definitions ---

# Common bug patterns (regex, description, severity)
BUG_PATTERNS: List[Tuple[str, str, str]] = [
    (r"\bexcept\s*:", "Bare except catches all exceptions including SystemExit/KeyboardInterrupt", "warning"),
    (r"== None\b|!= None\b", "Use 'is None' / 'is not None' instead of == / !=", "info"),
    (r"\beval\s*\(", "eval() is dangerous - can execute arbitrary code", "critical"),
    (r"\bexec\s*\(", "exec() is dangerous - can execute arbitrary code", "critical"),
    (r"\.format\(.*\bpassword\b", "Password may be exposed in formatted string", "warning"),
    (r"f['\"].*\{.*password.*\}", "Password may be exposed in f-string", "warning"),
    (r"\bprint\s*\(.*password", "Password may be printed to stdout", "warning"),
    (r"TODO|FIXME|HACK|XXX|TEMP", "Unresolved TODO/FIXME marker found", "info"),
    (r"import \*", "Wildcard import pollutes namespace", "warning"),
    (r"time\.sleep\(\s*\d{2,}", "Long sleep detected (>10s) - consider async approach", "info"),
    (r"while\s+True\s*:", "Infinite loop - ensure there's a break condition", "info"),
    (r"os\.system\s*\(", "os.system() is insecure - use subprocess.run() instead", "warning"),
    (r"pickle\.load", "pickle.load() can execute arbitrary code on untrusted data", "critical"),
    (r"yaml\.load\s*\((?!.*Loader)", "yaml.load() without Loader is unsafe", "warning"),
    (r"\.execute\s*\(.*%s|\.execute\s*\(.*\.format\(", "Possible SQL injection - use parameterized queries", "critical"),
    (r"verify\s*=\s*False", "SSL verification disabled - insecure in production", "warning"),
    (r"debug\s*=\s*True", "Debug mode enabled - disable in production", "info"),
    (r"SECRET|API_KEY|TOKEN.*=\s*['\"][^'\"]+['\"]", "Hardcoded secret/credential detected", "critical"),
]

# Security vulnerability patterns
SECURITY_PATTERNS: List[Tuple[str, str, str, str]] = [
    (r"\beval\s*\(", "Code Injection", "eval() can execute arbitrary code", "critical"),
    (r"\bexec\s*\(", "Code Injection", "exec() can execute arbitrary code", "critical"),
    (r"__import__\s*\(", "Dynamic Import", "Dynamic imports can load malicious modules", "warning"),
    (r"subprocess\..*shell\s*=\s*True", "Command Injection", "shell=True allows command injection", "critical"),
    (r"os\.system\s*\(", "Command Injection", "os.system passes commands through shell", "critical"),
    (r"os\.popen\s*\(", "Command Injection", "os.popen passes commands through shell", "warning"),
    (r"pickle\.loads?\s*\(", "Deserialization", "Pickle can execute arbitrary code", "critical"),
    (r"marshal\.loads?\s*\(", "Deserialization", "Marshal can execute arbitrary code", "critical"),
    (r"shelve\.open\s*\(", "Deserialization", "Shelve uses pickle internally", "warning"),
    (r"yaml\.load\s*\((?!.*Loader)", "Deserialization", "yaml.load without safe Loader", "critical"),
    (r"tempfile\.mktemp\s*\(", "Race Condition", "mktemp is vulnerable to race conditions, use mkstemp", "warning"),
    (r"chmod\s*\(.*0o?777", "Permissions", "World-writable file permissions", "critical"),
    (r"verify\s*=\s*False", "TLS", "SSL certificate verification disabled", "warning"),
    (r"http://(?!localhost|127\.0\.0\.1)", "TLS", "Non-HTTPS URL used (except localhost)", "info"),
    (r"md5\s*\(|\.md5\b", "Cryptography", "MD5 is cryptographically broken", "warning"),
    (r"sha1\s*\(|\.sha1\b", "Cryptography", "SHA1 is cryptographically weak", "info"),
    (r"random\.(random|randint|choice|shuffle)\b(?!.*secrets)", "Cryptography", "Use secrets module for security-sensitive randomness", "info"),
    (r"SECRET.*=\s*['\"][^'\"]{4,}['\"]", "Secrets", "Hardcoded secret value", "critical"),
    (r"password\s*=\s*['\"][^'\"]+['\"]", "Secrets", "Hardcoded password", "critical"),
    (r"api_key\s*=\s*['\"][^'\"]+['\"]", "Secrets", "Hardcoded API key", "critical"),
]

# Complexity thresholds
COMPLEXITY_THRESHOLDS = {
    "function_length": 50,  # lines
    "nesting_depth": 4,
    "parameters": 7,
    "cognitive_complexity": 15,
}


class CodeReviewSkill(Skill):
    """
    Automated code review providing static analysis, security scanning,
    and quality scoring for Python code.

    Can review raw code strings or unified diffs. Results are logged
    for trend analysis and self-improvement tracking.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        REVIEW_LOG.parent.mkdir(parents=True, exist_ok=True)
        if not REVIEW_LOG.exists():
            self._save_reviews([])

    def _load_reviews(self) -> List[Dict]:
        try:
            with open(REVIEW_LOG, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_reviews(self, reviews: List[Dict]):
        with open(REVIEW_LOG, "w") as f:
            json.dump(reviews[-MAX_REVIEWS:], f, indent=2)

    def _log_review(self, review_type: str, result: Dict):
        """Persist review result for trend analysis."""
        reviews = self._load_reviews()
        reviews.append({
            "type": review_type,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "issues_found": result.get("total_issues", 0),
                "score": result.get("score", None),
                "critical_count": result.get("critical_count", 0),
            },
        })
        self._save_reviews(reviews)

    # --- Manifest ---

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="code_review",
            name="Code Review",
            version="1.0.0",
            category="dev",
            description="Automated code analysis, review, security scanning, and quality scoring. "
                        "Serves self-improvement (review own changes) and revenue (offer as service).",
            actions=[
                SkillAction(
                    name="analyze",
                    description="Static analysis of Python code - find bugs, anti-patterns, complexity issues",
                    parameters={
                        "code": {"type": "string", "required": True, "description": "Python code to analyze"},
                        "filename": {"type": "string", "required": False, "description": "Optional filename for context"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="review_diff",
                    description="Review a unified diff for bugs, anti-patterns, and improvements",
                    parameters={
                        "diff": {"type": "string", "required": True, "description": "Unified diff text"},
                        "context": {"type": "string", "required": False, "description": "Description of what the change does"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="security_scan",
                    description="Scan code for common security vulnerabilities (injection, secrets, crypto)",
                    parameters={
                        "code": {"type": "string", "required": True, "description": "Code to scan"},
                        "filename": {"type": "string", "required": False, "description": "Optional filename"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="score",
                    description="Produce a quality score (0-100) with category breakdown",
                    parameters={
                        "code": {"type": "string", "required": True, "description": "Python code to score"},
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="history",
                    description="View recent code review history and quality trends",
                    parameters={
                        "limit": {"type": "integer", "required": False, "description": "Number of reviews to show (default 10)"},
                    },
                    estimated_cost=0.0,
                ),
            ],
            required_credentials=[],
        )

    def get_actions(self) -> List[SkillAction]:
        return self.manifest().actions

    async def initialize(self) -> bool:
        self.initialized = True
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        actions = {
            "analyze": self._analyze,
            "review_diff": self._review_diff,
            "security_scan": self._security_scan,
            "score": self._score,
            "history": self._history,
        }
        if action not in actions:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )
        return await actions[action](params)

    # --- Actions ---

    async def _analyze(self, params: Dict) -> SkillResult:
        """Static analysis of Python code."""
        code = params.get("code", "")
        filename = params.get("filename", "<input>")

        if not code.strip():
            return SkillResult(success=False, message="No code provided")

        issues = []
        lines = code.split("\n")

        # Pattern-based analysis
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            for pattern, desc, severity in BUG_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append({
                        "line": i,
                        "severity": severity,
                        "message": desc,
                        "code": stripped[:100],
                    })

        # AST-based analysis
        ast_issues = self._ast_analysis(code, filename)
        issues.extend(ast_issues)

        # Complexity metrics
        metrics = self._compute_metrics(code)

        # Complexity warnings
        if metrics["max_function_length"] > COMPLEXITY_THRESHOLDS["function_length"]:
            issues.append({
                "line": 0,
                "severity": "warning",
                "message": f"Function too long ({metrics['max_function_length']} lines, threshold: {COMPLEXITY_THRESHOLDS['function_length']})",
                "code": "",
            })
        if metrics["max_nesting_depth"] > COMPLEXITY_THRESHOLDS["nesting_depth"]:
            issues.append({
                "line": 0,
                "severity": "warning",
                "message": f"Deep nesting ({metrics['max_nesting_depth']} levels, threshold: {COMPLEXITY_THRESHOLDS['nesting_depth']})",
                "code": "",
            })

        critical_count = sum(1 for i in issues if i["severity"] == "critical")
        warning_count = sum(1 for i in issues if i["severity"] == "warning")
        info_count = sum(1 for i in issues if i["severity"] == "info")

        result = {
            "filename": filename,
            "total_issues": len(issues),
            "critical_count": critical_count,
            "warning_count": warning_count,
            "info_count": info_count,
            "issues": issues[:50],  # Cap at 50 issues
            "metrics": metrics,
        }

        self._log_review("analyze", result)

        severity_label = "CRITICAL" if critical_count > 0 else ("WARNING" if warning_count > 0 else "CLEAN")
        return SkillResult(
            success=True,
            message=f"Analysis of {filename}: {len(issues)} issues found [{severity_label}] "
                    f"({critical_count} critical, {warning_count} warnings, {info_count} info). "
                    f"Lines: {metrics['total_lines']}, Functions: {metrics['function_count']}, "
                    f"Classes: {metrics['class_count']}",
            data=result,
        )

    async def _review_diff(self, params: Dict) -> SkillResult:
        """Review a unified diff for issues."""
        diff = params.get("diff", "")
        context = params.get("context", "")

        if not diff.strip():
            return SkillResult(success=False, message="No diff provided")

        # Extract added lines from diff
        added_lines = []
        removed_lines = []
        current_file = None
        findings = []

        for line in diff.split("\n"):
            if line.startswith("+++ b/"):
                current_file = line[6:]
            elif line.startswith("--- a/"):
                pass
            elif line.startswith("+") and not line.startswith("+++"):
                added_code = line[1:]
                added_lines.append(added_code)

                # Check added lines against bug patterns
                for pattern, desc, severity in BUG_PATTERNS:
                    if re.search(pattern, added_code, re.IGNORECASE):
                        findings.append({
                            "file": current_file or "<unknown>",
                            "type": "bug_pattern",
                            "severity": severity,
                            "message": desc,
                            "line": added_code.strip()[:100],
                        })

                # Check security patterns
                for pattern, category, desc, severity in SECURITY_PATTERNS:
                    if re.search(pattern, added_code, re.IGNORECASE):
                        findings.append({
                            "file": current_file or "<unknown>",
                            "type": "security",
                            "severity": severity,
                            "message": f"[{category}] {desc}",
                            "line": added_code.strip()[:100],
                        })
            elif line.startswith("-") and not line.startswith("---"):
                removed_lines.append(line[1:])

        # Diff-level analysis
        added_count = len(added_lines)
        removed_count = len(removed_lines)

        # Check for large additions without tests
        added_code = "\n".join(added_lines)
        has_test_additions = bool(re.search(r"def test_|assert |pytest|unittest", added_code))
        if added_count > 20 and not has_test_additions:
            findings.append({
                "file": current_file or "<unknown>",
                "type": "best_practice",
                "severity": "info",
                "message": f"Large change ({added_count} lines added) with no test additions",
                "line": "",
            })

        # Try AST parse on the combined added code (may not work for partial code)
        if added_lines:
            ast_issues = self._ast_analysis(added_code, current_file or "<diff>")
            for issue in ast_issues:
                issue["type"] = "ast"
            findings.extend(ast_issues)

        critical_count = sum(1 for f in findings if f["severity"] == "critical")

        result = {
            "total_findings": len(findings),
            "critical_count": critical_count,
            "lines_added": added_count,
            "lines_removed": removed_count,
            "net_change": added_count - removed_count,
            "findings": findings[:50],
            "has_tests": has_test_additions,
        }

        self._log_review("review_diff", result)

        verdict = "REJECT" if critical_count > 0 else ("REVIEW" if len(findings) > 3 else "APPROVE")
        return SkillResult(
            success=True,
            message=f"Diff review: {len(findings)} findings [{verdict}]. "
                    f"+{added_count}/-{removed_count} lines. "
                    f"Critical: {critical_count}. Tests included: {has_test_additions}.",
            data=result,
        )

    async def _security_scan(self, params: Dict) -> SkillResult:
        """Scan code for security vulnerabilities."""
        code = params.get("code", "")
        filename = params.get("filename", "<input>")

        if not code.strip():
            return SkillResult(success=False, message="No code provided")

        vulnerabilities = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            for pattern, category, desc, severity in SECURITY_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerabilities.append({
                        "line": i,
                        "category": category,
                        "severity": severity,
                        "message": desc,
                        "code": stripped[:100],
                    })

        # Categorize
        by_category: Dict[str, int] = {}
        for v in vulnerabilities:
            cat = v["category"]
            by_category[cat] = by_category.get(cat, 0) + 1

        critical_count = sum(1 for v in vulnerabilities if v["severity"] == "critical")

        result = {
            "filename": filename,
            "total_vulnerabilities": len(vulnerabilities),
            "critical_count": critical_count,
            "by_category": by_category,
            "vulnerabilities": vulnerabilities[:50],
        }

        self._log_review("security_scan", result)

        if critical_count > 0:
            status = f"FAIL - {critical_count} critical vulnerabilities found"
        elif vulnerabilities:
            status = f"WARN - {len(vulnerabilities)} issues found (no critical)"
        else:
            status = "PASS - no security issues detected"

        return SkillResult(
            success=True,
            message=f"Security scan of {filename}: {status}. Categories: {by_category or 'none'}",
            data=result,
        )

    async def _score(self, params: Dict) -> SkillResult:
        """Score code quality on a 0-100 scale."""
        code = params.get("code", "")
        if not code.strip():
            return SkillResult(success=False, message="No code provided")

        metrics = self._compute_metrics(code)

        # Pattern issues
        issue_count = 0
        critical_count = 0
        for line in code.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            for pattern, _, severity in BUG_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    issue_count += 1
                    if severity == "critical":
                        critical_count += 1

        # Security issues
        sec_count = 0
        for line in code.split("\n"):
            for pattern, _, _, severity in SECURITY_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    sec_count += 1

        # AST issues
        ast_issues = self._ast_analysis(code, "<score>")

        # --- Scoring Categories ---

        # 1. Correctness (30 points) - critical bugs heavily penalized
        correctness = max(0, 30 - (critical_count * 10) - (len(ast_issues) * 3))

        # 2. Security (25 points)
        security = max(0, 25 - (sec_count * 5))

        # 3. Maintainability (25 points) - based on complexity and structure
        maint_deductions = 0
        if metrics["max_function_length"] > COMPLEXITY_THRESHOLDS["function_length"]:
            maint_deductions += 5
        if metrics["max_nesting_depth"] > COMPLEXITY_THRESHOLDS["nesting_depth"]:
            maint_deductions += 5
        if metrics["max_params"] > COMPLEXITY_THRESHOLDS["parameters"]:
            maint_deductions += 3
        if metrics["total_lines"] > 0 and metrics["docstring_count"] == 0:
            maint_deductions += 3
        if metrics["function_count"] > 0 and metrics["docstring_count"] / max(1, metrics["function_count"]) < 0.3:
            maint_deductions += 2
        maintainability = max(0, 25 - maint_deductions)

        # 4. Style (20 points) - minor issues
        style_deductions = min(15, issue_count - critical_count)  # Non-critical issues
        style = max(0, 20 - style_deductions)

        total_score = correctness + security + maintainability + style

        result = {
            "score": total_score,
            "breakdown": {
                "correctness": {"score": correctness, "max": 30},
                "security": {"score": security, "max": 25},
                "maintainability": {"score": maintainability, "max": 25},
                "style": {"score": style, "max": 20},
            },
            "metrics": metrics,
            "issue_counts": {
                "critical": critical_count,
                "security": sec_count,
                "ast": len(ast_issues),
                "style": max(0, issue_count - critical_count),
            },
        }

        self._log_review("score", result)

        grade = "A" if total_score >= 90 else "B" if total_score >= 75 else "C" if total_score >= 60 else "D" if total_score >= 40 else "F"
        return SkillResult(
            success=True,
            message=f"Quality Score: {total_score}/100 (Grade: {grade}). "
                    f"Correctness: {correctness}/30, Security: {security}/25, "
                    f"Maintainability: {maintainability}/25, Style: {style}/20.",
            data=result,
        )

    async def _history(self, params: Dict) -> SkillResult:
        """View review history and trends."""
        limit = params.get("limit", 10)
        reviews = self._load_reviews()

        if not reviews:
            return SkillResult(
                success=True,
                message="No code reviews recorded yet.",
                data={"reviews": [], "trends": {}},
            )

        recent = reviews[-limit:]

        # Compute trends
        scores = [r["summary"].get("score") for r in reviews if r["summary"].get("score") is not None]
        issues = [r["summary"].get("issues_found", 0) for r in reviews]
        criticals = [r["summary"].get("critical_count", 0) for r in reviews]

        trends = {
            "total_reviews": len(reviews),
            "avg_score": round(sum(s for s in scores) / len(scores), 1) if scores else None,
            "avg_issues": round(sum(issues) / len(issues), 1) if issues else 0,
            "total_criticals": sum(criticals),
            "review_types": {},
        }
        for r in reviews:
            t = r["type"]
            trends["review_types"][t] = trends["review_types"].get(t, 0) + 1

        # Score trend (last 5 vs previous 5)
        if len(scores) >= 10:
            recent_avg = sum(scores[-5:]) / 5
            prev_avg = sum(scores[-10:-5]) / 5
            trends["score_trend"] = "improving" if recent_avg > prev_avg else "declining"

        return SkillResult(
            success=True,
            message=f"Review history: {len(reviews)} total reviews. "
                    f"Avg score: {trends['avg_score']}, Avg issues: {trends['avg_issues']}, "
                    f"Total criticals: {trends['total_criticals']}.",
            data={"reviews": recent, "trends": trends},
        )

    # --- Analysis Helpers ---

    def _ast_analysis(self, code: str, filename: str) -> List[Dict]:
        """AST-based analysis for structural issues."""
        issues = []
        try:
            tree = ast.parse(code, filename=filename)
        except SyntaxError as e:
            issues.append({
                "line": e.lineno or 0,
                "severity": "critical",
                "message": f"Syntax error: {e.msg}",
                "code": str(e.text or "").strip()[:100],
            })
            return issues

        for node in ast.walk(tree):
            # Unused variables in assignments (simple check)
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                # Check for too many parameters
                args = node.args
                total_args = len(args.args) + len(args.posonlyargs) + len(args.kwonlyargs)
                if total_args > COMPLEXITY_THRESHOLDS["parameters"]:
                    issues.append({
                        "line": node.lineno,
                        "severity": "warning",
                        "message": f"Function '{node.name}' has {total_args} parameters (threshold: {COMPLEXITY_THRESHOLDS['parameters']})",
                        "code": f"def {node.name}(...)",
                    })

                # Check for empty function bodies (excluding pass/docstring)
                body = node.body
                real_stmts = [s for s in body if not (
                    isinstance(s, ast.Pass) or
                    (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))
                )]
                if not real_stmts and len(body) <= 1:
                    issues.append({
                        "line": node.lineno,
                        "severity": "info",
                        "message": f"Function '{node.name}' has no implementation (empty body)",
                        "code": f"def {node.name}(...)",
                    })

            # Check for mutable default arguments
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                for default in node.args.defaults + node.args.kw_defaults:
                    if default is not None and isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        issues.append({
                            "line": node.lineno,
                            "severity": "warning",
                            "message": f"Mutable default argument in '{node.name}' - use None and assign inside function",
                            "code": f"def {node.name}(...)",
                        })

            # Check for broad exception handling
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append({
                        "line": node.lineno,
                        "severity": "warning",
                        "message": "Bare except clause catches all exceptions",
                        "code": "except:",
                    })
                elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
                    # This is actually OK in many cases, just an info
                    pass

            # Detect unreachable code after return/raise
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for i, stmt in enumerate(node.body[:-1]):
                    if isinstance(stmt, (ast.Return, ast.Raise)):
                        next_stmt = node.body[i + 1]
                        if not isinstance(next_stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            issues.append({
                                "line": getattr(next_stmt, 'lineno', 0),
                                "severity": "warning",
                                "message": "Unreachable code after return/raise statement",
                                "code": "",
                            })

        return issues

    def _compute_metrics(self, code: str) -> Dict[str, Any]:
        """Compute code metrics."""
        lines = code.split("\n")
        total_lines = len(lines)
        blank_lines = sum(1 for l in lines if not l.strip())
        comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
        code_lines = total_lines - blank_lines - comment_lines

        # Count nesting depth
        max_depth = 0
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                depth = indent // 4  # Assume 4-space indentation
                max_depth = max(max_depth, depth)

        # AST-based metrics
        function_count = 0
        class_count = 0
        max_func_length = 0
        max_params = 0
        docstring_count = 0
        import_count = 0

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    function_count += 1
                    # Function length
                    if hasattr(node, 'end_lineno') and node.end_lineno:
                        func_len = node.end_lineno - node.lineno + 1
                        max_func_length = max(max_func_length, func_len)
                    # Parameters
                    args = node.args
                    nargs = len(args.args) + len(args.posonlyargs) + len(args.kwonlyargs)
                    max_params = max(max_params, nargs)
                    # Docstring
                    if (node.body and isinstance(node.body[0], ast.Expr) and
                            isinstance(node.body[0].value, ast.Constant) and
                            isinstance(node.body[0].value.value, str)):
                        docstring_count += 1
                elif isinstance(node, ast.ClassDef):
                    class_count += 1
                    if (node.body and isinstance(node.body[0], ast.Expr) and
                            isinstance(node.body[0].value, ast.Constant) and
                            isinstance(node.body[0].value.value, str)):
                        docstring_count += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_count += 1
        except SyntaxError:
            pass

        return {
            "total_lines": total_lines,
            "code_lines": code_lines,
            "blank_lines": blank_lines,
            "comment_lines": comment_lines,
            "function_count": function_count,
            "class_count": class_count,
            "import_count": import_count,
            "max_function_length": max_func_length,
            "max_nesting_depth": max_depth,
            "max_params": max_params,
            "docstring_count": docstring_count,
        }
