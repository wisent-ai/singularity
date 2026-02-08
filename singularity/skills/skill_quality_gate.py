"""
SkillQualityGateSkill - Automated quality validation and scoring for skills.

With 120+ skills in the ecosystem, maintaining consistent quality is critical.
This skill provides automated validation against best practices:

- Manifest completeness (skill_id, name, version, description, actions)
- Action parameter documentation quality
- Error handling patterns in execute()
- Test coverage detection (paired test file existence)
- Security patterns (credential checking, input validation)
- Code quality metrics (docstrings, complexity estimates)
- Inter-skill dependency tracking via SkillContext usage

Quality scores help identify skills that need improvement and establish
a quality bar for new skill submissions.
"""

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult

SKILLS_DIR = Path(__file__).parent
TESTS_DIR = Path(__file__).parent.parent.parent / "tests"


@dataclass
class QualityIssue:
    """A single quality issue found during validation."""
    severity: str  # "error", "warning", "info"
    category: str  # "manifest", "actions", "error_handling", "tests", "security", "docs"
    message: str
    line: int = 0
    suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {"severity": self.severity, "category": self.category, "message": self.message}
        if self.line:
            d["line"] = self.line
        if self.suggestion:
            d["suggestion"] = self.suggestion
        return d


@dataclass
class QualityReport:
    """Complete quality report for a skill."""
    skill_file: str
    score: float = 0.0
    grade: str = ""
    issues: List[QualityIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_file": self.skill_file,
            "score": round(self.score, 1),
            "grade": self.grade,
            "passed": self.passed,
            "issue_count": len(self.issues),
            "errors": sum(1 for i in self.issues if i.severity == "error"),
            "warnings": sum(1 for i in self.issues if i.severity == "warning"),
            "info": sum(1 for i in self.issues if i.severity == "info"),
            "issues": [i.to_dict() for i in self.issues],
            "metrics": self.metrics,
        }


def _score_to_grade(score: float) -> str:
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    return "F"


class SkillQualityAnalyzer:
    """Static analysis engine for skill quality validation."""

    # Minimum passing score
    PASSING_SCORE = 60.0

    # Category weights for final score
    WEIGHTS = {
        "manifest": 25,
        "actions": 20,
        "error_handling": 20,
        "tests": 15,
        "docs": 10,
        "security": 10,
    }

    def validate_file(self, filepath: str) -> QualityReport:
        """Run all validations on a skill file and produce a report."""
        report = QualityReport(skill_file=os.path.basename(filepath))
        path = Path(filepath)

        if not path.exists():
            report.issues.append(QualityIssue(
                severity="error", category="manifest",
                message=f"File not found: {filepath}"
            ))
            report.score = 0
            report.grade = "F"
            return report

        try:
            source = path.read_text(encoding="utf-8")
        except Exception as e:
            report.issues.append(QualityIssue(
                severity="error", category="manifest",
                message=f"Cannot read file: {e}"
            ))
            report.score = 0
            report.grade = "F"
            return report

        try:
            tree = ast.parse(source, filename=filepath)
        except SyntaxError as e:
            report.issues.append(QualityIssue(
                severity="error", category="manifest",
                message=f"Syntax error: {e}", line=e.lineno or 0
            ))
            report.score = 0
            report.grade = "F"
            return report

        # Collect metrics
        lines = source.split("\n")
        report.metrics["total_lines"] = len(lines)
        report.metrics["non_empty_lines"] = sum(1 for line in lines if line.strip())

        # Run all validation passes
        category_scores = {}
        category_scores["manifest"] = self._check_manifest(tree, source, report)
        category_scores["actions"] = self._check_actions(tree, source, report)
        category_scores["error_handling"] = self._check_error_handling(tree, source, report)
        category_scores["tests"] = self._check_tests(filepath, report)
        category_scores["docs"] = self._check_docs(tree, source, report)
        category_scores["security"] = self._check_security(tree, source, report)

        # Calculate weighted score
        total_weight = sum(self.WEIGHTS.values())
        weighted_score = sum(
            category_scores.get(cat, 0) * weight
            for cat, weight in self.WEIGHTS.items()
        ) / total_weight

        report.score = weighted_score
        report.grade = _score_to_grade(weighted_score)
        report.passed = weighted_score >= self.PASSING_SCORE
        report.metrics["category_scores"] = {
            cat: round(score, 1) for cat, score in category_scores.items()
        }

        return report

    def _find_skill_classes(self, tree: ast.Module) -> List[ast.ClassDef]:
        """Find all classes that likely inherit from Skill."""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = ""
                    if isinstance(base, ast.Name):
                        base_name = base.id
                    elif isinstance(base, ast.Attribute):
                        base_name = base.attr
                    if base_name == "Skill":
                        classes.append(node)
        return classes

    def _check_manifest(self, tree: ast.Module, source: str, report: QualityReport) -> float:
        """Check manifest property completeness. Returns score 0-100."""
        skill_classes = self._find_skill_classes(tree)
        if not skill_classes:
            report.issues.append(QualityIssue(
                severity="error", category="manifest",
                message="No Skill subclass found in file",
                suggestion="Define a class that inherits from Skill"
            ))
            return 0

        score = 100.0
        cls = skill_classes[0]
        report.metrics["skill_class"] = cls.name

        # Check for manifest property
        has_manifest = False
        for item in cls.body:
            if isinstance(item, ast.FunctionDef) and item.name == "manifest":
                # Check if it's a property (has @property decorator)
                for dec in item.decorator_list:
                    if isinstance(dec, ast.Name) and dec.id == "property":
                        has_manifest = True
                        break
                if not has_manifest:
                    # Also check if it returns SkillManifest
                    has_manifest = True  # Found manifest method at least
                break

        if not has_manifest:
            report.issues.append(QualityIssue(
                severity="error", category="manifest",
                message="Missing manifest property",
                suggestion="Add @property def manifest(self) -> SkillManifest"
            ))
            score -= 40

        # Check manifest content from source via regex (more robust than AST for return values)
        manifest_fields = {
            "skill_id": r'skill_id\s*=',
            "name": r'(?<!skill_)name\s*=',
            "version": r'version\s*=',
            "description": r'description\s*=',
            "actions": r'actions\s*=',
            "category": r'category\s*=',
        }

        # Extract manifest block from source
        manifest_section = self._extract_manifest_source(source)
        found_fields = 0
        for field_name, pattern in manifest_fields.items():
            if manifest_section and re.search(pattern, manifest_section):
                found_fields += 1
            else:
                severity = "error" if field_name in ("skill_id", "name", "actions") else "warning"
                deduction = 15 if severity == "error" else 5
                report.issues.append(QualityIssue(
                    severity=severity, category="manifest",
                    message=f"Manifest missing '{field_name}' field",
                    suggestion=f"Add {field_name} to SkillManifest constructor"
                ))
                score -= deduction

        report.metrics["manifest_fields_found"] = found_fields
        report.metrics["manifest_fields_total"] = len(manifest_fields)

        # Check for __init__ calling super().__init__
        has_super_init = False
        for item in cls.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                for node in ast.walk(item):
                    if isinstance(node, ast.Call):
                        func = node.func
                        if isinstance(func, ast.Attribute) and func.attr == "__init__":
                            if isinstance(func.value, ast.Call):
                                if isinstance(func.value.func, ast.Name) and func.value.func.id == "super":
                                    has_super_init = True
                break

        if not has_super_init:
            report.issues.append(QualityIssue(
                severity="warning", category="manifest",
                message="__init__ does not call super().__init__()",
                suggestion="Add super().__init__(credentials) in __init__"
            ))
            score -= 10

        return max(score, 0)

    def _extract_manifest_source(self, source: str) -> Optional[str]:
        """Extract the manifest property source block."""
        lines = source.split("\n")
        in_manifest = False
        depth = 0
        manifest_lines = []
        for line in lines:
            if "def manifest" in line and "property" in source[:source.index("def manifest")] if "def manifest" in source else False:
                in_manifest = True
            if in_manifest:
                manifest_lines.append(line)
                depth += line.count("(") - line.count(")")
                if depth <= 0 and len(manifest_lines) > 1 and line.strip() and not line.strip().startswith("#"):
                    # Check if we're past the return statement
                    if "return" in "".join(manifest_lines) and depth <= 0:
                        break
        return "\n".join(manifest_lines) if manifest_lines else None

    def _check_actions(self, tree: ast.Module, source: str, report: QualityReport) -> float:
        """Check action implementations. Returns score 0-100."""
        skill_classes = self._find_skill_classes(tree)
        if not skill_classes:
            return 0

        score = 100.0
        cls = skill_classes[0]

        # Check for execute method
        has_execute = False
        execute_node = None
        for item in cls.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "execute":
                has_execute = True
                execute_node = item
                break

        if not has_execute:
            report.issues.append(QualityIssue(
                severity="error", category="actions",
                message="Missing execute() method",
                suggestion="Implement async def execute(self, action: str, params: Dict) -> SkillResult"
            ))
            return 0

        # Check execute is async
        if isinstance(execute_node, ast.FunctionDef) and not isinstance(execute_node, ast.AsyncFunctionDef):
            report.issues.append(QualityIssue(
                severity="warning", category="actions",
                message="execute() is not async",
                suggestion="Use async def execute() for consistency",
                line=execute_node.lineno
            ))
            score -= 10

        # Count action branches in execute
        action_count = 0
        for node in ast.walk(execute_node):
            if isinstance(node, ast.Compare):
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                        action_count += 1

        report.metrics["action_branches"] = action_count

        if action_count == 0:
            report.issues.append(QualityIssue(
                severity="warning", category="actions",
                message="No action routing detected in execute()",
                suggestion="Route actions with if/elif: if action == 'action_name': ..."
            ))
            score -= 15

        # Check for unknown action handling
        source_lower = source.lower()
        has_unknown_handler = (
            "unknown action" in source_lower or
            "unsupported action" in source_lower or
            "invalid action" in source_lower or
            "not found" in source_lower or
            "else:" in self._get_method_source(source, "execute")
        )
        if not has_unknown_handler:
            report.issues.append(QualityIssue(
                severity="warning", category="actions",
                message="No handler for unknown actions in execute()",
                suggestion="Add else clause returning SkillResult(success=False, message=f'Unknown action: {action}')"
            ))
            score -= 10

        # Check for SkillResult usage
        skill_result_count = source.count("SkillResult(")
        report.metrics["skill_result_usages"] = skill_result_count
        if skill_result_count == 0:
            report.issues.append(QualityIssue(
                severity="error", category="actions",
                message="No SkillResult returns found",
                suggestion="Return SkillResult objects from execute()"
            ))
            score -= 30

        # Check for private helper methods per action (good practice)
        private_methods = [
            item.name for item in cls.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name.startswith("_") and not item.name.startswith("__")
        ]
        report.metrics["private_methods"] = len(private_methods)
        if action_count > 2 and len(private_methods) < action_count // 2:
            report.issues.append(QualityIssue(
                severity="info", category="actions",
                message=f"Consider extracting {action_count} actions into private helper methods",
                suggestion="Use _action_name() methods for better readability"
            ))
            score -= 5

        return max(score, 0)

    def _get_method_source(self, source: str, method_name: str) -> str:
        """Extract source for a method by name."""
        lines = source.split("\n")
        in_method = False
        method_lines = []
        indent = 0
        for line in lines:
            if re.match(rf'\s+(?:async\s+)?def\s+{method_name}\b', line):
                in_method = True
                indent = len(line) - len(line.lstrip())
                method_lines.append(line)
                continue
            if in_method:
                if line.strip() == "":
                    method_lines.append(line)
                elif len(line) - len(line.lstrip()) > indent:
                    method_lines.append(line)
                else:
                    break
        return "\n".join(method_lines)

    def _check_error_handling(self, tree: ast.Module, source: str, report: QualityReport) -> float:
        """Check error handling patterns. Returns score 0-100."""
        score = 100.0

        # Count try/except blocks
        try_count = 0
        except_count = 0
        bare_except_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                try_count += 1
                for handler in node.handlers:
                    except_count += 1
                    if handler.type is None:
                        bare_except_count += 1

        report.metrics["try_blocks"] = try_count
        report.metrics["except_handlers"] = except_count
        report.metrics["bare_excepts"] = bare_except_count

        # Check for try/except in execute
        execute_source = self._get_method_source(source, "execute")
        if "try:" not in execute_source and execute_source:
            report.issues.append(QualityIssue(
                severity="warning", category="error_handling",
                message="execute() has no try/except error handling",
                suggestion="Wrap action execution in try/except to return SkillResult on error"
            ))
            score -= 25

        # Check for bare except clauses
        if bare_except_count > 0:
            report.issues.append(QualityIssue(
                severity="warning", category="error_handling",
                message=f"Found {bare_except_count} bare except clause(s) — catches all exceptions",
                suggestion="Catch specific exceptions: except (ValueError, KeyError) as e:"
            ))
            score -= 10 * min(bare_except_count, 3)

        # Overall try/except coverage relative to file size
        lines_count = report.metrics.get("total_lines", 0)
        if lines_count > 100 and try_count == 0:
            report.issues.append(QualityIssue(
                severity="warning", category="error_handling",
                message=f"No try/except blocks in {lines_count}-line file",
                suggestion="Add error handling for robustness"
            ))
            score -= 20

        return max(score, 0)

    def _check_tests(self, filepath: str, report: QualityReport) -> float:
        """Check for corresponding test file. Returns score 0-100."""
        score = 0.0
        skill_name = Path(filepath).stem

        # Check standard test file locations
        test_file = TESTS_DIR / f"test_{skill_name}.py"
        report.metrics["test_file_expected"] = str(test_file)
        report.metrics["test_file_exists"] = test_file.exists()

        if test_file.exists():
            score = 60.0
            try:
                test_source = test_file.read_text(encoding="utf-8")
                test_lines = len(test_source.split("\n"))
                test_count = len(re.findall(r'def test_', test_source))
                test_class_count = len(re.findall(r'class Test', test_source))

                report.metrics["test_lines"] = test_lines
                report.metrics["test_count"] = test_count
                report.metrics["test_classes"] = test_class_count

                # Score based on test count
                if test_count >= 20:
                    score = 100.0
                elif test_count >= 10:
                    score = 85.0
                elif test_count >= 5:
                    score = 75.0
                else:
                    score = 65.0

                # Check for async test support
                if "pytest.mark.asyncio" in test_source or "async def test_" in test_source:
                    report.metrics["has_async_tests"] = True
                else:
                    report.metrics["has_async_tests"] = False

            except Exception:
                score = 50.0
        else:
            report.issues.append(QualityIssue(
                severity="warning", category="tests",
                message=f"No test file found at {test_file.name}",
                suggestion=f"Create tests/test_{skill_name}.py with pytest tests"
            ))

        return score

    def _check_docs(self, tree: ast.Module, source: str, report: QualityReport) -> float:
        """Check documentation quality. Returns score 0-100."""
        score = 100.0

        # Check module docstring
        if not ast.get_docstring(tree):
            report.issues.append(QualityIssue(
                severity="warning", category="docs",
                message="Missing module docstring",
                suggestion="Add a module-level docstring describing the skill's purpose"
            ))
            score -= 20

        # Check class docstring
        skill_classes = self._find_skill_classes(tree)
        if skill_classes:
            cls = skill_classes[0]
            if not ast.get_docstring(cls):
                report.issues.append(QualityIssue(
                    severity="warning", category="docs",
                    message=f"Class {cls.name} missing docstring",
                    suggestion="Add a class docstring describing the skill"
                ))
                score -= 20

            # Check method docstrings
            methods_total = 0
            methods_documented = 0
            for item in cls.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not item.name.startswith("_"):
                        methods_total += 1
                        if ast.get_docstring(item):
                            methods_documented += 1

            report.metrics["public_methods"] = methods_total
            report.metrics["documented_methods"] = methods_documented

            if methods_total > 0:
                doc_ratio = methods_documented / methods_total
                if doc_ratio < 0.5:
                    report.issues.append(QualityIssue(
                        severity="info", category="docs",
                        message=f"Only {methods_documented}/{methods_total} public methods documented",
                        suggestion="Add docstrings to public methods"
                    ))
                    score -= 15 * (1 - doc_ratio)

        # Check for type hints
        type_hint_count = source.count("->") + source.count(": str") + source.count(": Dict") + source.count(": List") + source.count(": int") + source.count(": float") + source.count(": bool") + source.count(": Optional")
        report.metrics["type_hint_indicators"] = type_hint_count
        lines_count = report.metrics.get("total_lines", 0)
        if lines_count > 50 and type_hint_count < 3:
            report.issues.append(QualityIssue(
                severity="info", category="docs",
                message="Minimal type hints detected",
                suggestion="Add type hints to function signatures for clarity"
            ))
            score -= 10

        return max(score, 0)

    def _check_security(self, tree: ast.Module, source: str, report: QualityReport) -> float:
        """Check security patterns. Returns score 0-100."""
        score = 100.0

        skill_classes = self._find_skill_classes(tree)
        if not skill_classes:
            return score

        cls = skill_classes[0]

        # Check for credential validation
        has_check_credentials = any(
            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == "check_credentials"
            for item in cls.body
        )

        # Check if required_credentials is non-empty in manifest
        has_required_creds = "required_credentials" in source and "required_credentials=[]" not in source.replace(" ", "")

        if has_required_creds and not has_check_credentials:
            report.issues.append(QualityIssue(
                severity="info", category="security",
                message="Has required credentials but no custom check_credentials()",
                suggestion="Override check_credentials() for custom validation logic"
            ))
            # Not a deduction — base class handles it

        # Check for hardcoded secrets
        secret_patterns = [
            (r'["\']sk-[a-zA-Z0-9]{20,}["\']', "Possible OpenAI API key"),
            (r'["\']ghp_[a-zA-Z0-9]{30,}["\']', "Possible GitHub token"),
            (r'["\'][A-Za-z0-9+/]{40,}={0,2}["\']', "Possible base64-encoded secret"),
            (r'password\s*=\s*["\'][^"\']{8,}["\']', "Possible hardcoded password"),
        ]

        for pattern, desc in secret_patterns:
            matches = re.findall(pattern, source)
            if matches:
                report.issues.append(QualityIssue(
                    severity="error", category="security",
                    message=f"Potential hardcoded secret: {desc}",
                    suggestion="Use credentials dict instead of hardcoded values"
                ))
                score -= 25

        # Check for dangerous operations without validation
        dangerous_patterns = [
            ("eval(", "Use of eval() — code injection risk"),
            ("exec(", "Use of exec() — code injection risk"),
            ("__import__", "Dynamic import — potential security risk"),
            ("os.system(", "os.system() — use subprocess instead"),
        ]

        for pattern, desc in dangerous_patterns:
            if pattern in source:
                report.issues.append(QualityIssue(
                    severity="warning", category="security",
                    message=desc,
                    suggestion="Consider safer alternatives"
                ))
                score -= 15

        # Check input validation
        execute_source = self._get_method_source(source, "execute")
        if execute_source:
            has_param_validation = (
                "params.get(" in execute_source or
                ".get(" in execute_source or
                "if not " in execute_source or
                "if action" in execute_source
            )
            if not has_param_validation:
                report.issues.append(QualityIssue(
                    severity="info", category="security",
                    message="No parameter validation detected in execute()",
                    suggestion="Validate required parameters before use"
                ))
                score -= 10

        return max(score, 0)


class SkillQualityGateSkill(Skill):
    """
    Automated quality validation and scoring for the Singularity skill ecosystem.

    Analyzes skill source files against best practices and produces quality reports
    with scores, grades, and actionable improvement suggestions. Useful for:
    - Pre-merge validation of new skills
    - Ecosystem-wide quality audits
    - Identifying skills that need improvement
    - Establishing and enforcing quality standards
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._analyzer = SkillQualityAnalyzer()
        self._reports: Dict[str, QualityReport] = {}

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="skill_quality_gate",
            name="Skill Quality Gate",
            version="1.0.0",
            category="development",
            description="Automated quality validation and scoring for skills",
            required_credentials=[],
            actions=[
                SkillAction(
                    name="validate",
                    description="Validate a single skill file and produce a quality report",
                    parameters={
                        "skill_file": {"type": "string", "required": True,
                                       "description": "Skill filename (e.g., 'shell.py') or full path"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                ),
                SkillAction(
                    name="audit",
                    description="Audit all skills in the ecosystem and rank by quality",
                    parameters={
                        "min_score": {"type": "float", "required": False,
                                      "description": "Only show skills below this score (default: 100)"},
                        "category": {"type": "string", "required": False,
                                     "description": "Filter by issue category"},
                        "limit": {"type": "int", "required": False,
                                  "description": "Max skills to audit (default: all)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=10,
                ),
                SkillAction(
                    name="compare",
                    description="Compare quality scores across skills",
                    parameters={
                        "skill_files": {"type": "list", "required": False,
                                        "description": "List of skill filenames to compare (default: all)"},
                        "sort_by": {"type": "string", "required": False,
                                    "description": "Sort by: score, tests, errors (default: score)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=5,
                ),
                SkillAction(
                    name="gate_check",
                    description="Pass/fail quality gate check for a skill (CI-friendly)",
                    parameters={
                        "skill_file": {"type": "string", "required": True,
                                       "description": "Skill filename to check"},
                        "min_score": {"type": "float", "required": False,
                                      "description": "Minimum passing score (default: 60)"},
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                ),
                SkillAction(
                    name="stats",
                    description="Get aggregate quality statistics for the ecosystem",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=5,
                ),
            ],
        )

    async def execute(self, action: str, params: Dict) -> SkillResult:
        try:
            if action == "validate":
                return self._validate(params)
            elif action == "audit":
                return self._audit(params)
            elif action == "compare":
                return self._compare(params)
            elif action == "gate_check":
                return self._gate_check(params)
            elif action == "stats":
                return self._stats(params)
            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, message=f"Error: {e}")

    def _resolve_skill_path(self, skill_file: str) -> Path:
        """Resolve a skill filename to a full path."""
        path = Path(skill_file)
        if path.is_absolute() and path.exists():
            return path
        # Try skills directory
        candidate = SKILLS_DIR / skill_file
        if candidate.exists():
            return candidate
        # Try with .py extension
        candidate = SKILLS_DIR / f"{skill_file}.py"
        if candidate.exists():
            return candidate
        return Path(skill_file)

    def _validate(self, params: Dict) -> SkillResult:
        """Validate a single skill file."""
        skill_file = params.get("skill_file", "")
        if not skill_file:
            return SkillResult(success=False, message="skill_file parameter required")

        path = self._resolve_skill_path(skill_file)
        report = self._analyzer.validate_file(str(path))
        self._reports[report.skill_file] = report

        return SkillResult(
            success=True,
            message=f"{report.skill_file}: {report.grade} ({report.score:.1f}/100) — "
                    f"{len(report.issues)} issues ({sum(1 for i in report.issues if i.severity == 'error')} errors, "
                    f"{sum(1 for i in report.issues if i.severity == 'warning')} warnings)",
            data=report.to_dict(),
        )

    def _audit(self, params: Dict) -> SkillResult:
        """Audit all skills and produce ecosystem report."""
        min_score = params.get("min_score", 100.0)
        category = params.get("category")
        limit = params.get("limit")

        skill_files = sorted(SKILLS_DIR.glob("*.py"))
        # Skip non-skill files
        skip = {"__init__.py", "base.py"}
        skill_files = [f for f in skill_files if f.name not in skip]

        if limit:
            skill_files = skill_files[:int(limit)]

        reports = []
        for sf in skill_files:
            report = self._analyzer.validate_file(str(sf))
            self._reports[report.skill_file] = report

            # Filter by score threshold
            if report.score > min_score:
                continue

            # Filter by category if specified
            if category:
                has_category = any(i.category == category for i in report.issues)
                if not has_category:
                    continue

            reports.append(report)

        # Sort by score ascending (worst first)
        reports.sort(key=lambda r: r.score)

        summary = {
            "total_skills_audited": len(skill_files),
            "skills_below_threshold": len(reports),
            "min_score_filter": min_score,
            "avg_score": round(
                sum(r.score for r in self._reports.values()) / max(len(self._reports), 1), 1
            ),
            "grade_distribution": {},
            "top_issues": self._aggregate_issues(reports),
            "skills": [r.to_dict() for r in reports[:50]],  # Cap at 50
        }

        # Grade distribution across ALL audited skills
        for report in self._reports.values():
            grade = report.grade
            summary["grade_distribution"][grade] = summary["grade_distribution"].get(grade, 0) + 1

        return SkillResult(
            success=True,
            message=f"Audited {len(skill_files)} skills. "
                    f"{len(reports)} below {min_score} threshold. "
                    f"Average score: {summary['avg_score']}",
            data=summary,
        )

    def _aggregate_issues(self, reports: List[QualityReport]) -> List[Dict]:
        """Aggregate common issues across reports."""
        issue_counts: Dict[str, int] = {}
        for report in reports:
            for issue in report.issues:
                key = f"{issue.category}:{issue.message}"
                issue_counts[key] = issue_counts.get(key, 0) + 1

        top = sorted(issue_counts.items(), key=lambda x: -x[1])[:10]
        return [{"issue": k, "count": v} for k, v in top]

    def _compare(self, params: Dict) -> SkillResult:
        """Compare quality across skills."""
        skill_files = params.get("skill_files", [])
        sort_by = params.get("sort_by", "score")

        if not skill_files:
            # Compare all
            all_files = sorted(SKILLS_DIR.glob("*.py"))
            skip = {"__init__.py", "base.py"}
            skill_files = [f.name for f in all_files if f.name not in skip]

        results = []
        for sf in skill_files:
            path = self._resolve_skill_path(sf)
            report = self._analyzer.validate_file(str(path))
            self._reports[report.skill_file] = report
            results.append({
                "file": report.skill_file,
                "score": round(report.score, 1),
                "grade": report.grade,
                "errors": sum(1 for i in report.issues if i.severity == "error"),
                "warnings": sum(1 for i in report.issues if i.severity == "warning"),
                "has_tests": report.metrics.get("test_file_exists", False),
                "test_count": report.metrics.get("test_count", 0),
                "lines": report.metrics.get("total_lines", 0),
            })

        # Sort
        if sort_by == "tests":
            results.sort(key=lambda r: r["test_count"], reverse=True)
        elif sort_by == "errors":
            results.sort(key=lambda r: r["errors"], reverse=True)
        else:
            results.sort(key=lambda r: r["score"], reverse=True)

        return SkillResult(
            success=True,
            message=f"Compared {len(results)} skills. Top: {results[0]['file']} ({results[0]['score']}), "
                    f"Bottom: {results[-1]['file']} ({results[-1]['score']})" if results else "No skills found",
            data={
                "skills": results,
                "total": len(results),
                "avg_score": round(sum(r["score"] for r in results) / max(len(results), 1), 1),
                "with_tests": sum(1 for r in results if r["has_tests"]),
                "without_tests": sum(1 for r in results if not r["has_tests"]),
            },
        )

    def _gate_check(self, params: Dict) -> SkillResult:
        """CI-friendly pass/fail gate check."""
        skill_file = params.get("skill_file", "")
        min_score = params.get("min_score", 60.0)

        if not skill_file:
            return SkillResult(success=False, message="skill_file parameter required")

        path = self._resolve_skill_path(skill_file)
        report = self._analyzer.validate_file(str(path))
        self._reports[report.skill_file] = report

        passed = report.score >= min_score
        errors = [i for i in report.issues if i.severity == "error"]

        return SkillResult(
            success=passed,
            message=f"{'PASS' if passed else 'FAIL'}: {report.skill_file} scored {report.score:.1f} "
                    f"(min: {min_score}). Grade: {report.grade}. "
                    f"{len(errors)} error(s).",
            data={
                "passed": passed,
                "score": round(report.score, 1),
                "grade": report.grade,
                "min_score": min_score,
                "errors": [e.to_dict() for e in errors],
                "all_issues": [i.to_dict() for i in report.issues],
                "metrics": report.metrics,
            },
        )

    def _stats(self, params: Dict) -> SkillResult:
        """Get aggregate ecosystem quality statistics."""
        # Audit all skills first if not already done
        all_files = sorted(SKILLS_DIR.glob("*.py"))
        skip = {"__init__.py", "base.py"}
        skill_files = [f for f in all_files if f.name not in skip]

        for sf in skill_files:
            if sf.name not in self._reports:
                report = self._analyzer.validate_file(str(sf))
                self._reports[sf.name] = report

        scores = [r.score for r in self._reports.values()]
        if not scores:
            return SkillResult(success=True, message="No skills found", data={})

        grades = {}
        categories_with_issues: Dict[str, int] = {}
        total_issues = 0
        total_errors = 0
        total_warnings = 0
        skills_with_tests = 0
        total_test_count = 0

        for report in self._reports.values():
            grades[report.grade] = grades.get(report.grade, 0) + 1
            for issue in report.issues:
                total_issues += 1
                if issue.severity == "error":
                    total_errors += 1
                elif issue.severity == "warning":
                    total_warnings += 1
                categories_with_issues[issue.category] = categories_with_issues.get(issue.category, 0) + 1

            if report.metrics.get("test_file_exists"):
                skills_with_tests += 1
                total_test_count += report.metrics.get("test_count", 0)

        stats = {
            "total_skills": len(scores),
            "avg_score": round(sum(scores) / len(scores), 1),
            "median_score": round(sorted(scores)[len(scores) // 2], 1),
            "min_score": round(min(scores), 1),
            "max_score": round(max(scores), 1),
            "passing": sum(1 for s in scores if s >= 60),
            "failing": sum(1 for s in scores if s < 60),
            "grade_distribution": dict(sorted(grades.items())),
            "total_issues": total_issues,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "issue_categories": dict(sorted(categories_with_issues.items(), key=lambda x: -x[1])),
            "test_coverage": {
                "skills_with_tests": skills_with_tests,
                "skills_without_tests": len(scores) - skills_with_tests,
                "coverage_pct": round(skills_with_tests / len(scores) * 100, 1),
                "total_test_count": total_test_count,
            },
        }

        return SkillResult(
            success=True,
            message=f"Ecosystem: {len(scores)} skills, avg score {stats['avg_score']}, "
                    f"{stats['passing']} passing, {stats['failing']} failing. "
                    f"Test coverage: {stats['test_coverage']['coverage_pct']}%",
            data=stats,
        )
