#!/usr/bin/env python3
"""
Code Analysis Skill - Static analysis for Python code.

Uses Python's built-in AST module to analyze code without external dependencies.
Provides:
- Code quality metrics (complexity, LOC, functions, classes)
- Issue detection (unused imports, bare excepts, mutable defaults, etc.)
- Dependency analysis (imports and their relationships)
- Security scanning (dangerous function calls, hardcoded secrets)
- Code review with actionable suggestions

Serves Revenue (offer code review as a service) and Self-Improvement
(agent can analyze its own code to find improvements).
"""

import ast
import os
import re
from typing import Dict, List, Optional, Any
from .base import Skill, SkillManifest, SkillAction, SkillResult


# Dangerous function patterns for security scanning
DANGEROUS_CALLS = {
    "eval": "Arbitrary code execution via eval()",
    "exec": "Arbitrary code execution via exec()",
    "compile": "Dynamic code compilation",
    "__import__": "Dynamic import (potential injection)",
    "subprocess.call": "Shell command execution (use subprocess.run with shell=False)",
    "subprocess.Popen": "Shell command execution",
    "os.system": "Shell command execution via os.system()",
    "os.popen": "Shell command execution via os.popen()",
    "pickle.loads": "Deserialization attack vector (pickle)",
    "pickle.load": "Deserialization attack vector (pickle)",
    "yaml.load": "Unsafe YAML loading (use yaml.safe_load)",
    "marshal.loads": "Deserialization attack vector (marshal)",
}

# Patterns that might indicate hardcoded secrets
SECRET_PATTERNS = [
    (r'(?i)(password|passwd|pwd)\s*=\s*["\'][^"\']+["\']', "Possible hardcoded password"),
    (r'(?i)(api_key|apikey|secret_key|secret)\s*=\s*["\'][^"\']+["\']', "Possible hardcoded API key"),
    (r'(?i)(token)\s*=\s*["\'][a-zA-Z0-9_\-]{20,}["\']', "Possible hardcoded token"),
    (r'(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}', "Possible GitHub token"),
    (r'sk-[A-Za-z0-9]{32,}', "Possible OpenAI API key"),
    (r'AKIA[0-9A-Z]{16}', "Possible AWS access key"),
]


class CodeAnalysisSkill(Skill):
    """
    Static code analysis skill using Python's AST module.

    No external dependencies required - uses only Python builtins.
    """

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="code_analysis",
            name="Code Analysis",
            version="1.0.0",
            category="dev",
            description="Static Python code analysis: metrics, issues, security, dependencies",
            actions=[
                SkillAction(
                    name="analyze",
                    description="Full analysis of a Python file: metrics, issues, and suggestions",
                    parameters={
                        "path": {
                            "type": "string",
                            "required": True,
                            "description": "Path to Python file to analyze",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="analyze_code",
                    description="Analyze Python code from a string (no file needed)",
                    parameters={
                        "code": {
                            "type": "string",
                            "required": True,
                            "description": "Python code to analyze",
                        },
                        "filename": {
                            "type": "string",
                            "required": False,
                            "description": "Optional filename for reporting",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="metrics",
                    description="Compute code metrics: LOC, complexity, function/class counts",
                    parameters={
                        "path": {
                            "type": "string",
                            "required": True,
                            "description": "Path to Python file",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="issues",
                    description="Find code quality issues (bare excepts, mutable defaults, etc.)",
                    parameters={
                        "path": {
                            "type": "string",
                            "required": True,
                            "description": "Path to Python file",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="security",
                    description="Security scan: dangerous calls, hardcoded secrets, injection risks",
                    parameters={
                        "path": {
                            "type": "string",
                            "required": True,
                            "description": "Path to Python file",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="dependencies",
                    description="Analyze imports and dependencies",
                    parameters={
                        "path": {
                            "type": "string",
                            "required": True,
                            "description": "Path to Python file",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="complexity",
                    description="Detailed complexity analysis per function/method",
                    parameters={
                        "path": {
                            "type": "string",
                            "required": True,
                            "description": "Path to Python file",
                        },
                        "threshold": {
                            "type": "integer",
                            "required": False,
                            "description": "Min complexity to report (default: 5)",
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="scan_directory",
                    description="Scan all Python files in a directory for issues",
                    parameters={
                        "path": {
                            "type": "string",
                            "required": True,
                            "description": "Directory path to scan",
                        },
                        "recursive": {
                            "type": "boolean",
                            "required": False,
                            "description": "Scan subdirectories (default: true)",
                        },
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True  # No credentials needed

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "analyze": self._analyze,
            "analyze_code": self._analyze_code,
            "metrics": self._metrics,
            "issues": self._issues,
            "security": self._security,
            "dependencies": self._dependencies,
            "complexity": self._complexity,
            "scan_directory": self._scan_directory,
        }

        handler = handlers.get(action)
        if handler:
            return await handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    def _read_file(self, path: str) -> tuple:
        """Read a file and return (content, error_result)."""
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            return None, SkillResult(success=False, message=f"File not found: {path}")
        if not path.endswith(".py"):
            return None, SkillResult(success=False, message=f"Not a Python file: {path}")
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read(), None
        except Exception as e:
            return None, SkillResult(success=False, message=f"Error reading file: {e}")

    def _parse_ast(self, code: str, filename: str = "<string>") -> tuple:
        """Parse code into AST and return (tree, error_result)."""
        try:
            tree = ast.parse(code, filename=filename)
            return tree, None
        except SyntaxError as e:
            return None, SkillResult(
                success=False,
                message=f"Syntax error at line {e.lineno}: {e.msg}",
                data={"line": e.lineno, "offset": e.offset, "error": e.msg},
            )

    # ──────────────────────────────────────────────
    # Metrics
    # ──────────────────────────────────────────────

    def _compute_metrics(self, code: str, tree: ast.AST) -> Dict:
        """Compute code metrics from source and AST."""
        lines = code.splitlines()
        total_lines = len(lines)
        blank_lines = sum(1 for line in lines if not line.strip())
        comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
        code_lines = total_lines - blank_lines - comment_lines

        # Count structures
        functions = []
        classes = []
        methods = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                # Check if it's a method (inside a class)
                is_method = False
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.ClassDef):
                        for child in ast.iter_child_nodes(parent):
                            if child is node:
                                is_method = True
                                break
                if is_method:
                    methods.append(node.name)
                else:
                    functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)

        # Count decorators
        decorator_count = 0
        for node in ast.walk(tree):
            if hasattr(node, "decorator_list"):
                decorator_count += len(node.decorator_list)

        # Count assertions
        assert_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Assert))

        # Average function length
        func_lengths = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if hasattr(node, "end_lineno") and node.end_lineno:
                    func_lengths.append(node.end_lineno - node.lineno + 1)
        avg_func_length = (
            round(sum(func_lengths) / len(func_lengths), 1) if func_lengths else 0
        )

        # Total complexity
        total_complexity = self._calculate_complexity(tree)

        return {
            "total_lines": total_lines,
            "code_lines": code_lines,
            "blank_lines": blank_lines,
            "comment_lines": comment_lines,
            "comment_ratio": round(comment_lines / max(code_lines, 1), 3),
            "functions": len(functions),
            "methods": len(methods),
            "classes": len(classes),
            "decorators": decorator_count,
            "assertions": assert_count,
            "avg_function_length": avg_func_length,
            "total_complexity": total_complexity,
            "function_names": functions,
            "class_names": classes,
        }

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of an AST node."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.IfExp)):
                complexity += 1
            elif isinstance(child, (ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, (ast.While,)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                # Each 'and'/'or' adds a branch
                complexity += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
                complexity += len(child.ifs)

        return complexity

    def _function_complexity(self, tree: ast.AST) -> List[Dict]:
        """Get complexity per function/method."""
        results = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cx = self._calculate_complexity(node)
                line_count = 0
                if hasattr(node, "end_lineno") and node.end_lineno:
                    line_count = node.end_lineno - node.lineno + 1
                results.append({
                    "name": node.name,
                    "line": node.lineno,
                    "complexity": cx,
                    "lines": line_count,
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "args": len(node.args.args),
                    "rating": (
                        "simple" if cx <= 5
                        else "moderate" if cx <= 10
                        else "complex" if cx <= 20
                        else "very complex"
                    ),
                })
        results.sort(key=lambda x: x["complexity"], reverse=True)
        return results

    # ──────────────────────────────────────────────
    # Issue Detection
    # ──────────────────────────────────────────────

    def _find_issues(self, code: str, tree: ast.AST) -> List[Dict]:
        """Find code quality issues."""
        issues = []

        for node in ast.walk(tree):
            # Bare except
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append({
                    "line": node.lineno,
                    "severity": "warning",
                    "code": "W001",
                    "message": "Bare except clause - catches all exceptions including SystemExit and KeyboardInterrupt",
                    "suggestion": "Use 'except Exception:' instead",
                })

            # except Exception: pass (exception swallowing)
            if isinstance(node, ast.ExceptHandler):
                if (
                    len(node.body) == 1
                    and isinstance(node.body[0], ast.Pass)
                ):
                    issues.append({
                        "line": node.lineno,
                        "severity": "warning",
                        "code": "W002",
                        "message": "Exception silently swallowed with pass",
                        "suggestion": "Log the exception or handle it explicitly",
                    })

            # Mutable default arguments
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults + node.args.kw_defaults:
                    if default and isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        issues.append({
                            "line": node.lineno,
                            "severity": "error",
                            "code": "E001",
                            "message": f"Mutable default argument in {node.name}()",
                            "suggestion": "Use None as default and create mutable in function body",
                        })

            # Star imports
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        issues.append({
                            "line": node.lineno,
                            "severity": "warning",
                            "code": "W003",
                            "message": f"Star import from {node.module}",
                            "suggestion": "Import specific names to avoid namespace pollution",
                        })

            # Global statement usage
            if isinstance(node, ast.Global):
                issues.append({
                    "line": node.lineno,
                    "severity": "info",
                    "code": "I001",
                    "message": f"Global statement: {', '.join(node.names)}",
                    "suggestion": "Consider using class attributes or function parameters instead",
                })

            # Too many arguments (> 7)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arg_count = len(node.args.args)
                if arg_count > 7:
                    issues.append({
                        "line": node.lineno,
                        "severity": "warning",
                        "code": "W004",
                        "message": f"{node.name}() has {arg_count} arguments",
                        "suggestion": "Consider using a configuration object or dataclass",
                    })

            # Very long function (> 50 lines)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if hasattr(node, "end_lineno") and node.end_lineno:
                    length = node.end_lineno - node.lineno + 1
                    if length > 50:
                        issues.append({
                            "line": node.lineno,
                            "severity": "info",
                            "code": "I002",
                            "message": f"{node.name}() is {length} lines long",
                            "suggestion": "Consider breaking into smaller functions",
                        })

            # Nested functions (depth > 1)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    if child is not node and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        for grandchild in ast.walk(child):
                            if grandchild is not child and isinstance(
                                grandchild, (ast.FunctionDef, ast.AsyncFunctionDef)
                            ):
                                issues.append({
                                    "line": grandchild.lineno,
                                    "severity": "warning",
                                    "code": "W005",
                                    "message": f"Deeply nested function: {grandchild.name}()",
                                    "suggestion": "Extract to module-level function",
                                })

            # TODO/FIXME/HACK comments
        lines = code.splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                upper = stripped.upper()
                if "TODO" in upper:
                    issues.append({
                        "line": i,
                        "severity": "info",
                        "code": "I003",
                        "message": f"TODO comment: {stripped[:80]}",
                        "suggestion": "Track in issue tracker",
                    })
                elif "FIXME" in upper or "HACK" in upper:
                    issues.append({
                        "line": i,
                        "severity": "warning",
                        "code": "W006",
                        "message": f"FIXME/HACK comment: {stripped[:80]}",
                        "suggestion": "Address the technical debt",
                    })

        # Sort by line number
        issues.sort(key=lambda x: x["line"])
        return issues

    # ──────────────────────────────────────────────
    # Security Scanning
    # ──────────────────────────────────────────────

    def _find_security_issues(self, code: str, tree: ast.AST) -> List[Dict]:
        """Find security-related issues."""
        issues = []

        # Check for dangerous function calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = self._get_call_name(node)
                if call_name and call_name in DANGEROUS_CALLS:
                    issues.append({
                        "line": node.lineno,
                        "severity": "security",
                        "code": "S001",
                        "message": DANGEROUS_CALLS[call_name],
                        "call": call_name,
                    })

            # Check for shell=True in subprocess calls
            if isinstance(node, ast.Call):
                call_name = self._get_call_name(node)
                if call_name and "subprocess" in call_name:
                    for keyword in node.keywords:
                        if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant):
                            if keyword.value.value is True:
                                issues.append({
                                    "line": node.lineno,
                                    "severity": "security",
                                    "code": "S002",
                                    "message": "subprocess with shell=True - command injection risk",
                                    "call": call_name,
                                })

        # Check for hardcoded secrets in source
        for pattern, desc in SECRET_PATTERNS:
            for match in re.finditer(pattern, code):
                line_num = code[:match.start()].count("\n") + 1
                issues.append({
                    "line": line_num,
                    "severity": "security",
                    "code": "S003",
                    "message": desc,
                    "match": match.group()[:40] + "..." if len(match.group()) > 40 else match.group(),
                })

        # Check for assert used for validation (removed in optimized mode)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                # Check if it's in a function (not test)
                issues.append({
                    "line": node.lineno,
                    "severity": "info",
                    "code": "S004",
                    "message": "Assert statement - removed with python -O flag",
                    "suggestion": "Use if/raise for runtime validation",
                })

        issues.sort(key=lambda x: x["line"])
        return issues

    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """Extract the function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            obj = node.func
            while isinstance(obj, ast.Attribute):
                parts.append(obj.attr)
                obj = obj.value
            if isinstance(obj, ast.Name):
                parts.append(obj.id)
            parts.reverse()
            return ".".join(parts)
        return None

    # ──────────────────────────────────────────────
    # Dependency Analysis
    # ──────────────────────────────────────────────

    def _analyze_dependencies(self, tree: ast.AST) -> Dict:
        """Analyze imports and dependencies."""
        stdlib_modules = {
            "abc", "ast", "asyncio", "base64", "collections", "contextlib",
            "copy", "csv", "dataclasses", "datetime", "decimal", "enum",
            "functools", "glob", "hashlib", "hmac", "html", "http",
            "importlib", "inspect", "io", "itertools", "json", "logging",
            "math", "multiprocessing", "operator", "os", "pathlib",
            "pickle", "platform", "pprint", "queue", "random", "re",
            "secrets", "shutil", "signal", "socket", "sqlite3", "string",
            "struct", "subprocess", "sys", "tempfile", "textwrap",
            "threading", "time", "traceback", "typing", "unittest",
            "urllib", "uuid", "warnings", "weakref", "xml", "zipfile",
        }

        imports = []
        from_imports = []
        stdlib = []
        third_party = []
        local = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    entry = {
                        "module": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno,
                    }
                    imports.append(entry)
                    root = alias.name.split(".")[0]
                    if root in stdlib_modules:
                        stdlib.append(alias.name)
                    elif alias.name.startswith("."):
                        local.append(alias.name)
                    else:
                        third_party.append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    entry = {
                        "module": module,
                        "name": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno,
                        "level": node.level,
                    }
                    from_imports.append(entry)

                if node.level > 0:
                    local.append(module)
                else:
                    root = module.split(".")[0] if module else ""
                    if root in stdlib_modules:
                        stdlib.append(module)
                    else:
                        third_party.append(module)

        return {
            "imports": imports,
            "from_imports": from_imports,
            "stdlib": sorted(set(stdlib)),
            "third_party": sorted(set(third_party)),
            "local": sorted(set(local)),
            "total_imports": len(imports) + len(from_imports),
        }

    # ──────────────────────────────────────────────
    # Action Handlers
    # ──────────────────────────────────────────────

    async def _analyze(self, params: Dict) -> SkillResult:
        """Full analysis of a Python file."""
        path = params.get("path", "").strip()
        if not path:
            return SkillResult(success=False, message="path parameter required")

        code, err = self._read_file(path)
        if err:
            return err

        tree, err = self._parse_ast(code, filename=path)
        if err:
            return err

        metrics = self._compute_metrics(code, tree)
        issues = self._find_issues(code, tree)
        security = self._find_security_issues(code, tree)
        deps = self._analyze_dependencies(tree)
        complexity = self._function_complexity(tree)

        # Generate quality score (0-100)
        score = self._quality_score(metrics, issues, security)

        # Summarize
        error_count = sum(1 for i in issues if i["severity"] == "error")
        warning_count = sum(1 for i in issues if i["severity"] == "warning")
        security_count = len([i for i in security if i["severity"] == "security"])

        summary = (
            f"Score: {score}/100 | "
            f"{metrics['code_lines']} LOC | "
            f"{metrics['functions']} functions, {metrics['classes']} classes | "
            f"{error_count} errors, {warning_count} warnings, {security_count} security issues"
        )

        return SkillResult(
            success=True,
            message=summary,
            data={
                "file": path,
                "quality_score": score,
                "metrics": metrics,
                "issues": issues,
                "security_issues": security,
                "dependencies": deps,
                "complexity": complexity[:10],  # Top 10 most complex functions
            },
        )

    async def _analyze_code(self, params: Dict) -> SkillResult:
        """Analyze code from a string."""
        code = params.get("code", "")
        filename = params.get("filename", "<input>")

        if not code:
            return SkillResult(success=False, message="code parameter required")

        tree, err = self._parse_ast(code, filename=filename)
        if err:
            return err

        metrics = self._compute_metrics(code, tree)
        issues = self._find_issues(code, tree)
        security = self._find_security_issues(code, tree)
        score = self._quality_score(metrics, issues, security)

        return SkillResult(
            success=True,
            message=f"Score: {score}/100 | {metrics['code_lines']} LOC | {len(issues)} issues",
            data={
                "quality_score": score,
                "metrics": metrics,
                "issues": issues,
                "security_issues": security,
            },
        )

    async def _metrics(self, params: Dict) -> SkillResult:
        """Compute code metrics."""
        path = params.get("path", "").strip()
        if not path:
            return SkillResult(success=False, message="path parameter required")

        code, err = self._read_file(path)
        if err:
            return err

        tree, err = self._parse_ast(code, filename=path)
        if err:
            return err

        metrics = self._compute_metrics(code, tree)
        return SkillResult(
            success=True,
            message=f"{metrics['code_lines']} LOC | {metrics['functions']} functions | {metrics['classes']} classes | Complexity: {metrics['total_complexity']}",
            data={"file": path, "metrics": metrics},
        )

    async def _issues(self, params: Dict) -> SkillResult:
        """Find code quality issues."""
        path = params.get("path", "").strip()
        if not path:
            return SkillResult(success=False, message="path parameter required")

        code, err = self._read_file(path)
        if err:
            return err

        tree, err = self._parse_ast(code, filename=path)
        if err:
            return err

        issues = self._find_issues(code, tree)
        error_count = sum(1 for i in issues if i["severity"] == "error")
        warning_count = sum(1 for i in issues if i["severity"] == "warning")
        info_count = sum(1 for i in issues if i["severity"] == "info")

        return SkillResult(
            success=True,
            message=f"Found {len(issues)} issues: {error_count} errors, {warning_count} warnings, {info_count} info",
            data={"file": path, "issues": issues, "counts": {"errors": error_count, "warnings": warning_count, "info": info_count}},
        )

    async def _security(self, params: Dict) -> SkillResult:
        """Security scan."""
        path = params.get("path", "").strip()
        if not path:
            return SkillResult(success=False, message="path parameter required")

        code, err = self._read_file(path)
        if err:
            return err

        tree, err = self._parse_ast(code, filename=path)
        if err:
            return err

        issues = self._find_security_issues(code, tree)
        critical = [i for i in issues if "S001" in i.get("code", "") or "S002" in i.get("code", "") or "S003" in i.get("code", "")]

        return SkillResult(
            success=True,
            message=f"Found {len(issues)} security findings ({len(critical)} critical)",
            data={"file": path, "security_issues": issues, "critical_count": len(critical)},
        )

    async def _dependencies(self, params: Dict) -> SkillResult:
        """Analyze dependencies."""
        path = params.get("path", "").strip()
        if not path:
            return SkillResult(success=False, message="path parameter required")

        code, err = self._read_file(path)
        if err:
            return err

        tree, err = self._parse_ast(code, filename=path)
        if err:
            return err

        deps = self._analyze_dependencies(tree)
        return SkillResult(
            success=True,
            message=f"{deps['total_imports']} imports: {len(deps['stdlib'])} stdlib, {len(deps['third_party'])} third-party, {len(deps['local'])} local",
            data={"file": path, "dependencies": deps},
        )

    async def _complexity(self, params: Dict) -> SkillResult:
        """Detailed complexity per function."""
        path = params.get("path", "").strip()
        threshold = params.get("threshold", 5)
        if not path:
            return SkillResult(success=False, message="path parameter required")

        code, err = self._read_file(path)
        if err:
            return err

        tree, err = self._parse_ast(code, filename=path)
        if err:
            return err

        functions = self._function_complexity(tree)
        above_threshold = [f for f in functions if f["complexity"] >= threshold]

        return SkillResult(
            success=True,
            message=f"{len(above_threshold)}/{len(functions)} functions above complexity threshold {threshold}",
            data={
                "file": path,
                "functions": functions,
                "above_threshold": above_threshold,
                "threshold": threshold,
            },
        )

    async def _scan_directory(self, params: Dict) -> SkillResult:
        """Scan all Python files in a directory."""
        path = params.get("path", "").strip()
        recursive = params.get("recursive", True)
        if not path:
            return SkillResult(success=False, message="path parameter required")

        path = os.path.expanduser(path)
        if not os.path.isdir(path):
            return SkillResult(success=False, message=f"Directory not found: {path}")

        # Find Python files
        py_files = []
        if recursive:
            for root, dirs, files in os.walk(path):
                # Skip common non-code directories
                dirs[:] = [d for d in dirs if d not in {"__pycache__", ".git", "node_modules", ".venv", "venv", ".tox", ".eggs"}]
                for f in files:
                    if f.endswith(".py"):
                        py_files.append(os.path.join(root, f))
        else:
            py_files = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.endswith(".py")
            ]

        if not py_files:
            return SkillResult(success=False, message=f"No Python files found in {path}")

        # Analyze each file
        results = []
        total_issues = 0
        total_security = 0
        total_loc = 0

        for filepath in py_files[:100]:  # Cap at 100 files
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    code = f.read()

                tree, _ = self._parse_ast(code, filename=filepath)
                if tree is None:
                    results.append({
                        "file": filepath,
                        "error": "Syntax error",
                        "quality_score": 0,
                    })
                    continue

                metrics = self._compute_metrics(code, tree)
                issues = self._find_issues(code, tree)
                security = self._find_security_issues(code, tree)
                score = self._quality_score(metrics, issues, security)

                total_issues += len(issues)
                total_security += len([s for s in security if s["severity"] == "security"])
                total_loc += metrics["code_lines"]

                results.append({
                    "file": filepath,
                    "quality_score": score,
                    "code_lines": metrics["code_lines"],
                    "issue_count": len(issues),
                    "security_count": len([s for s in security if s["severity"] == "security"]),
                    "complexity": metrics["total_complexity"],
                })
            except Exception as e:
                results.append({
                    "file": filepath,
                    "error": str(e),
                    "quality_score": 0,
                })

        # Sort by score (worst first)
        results.sort(key=lambda x: x.get("quality_score", 0))

        avg_score = (
            round(sum(r.get("quality_score", 0) for r in results) / len(results), 1)
            if results
            else 0
        )

        return SkillResult(
            success=True,
            message=f"Scanned {len(results)} files | Avg score: {avg_score}/100 | {total_loc} LOC | {total_issues} issues | {total_security} security findings",
            data={
                "directory": path,
                "file_count": len(results),
                "total_loc": total_loc,
                "total_issues": total_issues,
                "total_security_issues": total_security,
                "average_score": avg_score,
                "files": results,
            },
        )

    # ──────────────────────────────────────────────
    # Quality Scoring
    # ──────────────────────────────────────────────

    def _quality_score(self, metrics: Dict, issues: List[Dict], security: List[Dict]) -> int:
        """Calculate quality score 0-100."""
        score = 100

        # Deductions for issues
        for issue in issues:
            if issue["severity"] == "error":
                score -= 10
            elif issue["severity"] == "warning":
                score -= 3
            elif issue["severity"] == "info":
                score -= 1

        # Deductions for security
        for issue in security:
            if issue["severity"] == "security":
                score -= 8

        # Deduction for low comment ratio
        if metrics.get("comment_ratio", 0) < 0.05 and metrics.get("code_lines", 0) > 50:
            score -= 5

        # Deduction for very high complexity
        if metrics.get("total_complexity", 0) > 50:
            score -= 5
        elif metrics.get("total_complexity", 0) > 100:
            score -= 10

        # Bonus for good structure
        if metrics.get("avg_function_length", 0) > 0 and metrics.get("avg_function_length", 999) <= 20:
            score += 5

        return max(0, min(100, score))
