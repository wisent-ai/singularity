#!/usr/bin/env python3
"""
SelfTestingSkill - Enables the agent to run, analyze, and learn from its own tests.

Serves the Self-Improvement pillar by closing the test-driven development loop:

1. The agent modifies its own code (SelfModifySkill)
2. The agent runs its own tests (SelfTestingSkill) ← THIS
3. The agent detects failures and diagnoses root causes
4. The agent feeds results into FeedbackLoop for behavioral adaptation

Without this skill, self-modification is flying blind — the agent has no way
to verify its changes don't break existing functionality.

Actions:
  - run_tests: Execute pytest on the codebase, parse results into structured data
  - run_file: Run tests in a specific file
  - diagnose: Analyze a test failure and suggest probable causes
  - health: Show test suite health over time (pass rate, flaky tests, regressions)
  - discover: Find all test files and count test cases
  - regression_check: Compare current results against last known good run

No external dependencies required beyond pytest (already in dev deps).
"""

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from .base import Skill, SkillManifest, SkillAction, SkillResult


HISTORY_FILE = Path(__file__).parent.parent / "data" / "test_history.json"
MAX_HISTORY = 200
MAX_OUTPUT_LINES = 300  # Cap captured output to avoid memory issues


class SelfTestingSkill(Skill):
    """
    Enables the agent to run its own test suite, parse failures,
    diagnose issues, and track test health over time.

    Integrates with FeedbackLoop: test failures generate performance
    signals that drive behavioral adaptation.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._project_root: Optional[Path] = None
        self._ensure_data()

    def _ensure_data(self):
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not HISTORY_FILE.exists():
            self._save_history([])

    def _load_history(self) -> List[Dict]:
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_history(self, history: List[Dict]):
        with open(HISTORY_FILE, "w") as f:
            json.dump(history[-MAX_HISTORY:], f, indent=2)

    def _get_project_root(self) -> Path:
        """Get the project root directory."""
        if self._project_root:
            return self._project_root
        # Default: the singularity project root (parent of singularity package)
        return Path(__file__).parent.parent.parent

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="self_testing",
            name="Self Testing",
            version="1.0.0",
            category="dev",
            description=(
                "Run the agent's own test suite, parse failures into structured data, "
                "diagnose root causes, and track test health over time. Critical for "
                "safe self-modification — verifies changes don't break functionality."
            ),
            actions=[
                SkillAction(
                    name="run_tests",
                    description="Run the full test suite via pytest and return structured results",
                    parameters={
                        "project_root": {
                            "type": "string",
                            "required": False,
                            "description": "Project root path (defaults to singularity repo root)",
                        },
                        "markers": {
                            "type": "string",
                            "required": False,
                            "description": "Pytest marker expression (e.g. 'not slow')",
                        },
                        "timeout": {
                            "type": "integer",
                            "required": False,
                            "description": "Timeout in seconds (default 120)",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="run_file",
                    description="Run tests in a specific file",
                    parameters={
                        "file_path": {
                            "type": "string",
                            "required": True,
                            "description": "Path to the test file to run",
                        },
                        "test_name": {
                            "type": "string",
                            "required": False,
                            "description": "Specific test function name to run",
                        },
                        "timeout": {
                            "type": "integer",
                            "required": False,
                            "description": "Timeout in seconds (default 60)",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="diagnose",
                    description="Analyze a test failure and suggest probable root causes",
                    parameters={
                        "failure": {
                            "type": "object",
                            "required": True,
                            "description": "Failure dict from run_tests results (or raw error text)",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="health",
                    description="Show test suite health: pass rates, flaky tests, regressions",
                    parameters={
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of recent runs to analyze (default 20)",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="discover",
                    description="Find all test files and count test cases in the project",
                    parameters={
                        "project_root": {
                            "type": "string",
                            "required": False,
                            "description": "Project root path",
                        },
                    },
                    estimated_cost=0.0,
                ),
                SkillAction(
                    name="regression_check",
                    description="Compare current test results against last known good run",
                    parameters={
                        "project_root": {
                            "type": "string",
                            "required": False,
                            "description": "Project root path",
                        },
                        "timeout": {
                            "type": "integer",
                            "required": False,
                            "description": "Timeout in seconds (default 120)",
                        },
                    },
                    estimated_cost=0.0,
                ),
            ],
            required_credentials=[],
        )

    async def initialize(self) -> bool:
        self.initialized = True
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        actions = {
            "run_tests": self._run_tests,
            "run_file": self._run_file,
            "diagnose": self._diagnose,
            "health": self._health,
            "discover": self._discover,
            "regression_check": self._regression_check,
        }
        if action not in actions:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )
        return await actions[action](params)

    # --- Core Test Execution ---

    def _execute_pytest(
        self,
        args: List[str],
        cwd: str,
        timeout: int = 120,
    ) -> Dict[str, Any]:
        """
        Run pytest with given args and return structured results.

        Returns dict with:
          - exit_code: pytest exit code
          - passed, failed, errors, skipped: counts
          - failures: list of failure details
          - duration_seconds: wall-clock time
          - raw_output: truncated stdout+stderr
        """
        cmd = ["python", "-m", "pytest", "-v", "--tb=short", "--no-header"] + args

        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = time.time() - start
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
        except subprocess.TimeoutExpired:
            duration = time.time() - start
            return {
                "exit_code": -1,
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "skipped": 0,
                "failures": [],
                "duration_seconds": round(duration, 2),
                "raw_output": f"TIMEOUT after {timeout}s",
                "timed_out": True,
            }
        except FileNotFoundError:
            return {
                "exit_code": -2,
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "skipped": 0,
                "failures": [],
                "duration_seconds": 0,
                "raw_output": "pytest not found. Install with: pip install pytest",
                "timed_out": False,
            }

        # Parse the output
        output = stdout + "\n" + stderr
        # Truncate for storage
        output_lines = output.split("\n")
        if len(output_lines) > MAX_OUTPUT_LINES:
            output_lines = output_lines[:MAX_OUTPUT_LINES] + [
                f"... truncated ({len(output_lines) - MAX_OUTPUT_LINES} more lines)"
            ]
        truncated_output = "\n".join(output_lines)

        # Parse summary line: "X passed, Y failed, Z error, W skipped"
        counts = self._parse_summary(output)
        failures = self._parse_failures(output)

        return {
            "exit_code": proc.returncode,
            "passed": counts.get("passed", 0),
            "failed": counts.get("failed", 0),
            "errors": counts.get("errors", 0),
            "skipped": counts.get("skipped", 0),
            "warnings": counts.get("warnings", 0),
            "failures": failures,
            "duration_seconds": round(duration, 2),
            "raw_output": truncated_output,
            "timed_out": False,
        }

    def _parse_summary(self, output: str) -> Dict[str, int]:
        """Parse pytest summary line for counts."""
        counts = {}
        # Match patterns like "5 passed", "2 failed", "1 error", "3 skipped"
        for match in re.finditer(r"(\d+) (passed|failed|error|errors|skipped|warnings|warning)", output):
            num = int(match.group(1))
            key = match.group(2)
            # Normalize keys
            if key in ("error", "errors"):
                key = "errors"
            elif key in ("warning", "warnings"):
                key = "warnings"
            counts[key] = counts.get(key, 0) + num
        return counts

    def _parse_failures(self, output: str) -> List[Dict]:
        """Parse pytest output for failure details."""
        failures = []

        # Match FAILED test lines: "FAILED tests/test_foo.py::test_bar - AssertionError: ..."
        for match in re.finditer(
            r"FAILED\s+([\w/\\._-]+)::(\w+)(?:\s*-\s*(.+))?",
            output,
        ):
            file_path = match.group(1)
            test_name = match.group(2)
            error_msg = match.group(3) or ""
            failures.append({
                "file": file_path,
                "test": test_name,
                "error": error_msg.strip(),
                "type": "failure",
            })

        # Match ERROR lines: "ERROR tests/test_foo.py::test_bar"
        for match in re.finditer(
            r"ERROR\s+([\w/\\._-]+)::(\w+)(?:\s*-\s*(.+))?",
            output,
        ):
            file_path = match.group(1)
            test_name = match.group(2)
            error_msg = match.group(3) or ""
            failures.append({
                "file": file_path,
                "test": test_name,
                "error": error_msg.strip(),
                "type": "error",
            })

        # Extract traceback snippets for each failure
        # Look for short traceback sections
        tb_sections = re.split(r"_{5,}\s+", output)
        for section in tb_sections:
            # Match test identifier in traceback header
            header_match = re.search(r"([\w/\\._-]+)::(\w+)", section)
            if not header_match:
                continue
            test_file = header_match.group(1)
            test_name = header_match.group(2)

            # Find the error line (usually after "E   ")
            error_lines = []
            for line in section.split("\n"):
                stripped = line.strip()
                if stripped.startswith("E "):
                    error_lines.append(stripped[2:].strip())

            if error_lines:
                # Update matching failure with traceback details
                for f in failures:
                    if f["test"] == test_name and f["file"] == test_file:
                        f["traceback_errors"] = error_lines[:5]
                        break

        return failures

    # --- Actions ---

    async def _run_tests(self, params: Dict) -> SkillResult:
        """Run the full test suite."""
        project_root = params.get("project_root") or str(self._get_project_root())
        markers = params.get("markers", "")
        timeout = params.get("timeout", 120)

        args = []
        if markers:
            args.extend(["-m", markers])

        results = self._execute_pytest(args, cwd=project_root, timeout=timeout)

        # Record in history
        total = results["passed"] + results["failed"] + results["errors"]
        run_record = {
            "timestamp": datetime.now().isoformat(),
            "type": "full_suite",
            "total_tests": total,
            "passed": results["passed"],
            "failed": results["failed"],
            "errors": results["errors"],
            "skipped": results["skipped"],
            "duration_seconds": results["duration_seconds"],
            "pass_rate": round(results["passed"] / max(total, 1) * 100, 1),
            "failed_tests": [f["test"] for f in results["failures"]],
            "timed_out": results.get("timed_out", False),
        }
        history = self._load_history()
        history.append(run_record)
        self._save_history(history)

        # Build summary
        if results.get("timed_out"):
            status = "TIMEOUT"
        elif results["failed"] + results["errors"] == 0:
            status = "ALL PASS"
        else:
            status = "FAILURES"

        return SkillResult(
            success=True,
            message=(
                f"Test run [{status}]: {results['passed']} passed, "
                f"{results['failed']} failed, {results['errors']} errors, "
                f"{results['skipped']} skipped in {results['duration_seconds']}s. "
                f"Pass rate: {run_record['pass_rate']}%"
            ),
            data={
                "summary": run_record,
                "failures": results["failures"],
                "raw_output": results["raw_output"],
            },
        )

    async def _run_file(self, params: Dict) -> SkillResult:
        """Run tests in a specific file."""
        file_path = params.get("file_path", "")
        test_name = params.get("test_name", "")
        timeout = params.get("timeout", 60)

        if not file_path:
            return SkillResult(success=False, message="file_path is required")

        # Resolve relative to project root
        p = Path(file_path)
        if not p.is_absolute():
            p = self._get_project_root() / p

        if not p.exists():
            return SkillResult(success=False, message=f"Test file not found: {p}")

        args = [str(p)]
        if test_name:
            args = [f"{p}::{test_name}"]

        # Use the file's project root
        cwd = str(self._get_project_root())
        results = self._execute_pytest(args, cwd=cwd, timeout=timeout)

        total = results["passed"] + results["failed"] + results["errors"]
        status = "PASS" if results["failed"] + results["errors"] == 0 else "FAIL"

        # Record
        history = self._load_history()
        history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "single_file",
            "file": str(file_path),
            "total_tests": total,
            "passed": results["passed"],
            "failed": results["failed"],
            "errors": results["errors"],
            "duration_seconds": results["duration_seconds"],
            "failed_tests": [f["test"] for f in results["failures"]],
        })
        self._save_history(history)

        return SkillResult(
            success=True,
            message=(
                f"[{status}] {file_path}: {results['passed']} passed, "
                f"{results['failed']} failed, {results['errors']} errors "
                f"in {results['duration_seconds']}s"
            ),
            data={
                "file": str(file_path),
                "exit_code": results["exit_code"],
                "passed": results["passed"],
                "failed": results["failed"],
                "errors": results["errors"],
                "failures": results["failures"],
                "raw_output": results["raw_output"],
            },
        )

    async def _diagnose(self, params: Dict) -> SkillResult:
        """Analyze a test failure and suggest probable root causes."""
        failure = params.get("failure", {})

        if isinstance(failure, str):
            # Raw error text
            failure = {"error": failure, "test": "unknown", "file": "unknown"}

        error_msg = failure.get("error", "")
        traceback_errors = failure.get("traceback_errors", [])
        test_name = failure.get("test", "unknown")
        test_file = failure.get("file", "unknown")
        full_error = error_msg + " " + " ".join(traceback_errors)

        diagnosis = {
            "test": test_name,
            "file": test_file,
            "category": "unknown",
            "probable_causes": [],
            "suggested_fixes": [],
            "severity": "medium",
        }

        # Pattern-based diagnosis
        patterns = [
            (r"ImportError|ModuleNotFoundError", "import_error", "high", [
                "Missing dependency - check requirements/pyproject.toml",
                "Circular import - check import order",
                "Module was renamed or moved",
            ], [
                "pip install <missing_module>",
                "Check if module path changed in recent commits",
                "Verify __init__.py exports",
            ]),
            (r"AssertionError|assert.*==|assert.*!=", "assertion_failure", "medium", [
                "Expected value changed due to code modification",
                "Test data out of date",
                "Logic error in implementation",
            ], [
                "Compare expected vs actual values in assertion",
                "Check if the function contract changed",
                "Update test expectations if behavior intentionally changed",
            ]),
            (r"AttributeError", "attribute_error", "high", [
                "Object API changed - attribute was renamed or removed",
                "Wrong type passed to function",
                "Missing initialization in __init__",
            ], [
                "Check if class/object interface changed",
                "Verify the attribute name spelling",
                "Ensure proper type is being passed",
            ]),
            (r"TypeError.*argument|TypeError.*positional", "type_error_args", "high", [
                "Function signature changed (args added/removed)",
                "Wrong number of arguments in test call",
                "Method became classmethod/staticmethod or vice versa",
            ], [
                "Compare test call signature with current function signature",
                "Update test to match new function signature",
            ]),
            (r"TypeError", "type_error", "medium", [
                "Incompatible types in operation",
                "None returned where value expected",
                "Wrong data type passed to function",
            ], [
                "Add type checking or None guards",
                "Verify return types of called functions",
            ]),
            (r"KeyError", "key_error", "medium", [
                "Dictionary key was renamed or removed",
                "Expected data structure changed",
                "Missing key in configuration/params",
            ], [
                "Use .get() with default instead of direct access",
                "Check if data format changed",
            ]),
            (r"FileNotFoundError|No such file", "file_not_found", "medium", [
                "Test fixture file missing",
                "File path changed",
                "Temp directory not created",
            ], [
                "Check fixture setup creates required files",
                "Verify file paths are relative to test directory",
            ]),
            (r"TimeoutError|timed?\s*out", "timeout", "high", [
                "Test takes too long - performance regression",
                "Infinite loop in implementation",
                "Network call in test without mock",
            ], [
                "Add timeout to slow operations",
                "Mock external calls",
                "Check for infinite loops",
            ]),
            (r"ConnectionError|ConnectionRefused", "connection_error", "low", [
                "Test requires external service that's not running",
                "Missing mock for network call",
            ], [
                "Mock the network call",
                "Mark test as integration test with appropriate marker",
            ]),
            (r"PermissionError", "permission_error", "medium", [
                "File/directory permission issue",
                "Test tries to write to read-only location",
            ], [
                "Use tmp_path fixture for test files",
                "Check directory permissions",
            ]),
            (r"SyntaxError", "syntax_error", "critical", [
                "Malformed code in recent edit",
                "Incompatible Python version syntax",
            ], [
                "Check the file for syntax errors",
                "Revert recent changes and re-apply carefully",
            ]),
            (r"fixture.*not found|fixture.*error", "fixture_error", "high", [
                "Test fixture was removed or renamed",
                "Missing conftest.py",
                "Fixture scope issue",
            ], [
                "Check conftest.py for fixture definitions",
                "Verify fixture names match test parameters",
            ]),
        ]

        matched = False
        for pattern, category, severity, causes, fixes in patterns:
            if re.search(pattern, full_error, re.IGNORECASE):
                diagnosis["category"] = category
                diagnosis["severity"] = severity
                diagnosis["probable_causes"] = causes
                diagnosis["suggested_fixes"] = fixes
                matched = True
                break

        if not matched:
            diagnosis["category"] = "unknown"
            diagnosis["probable_causes"] = [
                "Error doesn't match known patterns",
                "May be a logic error in implementation",
                "Check the full traceback for details",
            ]
            diagnosis["suggested_fixes"] = [
                "Read the full error traceback",
                "Run the failing test in isolation with -v flag",
                "Check recent code changes that may have caused the failure",
            ]

        # Check flakiness from history
        history = self._load_history()
        test_appearances = []
        for run in history:
            failed_tests = run.get("failed_tests", [])
            if test_name in failed_tests:
                test_appearances.append({"run": run["timestamp"], "status": "failed"})
            elif run.get("total_tests", 0) > 0:
                test_appearances.append({"run": run["timestamp"], "status": "passed"})

        if test_appearances:
            fail_count = sum(1 for a in test_appearances if a["status"] == "failed")
            total_count = len(test_appearances)
            if 0 < fail_count < total_count:
                diagnosis["flaky"] = True
                diagnosis["flaky_rate"] = round(fail_count / total_count * 100, 1)
                diagnosis["probable_causes"].insert(0, f"FLAKY TEST: fails {diagnosis['flaky_rate']}% of the time")

        return SkillResult(
            success=True,
            message=(
                f"Diagnosis for {test_name}: [{diagnosis['category']}] "
                f"Severity: {diagnosis['severity']}. "
                f"Top cause: {diagnosis['probable_causes'][0] if diagnosis['probable_causes'] else 'unknown'}"
            ),
            data=diagnosis,
        )

    async def _health(self, params: Dict) -> SkillResult:
        """Show test suite health over time."""
        limit = params.get("limit", 20)
        history = self._load_history()

        if not history:
            return SkillResult(
                success=True,
                message="No test runs recorded yet. Run 'run_tests' first.",
                data={"runs": [], "health": {}},
            )

        # Only look at full suite runs for health
        suite_runs = [r for r in history if r.get("type") == "full_suite"]
        recent = suite_runs[-limit:] if suite_runs else []

        if not recent:
            return SkillResult(
                success=True,
                message="No full suite runs recorded. Run 'run_tests' first.",
                data={"runs": history[-limit:], "health": {}},
            )

        # Calculate health metrics
        pass_rates = [r.get("pass_rate", 0) for r in recent]
        avg_pass_rate = round(sum(pass_rates) / len(pass_rates), 1)
        current_pass_rate = pass_rates[-1] if pass_rates else 0
        durations = [r.get("duration_seconds", 0) for r in recent]
        avg_duration = round(sum(durations) / len(durations), 1)

        # Trend: compare last 3 vs previous 3
        trend = "stable"
        if len(pass_rates) >= 6:
            recent_avg = sum(pass_rates[-3:]) / 3
            prev_avg = sum(pass_rates[-6:-3]) / 3
            if recent_avg > prev_avg + 2:
                trend = "improving"
            elif recent_avg < prev_avg - 2:
                trend = "declining"

        # Find flaky tests: tests that sometimes pass, sometimes fail
        test_results: Dict[str, List[bool]] = {}
        for run in recent:
            failed_set = set(run.get("failed_tests", []))
            # We don't know which tests passed individually, but we know which failed
            for t in failed_set:
                if t not in test_results:
                    test_results[t] = []
                test_results[t].append(False)

        flaky_tests = []
        always_failing = []
        for test_name, results in test_results.items():
            fail_count = len(results)
            if fail_count < len(recent):
                # Failed in some runs but not all = flaky
                flaky_tests.append({
                    "test": test_name,
                    "fail_rate": round(fail_count / len(recent) * 100, 1),
                })
            else:
                always_failing.append(test_name)

        # Regressions: tests that passed in older runs but fail now
        regressions = []
        if len(recent) >= 2:
            last_failures = set(recent[-1].get("failed_tests", []))
            prev_failures = set(recent[-2].get("failed_tests", []))
            new_failures = last_failures - prev_failures
            if new_failures:
                regressions = list(new_failures)

        health = {
            "total_runs": len(recent),
            "avg_pass_rate": avg_pass_rate,
            "current_pass_rate": current_pass_rate,
            "trend": trend,
            "avg_duration_seconds": avg_duration,
            "flaky_tests": flaky_tests,
            "always_failing": always_failing,
            "regressions": regressions,
            "last_run": recent[-1] if recent else None,
        }

        status = "HEALTHY" if current_pass_rate >= 95 else "DEGRADED" if current_pass_rate >= 70 else "BROKEN"

        return SkillResult(
            success=True,
            message=(
                f"Test health [{status}]: {current_pass_rate}% pass rate "
                f"(avg {avg_pass_rate}%), trend: {trend}. "
                f"{len(flaky_tests)} flaky, {len(always_failing)} always failing, "
                f"{len(regressions)} new regressions. Avg duration: {avg_duration}s"
            ),
            data={"health": health, "recent_runs": recent[-5:]},
        )

    async def _discover(self, params: Dict) -> SkillResult:
        """Find all test files and count test cases."""
        project_root = params.get("project_root") or str(self._get_project_root())
        root = Path(project_root)

        if not root.exists():
            return SkillResult(success=False, message=f"Project root not found: {root}")

        # Find test files
        test_files = []
        patterns = ["test_*.py", "*_test.py"]
        for pattern in patterns:
            test_files.extend(root.rglob(pattern))

        # Deduplicate and filter
        seen = set()
        unique_files = []
        for f in sorted(test_files):
            # Skip __pycache__, .git, etc.
            parts = f.parts
            if any(p.startswith(".") or p == "__pycache__" for p in parts):
                continue
            rel = str(f.relative_to(root))
            if rel not in seen:
                seen.add(rel)
                unique_files.append(f)

        # Count test functions in each file
        file_details = []
        total_tests = 0
        for f in unique_files:
            try:
                content = f.read_text(errors="ignore")
                # Count test functions: def test_ or async def test_
                test_count = len(re.findall(r"(?:async\s+)?def\s+(test_\w+)", content))
                total_tests += test_count
                file_details.append({
                    "file": str(f.relative_to(root)),
                    "test_count": test_count,
                })
            except Exception:
                file_details.append({
                    "file": str(f.relative_to(root)),
                    "test_count": 0,
                    "error": "Could not read file",
                })

        return SkillResult(
            success=True,
            message=(
                f"Discovered {len(unique_files)} test files with "
                f"{total_tests} test functions in {root}"
            ),
            data={
                "project_root": str(root),
                "test_file_count": len(unique_files),
                "total_test_count": total_tests,
                "files": file_details,
            },
        )

    async def _regression_check(self, params: Dict) -> SkillResult:
        """Compare current test results against last known good run."""
        project_root = params.get("project_root") or str(self._get_project_root())
        timeout = params.get("timeout", 120)

        # Find last known good run (100% pass rate)
        history = self._load_history()
        suite_runs = [r for r in history if r.get("type") == "full_suite"]
        last_good = None
        for run in reversed(suite_runs):
            if run.get("pass_rate", 0) == 100.0:
                last_good = run
                break

        # Run current tests
        results = self._execute_pytest([], cwd=project_root, timeout=timeout)
        total = results["passed"] + results["failed"] + results["errors"]
        current_pass_rate = round(results["passed"] / max(total, 1) * 100, 1)

        # Record
        run_record = {
            "timestamp": datetime.now().isoformat(),
            "type": "full_suite",
            "total_tests": total,
            "passed": results["passed"],
            "failed": results["failed"],
            "errors": results["errors"],
            "skipped": results["skipped"],
            "duration_seconds": results["duration_seconds"],
            "pass_rate": current_pass_rate,
            "failed_tests": [f["test"] for f in results["failures"]],
        }
        history.append(run_record)
        self._save_history(history)

        # Compare
        comparison = {
            "current": run_record,
            "baseline": last_good,
            "regressions": [],
            "improvements": [],
        }

        if last_good:
            baseline_failed = set(last_good.get("failed_tests", []))
            current_failed = set(run_record["failed_tests"])

            new_failures = current_failed - baseline_failed
            fixed = baseline_failed - current_failed

            comparison["regressions"] = list(new_failures)
            comparison["improvements"] = list(fixed)

            if new_failures:
                verdict = f"REGRESSION: {len(new_failures)} new failures"
            elif fixed:
                verdict = f"IMPROVED: {len(fixed)} tests fixed"
            elif current_pass_rate == 100.0:
                verdict = "PERFECT: All tests pass"
            else:
                verdict = f"STABLE: Same {len(current_failed)} failures as baseline"
        else:
            if current_pass_rate == 100.0:
                verdict = "PERFECT: All tests pass (first baseline)"
            else:
                verdict = f"NO BASELINE: {results['failed']} failures, no prior good run to compare"

        return SkillResult(
            success=True,
            message=(
                f"Regression check [{verdict}]: {results['passed']}/{total} passed "
                f"({current_pass_rate}%) in {results['duration_seconds']}s"
            ),
            data=comparison,
        )
