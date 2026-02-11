"""
Skill Doctor - diagnostic tool for the singularity skill system.

Tests each registered skill's import chain, validates registry entries,
checks for missing dependencies, circular imports, and missing files.
Produces actionable diagnostic reports.

Usage:
    python -m singularity.skills.doctor           # Run full diagnostic
    python -m singularity.skills.doctor --json     # JSON output
    python -m singularity.skills.doctor twitter    # Check specific skill
"""

import importlib
import json
import sys
import traceback
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class Severity(str, Enum):
    OK = "ok"
    WARN = "warn"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DiagnosticIssue:
    """A single diagnostic finding."""
    severity: Severity
    code: str          # e.g. "CIRCULAR_IMPORT", "MISSING_MODULE", "MISSING_FILE"
    message: str
    fix_hint: str = ""


@dataclass
class SkillDiagnostic:
    """Diagnostic result for a single skill."""
    skill_id: str
    module_path: str
    class_name: str
    category: str
    can_import: bool = False
    can_instantiate: bool = False
    action_count: int = 0
    issues: List[DiagnosticIssue] = field(default_factory=list)
    import_error: str = ""
    instantiate_error: str = ""

    @property
    def status(self) -> Severity:
        if not self.issues:
            return Severity.OK
        severities = [i.severity for i in self.issues]
        if Severity.CRITICAL in severities:
            return Severity.CRITICAL
        if Severity.ERROR in severities:
            return Severity.ERROR
        if Severity.WARN in severities:
            return Severity.WARN
        return Severity.OK


@dataclass
class DoctorReport:
    """Full diagnostic report across all skills."""
    total_skills: int = 0
    importable: int = 0
    instantiable: int = 0
    broken: int = 0
    warnings: int = 0
    skills: Dict[str, SkillDiagnostic] = field(default_factory=dict)

    def summary_line(self) -> str:
        return (
            f"{self.importable}/{self.total_skills} importable, "
            f"{self.instantiable}/{self.total_skills} instantiable, "
            f"{self.broken} broken, {self.warnings} warnings"
        )

    def to_dict(self) -> dict:
        d = {
            "total_skills": self.total_skills,
            "importable": self.importable,
            "instantiable": self.instantiable,
            "broken": self.broken,
            "warnings": self.warnings,
            "skills": {},
        }
        for sid, diag in self.skills.items():
            d["skills"][sid] = {
                "skill_id": diag.skill_id,
                "module_path": diag.module_path,
                "class_name": diag.class_name,
                "category": diag.category,
                "can_import": diag.can_import,
                "can_instantiate": diag.can_instantiate,
                "action_count": diag.action_count,
                "status": diag.status.value,
                "import_error": diag.import_error,
                "instantiate_error": diag.instantiate_error,
                "issues": [
                    {
                        "severity": i.severity.value,
                        "code": i.code,
                        "message": i.message,
                        "fix_hint": i.fix_hint,
                    }
                    for i in diag.issues
                ],
            }
        return d


def _classify_import_error(error_str: str, tb_str: str, module_path: str) -> DiagnosticIssue:
    """Classify an import error into a specific diagnostic issue with fix hints."""

    # Circular import detection
    if "cannot import name" in error_str and "partially initialized module" in error_str:
        return DiagnosticIssue(
            severity=Severity.CRITICAL,
            code="CIRCULAR_IMPORT",
            message=f"Circular import detected: {error_str}",
            fix_hint=(
                "Move constants to a separate constants.py file, or use "
                "deferred imports (import inside function body) to break the cycle."
            ),
        )

    # Missing module
    if "No module named" in error_str:
        missing = error_str.split("No module named ")[-1].strip("'\"")
        # Check if it's a relative vs absolute import issue
        if missing.startswith("skills.") and not missing.startswith("singularity."):
            return DiagnosticIssue(
                severity=Severity.ERROR,
                code="WRONG_IMPORT_PATH",
                message=f"Wrong import path: '{missing}' should be 'singularity.{missing}'",
                fix_hint=f"Change `from {missing}` to `from singularity.{missing}`",
            )
        # Check if it's an optional external dependency
        known_optional = {
            "payments": "Internal payments module (missing from repo)",
            "jwt": "PyJWT - install with: pip install PyJWT",
            "httpx": "httpx - install with: pip install httpx",
            "browser_use": "browser-use - install with: pip install browser-use",
            "playwright": "playwright - install with: pip install playwright",
            "mcp": "MCP SDK - install with: pip install mcp",
        }
        for pkg, hint in known_optional.items():
            if missing == pkg or missing.startswith(f"{pkg}."):
                return DiagnosticIssue(
                    severity=Severity.WARN,
                    code="MISSING_OPTIONAL_DEP",
                    message=f"Optional dependency missing: {missing}",
                    fix_hint=f"{hint}. Guard with try/except ImportError.",
                )
        return DiagnosticIssue(
            severity=Severity.ERROR,
            code="MISSING_MODULE",
            message=f"Module not found: {missing}",
            fix_hint=f"Ensure '{missing}' exists or guard the import with try/except.",
        )

    # Attribute errors (often from wrong class names or missing exports)
    if "has no attribute" in error_str or "cannot import name" in error_str:
        return DiagnosticIssue(
            severity=Severity.ERROR,
            code="MISSING_ATTRIBUTE",
            message=error_str,
            fix_hint="Check __init__.py exports and class names in the skill module.",
        )

    # Generic fallback
    return DiagnosticIssue(
        severity=Severity.ERROR,
        code="IMPORT_ERROR",
        message=error_str,
        fix_hint="Check the full traceback for details.",
    )


def _check_source_files(module_path: str) -> List[DiagnosticIssue]:
    """Check if the skill's source files actually exist on disk."""
    issues = []
    parts = module_path.split(".")
    # Resolve module path to filesystem
    # singularity.skills.builtin.twitter -> singularity/skills/builtin/twitter/
    base = Path(__file__).parent.parent.parent  # repo root
    module_dir = base / Path(*parts)
    module_file = base / Path(*parts[:-1]) / f"{parts[-1]}.py"

    if module_dir.is_dir():
        init_file = module_dir / "__init__.py"
        if not init_file.exists():
            # Check if there's a __pycache__ (files were deleted but bytecode remains)
            pycache = module_dir / "__pycache__"
            if pycache.exists() and list(pycache.glob("*.pyc")):
                issues.append(DiagnosticIssue(
                    severity=Severity.CRITICAL,
                    code="SOURCE_DELETED",
                    message=f"Source files deleted but __pycache__ exists at {module_dir}",
                    fix_hint=(
                        "Source .py files were deleted. Reconstruct from __pycache__ bytecode "
                        "using: python3 -c 'import dis,marshal; "
                        "f=open(\"file.pyc\",\"rb\"); f.read(16); "
                        "code=marshal.load(f); dis.dis(code)'"
                    ),
                ))
            else:
                issues.append(DiagnosticIssue(
                    severity=Severity.ERROR,
                    code="MISSING_INIT",
                    message=f"Missing __init__.py in {module_dir}",
                    fix_hint=f"Create {init_file} with appropriate exports.",
                ))
        # Check for empty packages (only __init__.py, no real code)
        py_files = list(module_dir.glob("*.py"))
        if len(py_files) == 1 and py_files[0].name == "__init__.py":
            init_size = py_files[0].stat().st_size
            if init_size < 50:  # Nearly empty init
                issues.append(DiagnosticIssue(
                    severity=Severity.WARN,
                    code="EMPTY_PACKAGE",
                    message=f"Package {module_dir} has only a minimal __init__.py ({init_size} bytes)",
                    fix_hint="Skill implementation may be incomplete.",
                ))
    elif not module_file.exists():
        issues.append(DiagnosticIssue(
            severity=Severity.ERROR,
            code="MISSING_SOURCE",
            message=f"Neither {module_dir}/ nor {module_file} exists",
            fix_hint=f"Create the skill module at {module_dir}/__init__.py",
        ))

    return issues


def diagnose_skill(skill_id: str, module_path: str, class_name: str, category: str = "") -> SkillDiagnostic:
    """Run full diagnostics on a single skill."""
    diag = SkillDiagnostic(
        skill_id=skill_id,
        module_path=module_path,
        class_name=class_name,
        category=category,
    )

    # Check source files first
    diag.issues.extend(_check_source_files(module_path))

    # Try importing the module
    try:
        module = importlib.import_module(module_path)
        diag.can_import = True
    except Exception as e:
        diag.can_import = False
        diag.import_error = str(e)
        tb_str = traceback.format_exc()
        diag.issues.append(_classify_import_error(str(e), tb_str, module_path))
        return diag

    # Check that the class exists
    if not hasattr(module, class_name):
        diag.issues.append(DiagnosticIssue(
            severity=Severity.ERROR,
            code="MISSING_CLASS",
            message=f"Module '{module_path}' has no class '{class_name}'",
            fix_hint=f"Ensure '{class_name}' is defined and exported in {module_path}.__init__",
        ))
        return diag

    # Try instantiating
    try:
        cls = getattr(module, class_name)
        instance = cls(credentials={})
        diag.can_instantiate = True
        # Check manifest
        if hasattr(instance, 'manifest') and hasattr(instance.manifest, 'actions'):
            diag.action_count = len(instance.manifest.actions)
            if diag.action_count == 0:
                diag.issues.append(DiagnosticIssue(
                    severity=Severity.WARN,
                    code="NO_ACTIONS",
                    message=f"Skill '{skill_id}' has 0 registered actions",
                    fix_hint="Add actions to the skill manifest.",
                ))
    except Exception as e:
        diag.can_instantiate = False
        diag.instantiate_error = str(e)
        diag.issues.append(DiagnosticIssue(
            severity=Severity.WARN,
            code="INSTANTIATE_FAIL",
            message=f"Cannot instantiate with empty credentials: {e}",
            fix_hint="This may be expected if credentials are required.",
        ))

    return diag


def run_doctor(skill_ids: Optional[List[str]] = None) -> DoctorReport:
    """Run diagnostics on all registered skills (or a subset)."""
    registry_path = Path(__file__).parent / "registry.json"
    if not registry_path.exists():
        print(f"ERROR: Registry not found at {registry_path}")
        return DoctorReport()

    with open(registry_path) as f:
        data = json.load(f)

    skills_data = data.get("skills", {})
    report = DoctorReport()

    for skill_id, sd in sorted(skills_data.items()):
        if skill_ids and skill_id not in skill_ids:
            continue

        module_path = sd.get("module", "")
        class_name = sd.get("class", "")
        category = sd.get("manifest", {}).get("category", "")

        diag = diagnose_skill(skill_id, module_path, class_name, category)
        report.skills[skill_id] = diag
        report.total_skills += 1
        if diag.can_import:
            report.importable += 1
        if diag.can_instantiate:
            report.instantiable += 1
        if diag.status in (Severity.ERROR, Severity.CRITICAL):
            report.broken += 1
        elif diag.status == Severity.WARN:
            report.warnings += 1

    return report


def print_report(report: DoctorReport, use_json: bool = False) -> None:
    """Print a human-readable or JSON diagnostic report."""
    if use_json:
        print(json.dumps(report.to_dict(), indent=2))
        return

    STATUS_ICONS = {
        Severity.OK: "✅",
        Severity.WARN: "⚠️ ",
        Severity.ERROR: "❌",
        Severity.CRITICAL: "🔴",
    }

    print("=" * 60)
    print("  Singularity Skill Doctor")
    print("=" * 60)
    print(f"\n  Summary: {report.summary_line()}\n")

    # Group by status
    for status in [Severity.CRITICAL, Severity.ERROR, Severity.WARN, Severity.OK]:
        matching = [
            (sid, d) for sid, d in report.skills.items() if d.status == status
        ]
        if not matching:
            continue
        print(f"  {STATUS_ICONS[status]} {status.value.upper()} ({len(matching)}):")
        for sid, d in matching:
            label = f"    {sid:<25} [{d.category}]"
            if d.can_import and d.can_instantiate:
                label += f"  ({d.action_count} actions)"
            elif d.can_import:
                label += "  (import ok, instantiate failed)"
            else:
                label += f"  IMPORT FAIL"
            print(label)
            for issue in d.issues:
                print(f"      {STATUS_ICONS[issue.severity]} [{issue.code}] {issue.message}")
                if issue.fix_hint:
                    print(f"         → {issue.fix_hint}")
        print()

    print("=" * 60)
    print(f"  {report.importable}/{report.total_skills} importable | "
          f"{report.instantiable}/{report.total_skills} instantiable | "
          f"{report.broken} broken")
    print("=" * 60)


def main():
    """CLI entry point."""
    args = sys.argv[1:]
    use_json = "--json" in args
    skill_ids = [a for a in args if not a.startswith("--")]

    report = run_doctor(skill_ids or None)
    print_report(report, use_json=use_json)

    # Exit code: 0 if all ok, 1 if any broken
    sys.exit(1 if report.broken > 0 else 0)


if __name__ == "__main__":
    main()
