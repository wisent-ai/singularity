"""
Singularity CLI — developer tools for the singularity framework.

Usage:
    singularity-cli validate [--all] [SKILL_ID ...]
    singularity-cli list [--category CATEGORY]
    singularity-cli info SKILL_ID
    singularity-cli check-registry
"""

import argparse
import json
import sys
import importlib
from pathlib import Path
from typing import List, Dict


# ─── Validation ──────────────────────────────────────────────────────────


def _load_registry() -> Dict:
    """Load the skill registry from disk."""
    registry_path = Path(__file__).parent / "skills" / "registry.json"
    if not registry_path.exists():
        print(f"ERROR: Registry not found at {registry_path}", file=sys.stderr)
        sys.exit(1)
    return json.loads(registry_path.read_text())


def _check_skill_directory(skill_id: str, module_path: str) -> List[str]:
    """Check if a skill's directory and files exist."""
    issues = []
    if not module_path.startswith("singularity.skills.builtin."):
        return issues  # External skill, can't validate

    dir_name = module_path.split(".")[-1]
    builtin_dir = Path(__file__).parent / "skills" / "builtin"
    skill_dir = builtin_dir / dir_name

    if not skill_dir.exists():
        issues.append(f"Directory missing: {skill_dir}")
        return issues

    # Check for Python files
    py_files = list(skill_dir.glob("*.py"))
    init_files = [f for f in py_files if f.name == "__init__.py"]
    impl_files = [f for f in py_files if f.name != "__init__.py"]

    if not py_files:
        issues.append(f"No Python files in {skill_dir}")
    elif not init_files and not impl_files:
        issues.append(f"No implementation files in {skill_dir}")

    return issues


def _try_import_skill(skill_id: str, module_path: str, class_name: str) -> List[str]:
    """Try to import a skill class."""
    issues = []
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        issues.append(f"Import failed: {e}")
        return issues
    except Exception as e:
        issues.append(f"Import error: {e}")
        return issues

    if not hasattr(module, class_name):
        issues.append(f"Class '{class_name}' not found in {module_path}")
        return issues

    skill_class = getattr(module, class_name)

    # Check it's a proper Skill subclass
    try:
        from singularity.skills.base.skill import Skill
        if not issubclass(skill_class, Skill):
            issues.append(f"{class_name} does not inherit from Skill")
    except Exception:
        pass

    return issues


def _validate_manifest(skill_id: str, manifest_data: Dict) -> List[str]:
    """Validate manifest data consistency."""
    issues = []

    if manifest_data.get("skill_id") != skill_id:
        issues.append(
            f"manifest.skill_id '{manifest_data.get('skill_id')}' != registry key '{skill_id}'"
        )

    required = ["skill_id", "name", "version", "category", "description"]
    for field in required:
        if not manifest_data.get(field):
            issues.append(f"Missing required manifest field: {field}")

    # Validate version format
    version = manifest_data.get("version", "")
    if version and not all(p.isdigit() for p in version.split(".")):
        issues.append(f"Invalid version format: '{version}' (expected semver)")

    return issues


def validate_skill(skill_id: str, skill_data: Dict, verbose: bool = False) -> Dict:
    """Validate a single skill. Returns dict with status and issues."""
    module_path = skill_data.get("module", "")
    class_name = skill_data.get("class", "")
    manifest = skill_data.get("manifest", {})
    issues = []

    # 1. Check manifest consistency
    issues.extend(_validate_manifest(skill_id, manifest))

    # 2. Check directory exists
    issues.extend(_check_skill_directory(skill_id, module_path))

    # 3. Try to import (only if directory exists)
    dir_issues = _check_skill_directory(skill_id, module_path)
    if not any("Directory missing" in i or "No Python files" in i for i in dir_issues):
        import_issues = _try_import_skill(skill_id, module_path, class_name)
        issues.extend(import_issues)

    status = "PASS" if not issues else "FAIL"
    return {"skill_id": skill_id, "status": status, "issues": issues}


def validate_all(registry: Dict, verbose: bool = False) -> List[Dict]:
    """Validate all skills in the registry."""
    results = []
    for skill_id, skill_data in sorted(registry.get("skills", {}).items()):
        result = validate_skill(skill_id, skill_data, verbose)
        results.append(result)
    return results


# ─── Commands ────────────────────────────────────────────────────────────


def cmd_validate(args):
    """Validate skills."""
    registry = _load_registry()
    skills = registry.get("skills", {})

    if args.all:
        results = validate_all(registry, args.verbose)
    elif args.skill_ids:
        results = []
        for sid in args.skill_ids:
            if sid not in skills:
                results.append({"skill_id": sid, "status": "FAIL",
                                "issues": ["Not found in registry"]})
            else:
                results.append(validate_skill(sid, skills[sid], args.verbose))
    else:
        print("Specify --all or one or more SKILL_IDs to validate")
        sys.exit(1)

    # Print results
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")

    for r in results:
        icon = "✓" if r["status"] == "PASS" else "✗"
        print(f"  {icon} {r['skill_id']}")
        if r["issues"] and (args.verbose or r["status"] == "FAIL"):
            for issue in r["issues"]:
                print(f"    └─ {issue}")

    print(f"\n{passed} passed, {failed} failed, {len(results)} total")
    sys.exit(1 if failed > 0 else 0)


def cmd_list(args):
    """List registered skills."""
    registry = _load_registry()
    skills = registry.get("skills", {})

    for skill_id, skill_data in sorted(skills.items()):
        manifest = skill_data.get("manifest", {})
        category = manifest.get("category", "?")

        if args.category and category != args.category:
            continue

        name = manifest.get("name", skill_id)
        creds = manifest.get("required_credentials", [])
        wiring = skill_data.get("wiring") or ""

        cred_str = f" [{', '.join(creds)}]" if creds else ""
        wire_str = f" (wiring: {wiring})" if wiring else ""
        print(f"  {skill_id:25s} {category:12s} {name}{cred_str}{wire_str}")

    print(f"\n{len(skills)} skills registered")


def cmd_info(args):
    """Show detailed info about a skill."""
    registry = _load_registry()
    skills = registry.get("skills", {})

    skill_id = args.skill_id
    if skill_id not in skills:
        print(f"Skill '{skill_id}' not found in registry", file=sys.stderr)
        sys.exit(1)

    skill_data = skills[skill_id]
    manifest = skill_data.get("manifest", {})

    print(f"Skill: {skill_id}")
    print(f"  Name: {manifest.get('name', '?')}")
    print(f"  Version: {manifest.get('version', '?')}")
    print(f"  Category: {manifest.get('category', '?')}")
    print(f"  Description: {manifest.get('description', '?')}")
    print(f"  Module: {skill_data.get('module', '?')}")
    print(f"  Class: {skill_data.get('class', '?')}")
    print(f"  Wiring: {skill_data.get('wiring') or 'none'}")
    print(f"  Author: {manifest.get('author', '?')}")

    creds = manifest.get("required_credentials", [])
    if creds:
        print(f"  Required Credentials: {', '.join(creds)}")
    else:
        print("  Required Credentials: none")

    actions = manifest.get("actions", [])
    if actions:
        print(f"  Actions ({len(actions)}):")
        for action in actions:
            if isinstance(action, dict):
                print(f"    - {action.get('name', '?')}: {action.get('description', '')}")
            else:
                print(f"    - {action}")


def cmd_check_registry(args):
    """Check registry.json for consistency issues."""
    registry = _load_registry()
    skills = registry.get("skills", {})
    issues = []

    # Check version
    if "version" not in registry:
        issues.append("Missing 'version' field in registry")

    # Check each skill
    for skill_id, skill_data in skills.items():
        # Required fields
        for field in ["module", "class", "manifest"]:
            if field not in skill_data:
                issues.append(f"[{skill_id}] Missing field: {field}")

        manifest = skill_data.get("manifest", {})

        # ID consistency
        if manifest.get("skill_id") != skill_id:
            issues.append(
                f"[{skill_id}] manifest.skill_id mismatch: '{manifest.get('skill_id')}'"
            )

        # Required manifest fields
        for field in ["name", "version", "category", "description"]:
            if not manifest.get(field):
                issues.append(f"[{skill_id}] Missing manifest.{field}")

    if issues:
        print("Registry issues found:")
        for issue in issues:
            print(f"  ✗ {issue}")
        print(f"\n{len(issues)} issues")
        sys.exit(1)
    else:
        print(f"Registry OK: {len(skills)} skills, no issues")


# ─── Main ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        prog="singularity-cli",
        description="Singularity framework developer tools",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate skills")
    p_validate.add_argument("skill_ids", nargs="*", help="Skill IDs to validate")
    p_validate.add_argument("--all", action="store_true", help="Validate all skills")
    p_validate.add_argument("-v", "--verbose", action="store_true")
    p_validate.set_defaults(func=cmd_validate)

    # list
    p_list = subparsers.add_parser("list", help="List registered skills")
    p_list.add_argument("--category", help="Filter by category")
    p_list.set_defaults(func=cmd_list)

    # info
    p_info = subparsers.add_parser("info", help="Show skill details")
    p_info.add_argument("skill_id", help="Skill ID")
    p_info.set_defaults(func=cmd_info)

    # check-registry
    p_check = subparsers.add_parser("check-registry", help="Check registry.json consistency")
    p_check.set_defaults(func=cmd_check_registry)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
