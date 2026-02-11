# Contributing to Singularity

Thank you for your interest in contributing to Singularity! This guide will help you get started.

## Getting Started

1. **Fork the repository** and clone it locally
2. **Install dependencies**: `pip install -e ".[dev]"`
3. **Run tests**: `pytest tests/`
4. **Create a branch** for your changes

## Development Setup

```bash
# Clone the repo
git clone https://github.com/wisent-ai/singularity.git
cd singularity

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Project Structure

```
singularity/
├── singularity/
│   ├── core/           # Core agent engine (cognition, memory, planning)
│   ├── skills/
│   │   ├── base.py     # Skill base classes (Skill, SkillResult, SkillManifest)
│   │   └── builtin/    # Built-in skills (shell, social media, crypto, etc.)
│   └── utils/          # Utilities and helpers
├── tests/              # Test suite
├── examples/           # Example scripts
└── docs/               # Documentation
```

## Writing a New Skill

Skills extend the `Skill` base class from `singularity.skills.base`:

```python
from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction
from typing import Dict

class MySkill(Skill):
    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="my_skill",
            display_name="My Skill",
            version="1.0.0",
            category="utility",
            description="What this skill does",
            cost_estimate=0,
            actions=[
                SkillAction(
                    name="do_something",
                    description="Description of the action",
                    parameters={"input": {"type": "string", "description": "Input value"}},
                ),
            ],
        )

    async def check_credentials(self) -> bool:
        return True  # or verify required credentials

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if action == "do_something":
            return SkillResult(success=True, data={"result": "done"}, message="Action completed")
        return SkillResult(success=False, data=None, message=f"Unknown action: {action}")
```

### Skill Guidelines

- Each skill goes in its own directory under `singularity/skills/builtin/`
- Include `__init__.py` that re-exports the main skill class
- Guard optional dependencies with `try/except ImportError`
- Return `SkillResult` from all actions (never raise uncaught exceptions)
- Include docstrings for all public methods

## Pull Request Guidelines

1. **Keep PRs focused** — one feature or fix per PR
2. **Add tests** for new functionality
3. **Follow existing code style** — the project uses standard Python conventions
4. **Write descriptive commit messages** — explain the "why", not just the "what"
5. **Don't break existing imports** — run the test suite before submitting

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include reproduction steps for bugs
- Specify the Python version and OS

## Code of Conduct

Be respectful and constructive. We're building something new here — both humans and AI agents contribute to this project.

## License

By contributing, you agree that your contributions will be licensed under the project's existing license.
