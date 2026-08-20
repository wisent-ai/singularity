#!/usr/bin/env python3
"""Behavioral tests for the fail-closed capability deployment preflight."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
PREFLIGHT = REPOSITORY / "deploy" / "capabilities" / "capability-preflight.py"
DEPLOYMENT = PREFLIGHT.parent


def run_preflight(*arguments: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PREFLIGHT), *(str(argument) for argument in arguments)],
        cwd=REPOSITORY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        },
    )


def load_preflight_module():
    spec = importlib.util.spec_from_file_location("capability_preflight_under_test", PREFLIGHT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PREFLIGHT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StaticDeploymentPreflightTests(unittest.TestCase):
    def test_checked_in_deployment_passes_static_preflight(self) -> None:
        result = run_preflight("deployment-static", DEPLOYMENT)

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual("", result.stderr)

    def test_static_preflight_rejects_removed_isolation_controls(self) -> None:
        cases = (
            (
                "dedicated workload UID",
                Path("systemd/capabilities.sysusers"),
                'u wisent-agent-brama - "Brama capability consumer"',
                '# removed wisent-agent-brama',
            ),
            (
                "owner-only runtime directory",
                Path("systemd/wisent-agent@.service"),
                "RuntimeDirectoryMode=0700",
                "RuntimeDirectoryMode=0755",
            ),
            (
                "capability socket client group",
                Path("systemd/wisent-agent@.service"),
                "SupplementaryGroups=skarbiec-capability-clients",
                "SupplementaryGroups=users",
            ),
            (
                "deny-by-default workload egress",
                Path("systemd/wisent-agent@brama.service.d/20-egress-allowlist.conf.example"),
                "IPAddressDeny=any",
                "IPAddressDeny=none",
            ),
        )
        for name, relative_path, required, weakened in cases:
            with self.subTest(control=name), tempfile.TemporaryDirectory() as temporary:
                deployment = Path(temporary) / "capabilities"
                shutil.copytree(DEPLOYMENT, deployment)
                target = deployment / relative_path
                original = target.read_text(encoding="utf-8")
                self.assertIn(required, original, f"fixture no longer contains {name}")
                target.write_text(original.replace(required, weakened, 1), encoding="utf-8")

                result = run_preflight("deployment-static", deployment)

                self.assertEqual(78, result.returncode, result.stderr)
                self.assertIn("capability-preflight:", result.stderr)

    def test_static_preflight_rejects_world_routable_egress_allowlists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            deployment = Path(temporary) / "capabilities"
            shutil.copytree(DEPLOYMENT, deployment)
            allowlist = deployment / "systemd/wisent-agent@weles.service.d/20-egress-allowlist.conf.example"
            allowlist.write_text(
                allowlist.read_text(encoding="utf-8") + "IPAddressAllow=0.0.0.0/0\n",
                encoding="utf-8",
            )

            result = run_preflight("deployment-static", deployment)

            self.assertEqual(78, result.returncode, result.stderr)
            self.assertIn("broad egress route is forbidden", result.stderr)

    def test_macos_broker_refuses_launch_without_external_egress_sandbox(self) -> None:
        result = run_preflight("broker-macos")

        self.assertEqual(78, result.returncode, result.stderr)
        if sys.platform == "darwin":
            self.assertIn("externally enforced deny-all sandbox", result.stderr)
        else:
            self.assertIn("only valid on macOS", result.stderr)


class ExecEnvironmentIsolationTests(unittest.TestCase):
    def test_exec_environment_contains_only_validated_file_values_and_safe_locale(self) -> None:
        preflight = load_preflight_module()
        with tempfile.TemporaryDirectory() as temporary:
            environment_file = Path(temporary).resolve() / "agent.env"
            environment_file.write_text(
                "SINGULARITY_BOOTSTRAP_BINARY=/bin/true\n"
                "SKARBIEC_WORKLOAD_ID=brama\n",
                encoding="utf-8",
            )
            environment_file.chmod(0o600)
            observed: dict[str, str] = {}

            def capture_exec(executable: str, argv: list[str]) -> None:
                observed.update(os.environ)
                raise SystemExit((executable, argv))

            argv = [
                str(PREFLIGHT),
                "agent",
                str(environment_file),
                "--exec",
            ]
            inherited = {
                "PATH": "/attacker/bin",
                "LANG": "attacker-locale",
                "AWS_SECRET_ACCESS_KEY": "must-not-survive",
                "UNRELATED_PARENT_VALUE": "must-not-survive",
            }
            with mock.patch.object(preflight, "validate_agent"), mock.patch.object(
                preflight.os, "execv", side_effect=capture_exec
            ), mock.patch.object(preflight.sys, "argv", argv), mock.patch.dict(
                os.environ, inherited, clear=True
            ):
                with self.assertRaises(SystemExit) as exit_context:
                    preflight.main()

            self.assertEqual(("/bin/true", ["/bin/true"]), exit_context.exception.code)
            self.assertEqual(
                {
                    "SINGULARITY_BOOTSTRAP_BINARY": "/bin/true",
                    "SKARBIEC_WORKLOAD_ID": "brama",
                    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin",
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                },
                observed,
            )


if __name__ == "__main__":
    unittest.main()
