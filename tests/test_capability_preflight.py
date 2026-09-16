#!/usr/bin/env python3
"""Behavioral tests for the fail-closed capability deployment preflight."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
