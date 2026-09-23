#!/usr/bin/env python3
"""Fail-closed deployment preflight for capability-isolated processes."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preflight import (  # noqa: E402  (the package sits beside this script)
    CONFIGURATION_EXIT_CODE,
    LAUNCH_LOCALE,
    LAUNCH_PATH,
    fail,
    load_env,
    validate_agent,
    validate_broker,
    validate_deployment,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("broker-linux", "broker-macos", "agent", "deployment-static"))
    parser.add_argument("environment_file", type=Path, nargs="?")
    parser.add_argument("--exec", action="store_true", dest="launch")
    args = parser.parse_args()
    if args.kind == "deployment-static":
        if args.launch:
            fail("--exec is invalid for deployment-static")
        validate_deployment(args.environment_file or Path(__file__).resolve().parent)
        return 0
    if args.kind == "broker-linux" and not sys.platform.startswith("linux"):
        fail("broker-linux requires Linux systemd egress enforcement")
    if args.kind == "broker-macos" and sys.platform != "darwin":
        fail("broker-macos is only valid on macOS")
    if args.kind == "broker-macos":
        fail("launchd has no egress sandbox; broker startup requires an externally enforced deny-all sandbox")
    if args.environment_file is None:
        fail("environment_file is required")
    values = load_env(args.environment_file)
    if args.kind == "broker-linux":
        validate_broker(values)
        executable = values["SKARBIEC_BINARY"]
        # The host's one Skarbiec process, serving only its capability socket:
        # this deployment allows the broker no TCP, and Skarbiec has no
        # separate broker command any more.
        argv = [executable, "serve", "--no-http"]
    else:
        validate_agent(values)
        executable = values["SINGULARITY_BOOTSTRAP_BINARY"]
        argv = [executable]
    if args.launch:
        os.environ.clear()
        os.environ.update(values)
        os.environ["PATH"] = LAUNCH_PATH
        os.environ["LANG"] = LAUNCH_LOCALE
        os.environ["LC_ALL"] = LAUNCH_LOCALE
        os.execv(executable, argv)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"capability-preflight: {error}", file=sys.stderr)
        raise SystemExit(CONFIGURATION_EXIT_CODE)
