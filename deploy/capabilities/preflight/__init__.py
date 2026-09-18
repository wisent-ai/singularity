"""The fail-closed checks the capability deployment preflight runs."""

from preflight.contract import (
    CONFIGURATION_EXIT_CODE,
    LAUNCH_LOCALE,
    LAUNCH_PATH,
    fail,
    require_secure,
    verify_binary,
)
from preflight.deployment import validate_deployment
from preflight.environment import load_env, validate_agent, validate_broker

__all__ = [
    "CONFIGURATION_EXIT_CODE",
    "LAUNCH_LOCALE",
    "LAUNCH_PATH",
    "fail",
    "load_env",
    "require_secure",
    "validate_agent",
    "validate_broker",
    "validate_deployment",
    "verify_binary",
]
