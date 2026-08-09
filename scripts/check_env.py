"""Validate the local runtime before a live anomaly scan.

This check is intentionally network-free: it verifies the interpreter, required
Python distributions, importability, available torch device, and credential
presence without printing credential values.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import sys


EXPECTED_VERSIONS = {
    "numpy": "1.26.4",
    "pandas": "2.3.3",
    "torch": "2.13.0",
    "panda_data": "0.0.12",
    "pytest": "9.0.2",
}


def _torch_device(torch_module) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    errors: list[str] = []
    print(f"python={platform.python_version()} executable={sys.executable}")
    if sys.version_info[:2] != (3, 10):
        errors.append("requires CPython 3.10")

    for distribution_name, expected in EXPECTED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            print(f"{distribution_name}=MISSING")
            errors.append(f"missing distribution {distribution_name}")
            continue
        print(f"{distribution_name}={actual}")
        if actual != expected:
            errors.append(f"{distribution_name}=={expected} required, got {actual}")

    for module_name in ("numpy", "pandas", "torch", "panda_data"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - exercised in broken envs
            errors.append(f"import {module_name} failed: {type(exc).__name__}: {exc}")

    try:
        torch = importlib.import_module("torch")
        print(f"torch_device={_torch_device(torch)}")
    except Exception:
        pass

    for env_name in ("PANDA_DATA_USERNAME", "PANDA_DATA_PASSWORD"):
        state = "set" if os.environ.get(env_name) else "missing"
        print(f"{env_name}={state}")

    if errors:
        for error in errors:
            print(f"[error] {error}", file=sys.stderr)
        return 1
    print("[ok] runtime dependencies and imports verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
