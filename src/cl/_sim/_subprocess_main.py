"""Neutral entry point for SDK Popen child processes."""
from __future__ import annotations

import argparse
import importlib
import json
from typing import Any

def _load_target(target: str):
    module_name, separator, function_name = target.partition(":")
    if not separator:
        raise ValueError(f"Subprocess target must be 'module:function', got {target!r}")
    module = importlib.import_module(module_name)
    return getattr(module, function_name)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CL SDK subprocess target")
    parser.add_argument("--process-name", default="cl-subprocess")
    parser.add_argument("--target", required=True)
    parser.add_argument("--config-json")
    args = parser.parse_args()

    target = _load_target(args.target)
    if args.config_json is None:
        target()
    else:
        config: dict[str, Any] = json.loads(args.config_json)
        target(config)

if __name__ == "__main__":
    main()
