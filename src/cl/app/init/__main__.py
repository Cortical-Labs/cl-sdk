"""
cl.app.init - Application boilerplate generator

Usage:
    python -m cl.app.init [target_directory]

Creates an application skeleton in the target directory with the following structure:
    <target>/
    ├── info.json          # Application metadata
    ├── default.json       # Default configuration
    ├── src/
    │   ├── __init__.py    # Application entry point
    │   └── app.py         # Application class definitions
    ├── web/               # Custom visualisation (empty)
    └── presets/           # Preset configurations (empty)
"""

import json
import sys
from pathlib import Path

_INFO_JSON = {
    "name": "My Application",
    "version": "0.1.0",
    "description": "A brief description of what this application does.",
    "author": "Your Name",
    "config_version": 1,
}

_DEFAULT_JSON = {
    "name": "Default",
    "timeout_s": 43200,  # 12 hours
}

_APP_PY = '''\
from typing import Annotated, override

from cl.app import BaseApplication, BaseApplicationConfig, OutputType, RunSummary
from cl.app.model import DurationSeconds
from pydantic import Field


class MyApplicationConfig(BaseApplicationConfig):
    """Configuration for MyApplication."""

    # Example configuration field - replace with your own
    duration: Annotated[
        DurationSeconds,
        Field(
            title="Duration",
            description="Duration of the application run in seconds.",
            default=60,
        ),
    ]

    @override
    def estimate_duration_s(self) -> float:
        return self.duration


class MyApplication(BaseApplication[MyApplicationConfig]):
    """My custom application."""

    @override
    def run(self, config: MyApplicationConfig, output_directory: str) -> RunSummary | None:
        # TODO: Implement your application logic here
        return RunSummary(
            type    = OutputType.TEXT,
            content = "Application completed successfully.",
        )

    @staticmethod
    @override
    def config_class() -> type[MyApplicationConfig]:
        return MyApplicationConfig
'''

_INIT_PY = '''\
from .app import MyApplication

application = MyApplication()
'''

def main() -> int:
    # Determine target directory
    if len(sys.argv) > 2:
        print("Usage: python -m cl.app.init [target_directory]", file=sys.stderr)
        return 1

    if len(sys.argv) == 2:
        target = Path(sys.argv[1]).resolve()
    else:
        target = Path.cwd()

    # Create target directory if it doesn't exist
    target.mkdir(parents=True, exist_ok=True)

    # Right now we'll only support empty target directories so we don't accidentally overwrite files
    if any(target.iterdir()):
        print(f"Error: Target directory '{target}' is not empty.", file=sys.stderr)
        return 1

    # Create subdirectories
    src_dir = target / "src"
    web_dir = target / "web"
    presets_dir = target / "presets"

    src_dir.mkdir(exist_ok=True)
    web_dir.mkdir(exist_ok=True)
    presets_dir.mkdir(exist_ok=True)

    # Create info.json
    info_file = target / "info.json"
    info_file.write_text(json.dumps(_INFO_JSON, indent=4) + "\n")
    print(f"Created: {info_file}")

    # Create default.json
    default_file = target / "default.json"
    default_file.write_text(json.dumps(_DEFAULT_JSON, indent=4) + "\n")
    print(f"Created: {default_file}")

    # Create src/__init__.py
    init_file = src_dir / "__init__.py"
    init_file.write_text(_INIT_PY)
    print(f"Created: {init_file}")

    # Create src/app.py
    app_file = src_dir / "app.py"
    app_file.write_text(_APP_PY)
    print(f"Created: {app_file}")

    print(f"\nApplication skeleton created in: {target}")
    print("\nNext steps:")
    print("  1. Edit info.json with your application details")
    print("  2. Modify src/app.py with your application logic")
    print("  3. Update default.json with your default configuration values")
    print("  4. Run 'python -m cl.app.pack' to package for installation")

    return 0

if __name__ == "__main__":
    sys.exit(main())
