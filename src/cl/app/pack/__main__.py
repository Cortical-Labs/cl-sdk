"""
cl.app.pack - Application packager

Usage:
    python -m cl.app.pack [target_directory]

Validates and packages an application directory into a ZIP file suitable
for installation on Cortical Labs devices.

The resulting ZIP file is created in the parent directory of the target
and named <folder_name>.zip.
"""

import json
import re
import sys
import zipfile
from pathlib import Path

# Patterns to exclude from the package
_EXCLUDE_PATTERNS = {
    # OS-generated files
    ".DS_Store",
    ".ds_store",
    "Thumbs.db",
    "thumbs.db",
    "Desktop.ini",
    "desktop.ini",
    ".Spotlight-V100",
    ".Trashes",
    "ehthumbs.db",
    "ehthumbs_vista.db",
    # macOS resource forks
    "__MACOSX",
    # Python cache
    "__pycache__",
    # Byte-compiled files
    "*.pyc",
    "*.pyo",
    "*.pyd",
    # Editor/IDE files
    ".vscode",
    ".idea",
    "*.swp",
    "*.swo",
    # Version control
    ".git",
    ".svn",
    ".hg",
}

def _should_exclude(path: Path, base_path: Path) -> bool:
    """Check if a path should be excluded from the package."""
    rel_path = path.relative_to(base_path)

    for part in rel_path.parts:
        # Exclude hidden directories (starting with .)
        if part.startswith(".") and part not in {".", ".."}:
            return True

        # Exclude by name
        if part in _EXCLUDE_PATTERNS:
            return True

        # Check wildcard patterns
        for pattern in _EXCLUDE_PATTERNS:
            if pattern.startswith("*") and part.endswith(pattern[1:]):
                return True

    return False

def _validate_folder_name(folder_name: str) -> list[str]:
    """
    Validate the application folder name (app ID).

    Returns a list of error messages, empty if valid.
    """
    errors: list[str] = []

    # Pattern: alphanumeric, hyphens, underscores, periods only
    if not re.match(r"^[\w.-]+$", folder_name):
        errors.append(
            f"Invalid folder name '{folder_name}': must contain only alphanumeric characters, "
            "hyphens, underscores, and periods"
        )

    # Warn about cl- prefix (reserved for system apps)
    if folder_name.startswith("cl-"):
        print(
            f"Warning: Folder name '{folder_name}' uses the 'cl-' prefix which is reserved for system applications"
        )

    return errors


def _validate_info_json(target: Path) -> list[str]:
    """
    Validate info.json file.

    Returns a list of error messages, empty if valid.
    """
    errors: list[str] = []
    info_json = target / "info.json"

    if not info_json.exists():
        return errors  # Already caught by structure validation

    try:
        with open(info_json, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"info.json is not valid JSON: {e}")
        return errors
    except Exception as e:
        errors.append(f"Failed to read info.json: {e}")
        return errors

    # Check required fields
    required_fields = {
        "name": str,
        "version": str,
        "description": str,
        "author": str,
        "config_version": int,
    }

    for field, expected_type in required_fields.items():
        if field not in data:
            errors.append(f"info.json missing required field: '{field}'")
        elif not isinstance(data[field], expected_type):
            errors.append(
                f"info.json field '{field}' must be {expected_type.__name__}, "
                f"got {type(data[field]).__name__}"
            )

    return errors


def _validate_web_visualisation(target: Path) -> list[str]:
    """
    Validate web visualisation files.

    Returns a list of error messages, empty if valid.
    """
    errors: list[str] = []
    web_dir = target / "web"

    if not web_dir.exists():
        return errors  # Web visualisation is optional

    vis_html = web_dir / "vis.html"
    vis_mjs = web_dir / "vis.mjs"

    # If either exists, both must exist
    if vis_html.exists() and not vis_mjs.exists():
        errors.append("web/vis.html exists but web/vis.mjs is missing")
    elif vis_mjs.exists() and not vis_html.exists():
        errors.append("web/vis.mjs exists but web/vis.html is missing")

    # Validate vis.mjs content
    if vis_mjs.exists():
        try:
            content = vis_mjs.read_text()

            # Check for dataStreams const array
            if not re.search(r"\bconst\s+dataStreams\s*=\s*\[", content):
                errors.append(
                    "web/vis.mjs must define 'const dataStreams = [...]' "
                    "(array of data stream names)"
                )

            # Check for createVisualiser function
            if not re.search(r"\bfunction\s+createVisualiser\s*\(\s*\w+\s*,\s*\w+\s*\)", content):
                errors.append(
                    "web/vis.mjs must define 'function createVisualiser(uniqueId, div)'"
                )

        except Exception as e:
            errors.append(f"Failed to read web/vis.mjs: {e}")

    return errors

def _validate_application(target: Path) -> list[str]:
    """
    Validate the application structure.

    Returns a list of error messages, empty if valid.
    """
    errors: list[str] = []

    # Check required files
    info_json    = target / "info.json"
    default_json = target / "default.json"
    src_dir      = target / "src"
    src_init     = src_dir / "__init__.py"

    if not info_json.exists():
        errors.append("Missing required file: info.json")

    if not default_json.exists():
        errors.append("Missing required file: default.json")

    if not src_dir.exists() or not src_dir.is_dir():
        errors.append("Missing required directory: src/")
    elif not src_init.exists():
        errors.append("Missing required file: src/__init__.py")

    return errors

def _validate_import(target: Path) -> tuple[list[str], type | None]:
    """
    Validate that the application can be imported and has `application` defined.

    Returns a tuple of (errors, config_class) where errors is a list of error messages
    and config_class is the application's config class if successfully loaded.
    """
    import importlib

    errors: list[str] = []
    src_dir = target / "src"
    src_init = src_dir / "__init__.py"

    if not src_init.exists():
        return errors, None  # Already caught by structure validation

    # Add target to path so 'src' can be imported as a package
    sys.path.insert(0, str(target))

    config_class = None

    try:
        # Import src as a proper package (this handles relative imports)
        src_module = importlib.import_module("src")

        if not hasattr(src_module, "application"):
            errors.append("src/__init__.py must define an 'application' variable")
        else:
            # Check if it's an instance of BaseApplication
            from .. import BaseApplication

            if not isinstance(src_module.application, BaseApplication):
                errors.append(
                    f"'application' must be an instance of BaseApplication, "
                    f"got {type(src_module.application).__name__}"
                )
            else:
                # Get the config class
                try:
                    config_class = src_module.application.config_class()
                except Exception as e:
                    errors.append(f"Failed to get config class: {e}")

    except Exception as e:
        errors.append(f"Failed to import application: {e}")
    finally:
        sys.path.pop(0)
        # Clean up all src.* modules from sys.modules
        modules_to_remove = [key for key in sys.modules.keys() if key == "src" or key.startswith("src.")]
        for key in modules_to_remove:
            sys.modules.pop(key, None)

    return errors, config_class

def _validate_configs(target: Path, config_class: type | None) -> list[str]:
    """
    Validate default.json and preset configurations.

    Returns a list of error messages, empty if valid.
    """
    errors: list[str] = []

    if config_class is None:
        errors.append("Cannot validate configs: config class not loaded")
        return errors

    config_names: set[str] = set()

    # Validate default.json
    default_json = target / "default.json"
    if default_json.exists():
        try:
            with open(default_json, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"default.json is not valid JSON: {e}")
            return errors
        except Exception as e:
            errors.append(f"Failed to read default.json: {e}")
            return errors

        # Check required fields
        if "name" not in data:
            errors.append("default.json missing required field: 'name'")
        else:
            config_names.add(data["name"])

        if "timeout_s" not in data:
            errors.append("default.json missing required field: 'timeout_s'")

        # Try to deserialize as config object
        try:
            config_class(**data)
        except Exception as e:
            errors.append(f"default.json does not match config schema: {e}")

    # Validate presets
    presets_dir = target / "presets"
    if presets_dir.exists() and presets_dir.is_dir():
        preset_files = sorted(presets_dir.glob("*.json"))

        for preset_file in preset_files:
            try:
                with open(preset_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                errors.append(f"{preset_file.name} is not valid JSON: {e}")
                continue
            except Exception as e:
                errors.append(f"Failed to read {preset_file.name}: {e}")
                continue

            # Check required fields
            if "name" not in data:
                errors.append(f"{preset_file.name} missing required field: 'name'")
            else:
                preset_name = data["name"]
                if preset_name in config_names:
                    errors.append(
                        f"{preset_file.name} has duplicate config name '{preset_name}'"
                    )
                else:
                    config_names.add(preset_name)

            if "timeout_s" not in data:
                errors.append(f"{preset_file.name} missing required field: 'timeout_s'")

            # Try to deserialize as config object
            try:
                config_class(**data)
            except Exception as e:
                errors.append(f"{preset_file.name} does not match config schema: {e}")

    return errors

def _create_package(target: Path, output_path: Path) -> tuple[int, int]:
    """
    Create the ZIP package.

    Returns a tuple of (file_count, uncompressed_size_bytes).
    """
    folder_name = target.name
    file_count = 0
    total_size = 0

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in target.rglob("*"):
            if file_path.is_file() and not _should_exclude(file_path, target):
                arcname = f"{folder_name}/{file_path.relative_to(target)}"
                zf.write(file_path, arcname)
                file_count += 1
                total_size += file_path.stat().st_size

    return file_count, total_size

def main() -> int:
    # Determine target directory
    if len(sys.argv) > 2:
        print("Usage: python -m cl.app.pack [target_directory]", file=sys.stderr)
        return 1

    if len(sys.argv) == 2:
        target = Path(sys.argv[1]).resolve()
    else:
        target = Path.cwd()

    if not target.exists():
        print(f"Error: Target directory does not exist: {target}", file=sys.stderr)
        return 1

    if not target.is_dir():
        print(f"Error: Target is not a directory: {target}", file=sys.stderr)
        return 1

    print(f"Validating application: {target}")

    # Validate folder name
    errors = _validate_folder_name(target.name)
    if errors:
        print("\nFolder name validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("  ✓ Folder name validation passed")

    # Validate structure
    errors = _validate_application(target)
    if errors:
        print("\nStructure validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("  ✓ Structure validation passed")

    # Validate info.json
    errors = _validate_info_json(target)
    if errors:
        print("\ninfo.json validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("  ✓ info.json validation passed")

    # Validate web visualisation
    errors = _validate_web_visualisation(target)
    if errors:
        print("\nWeb visualisation validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("  ✓ Web visualisation validation passed")

    # Validate import
    errors, config_class = _validate_import(target)
    if errors:
        print("\nImport validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("  ✓ Import validation passed")

    # Validate configs
    errors = _validate_configs(target, config_class)
    if errors:
        print("\nConfiguration validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("  ✓ Configuration validation passed")

    # Create package
    output_path = target.parent / f"{target.name}.zip"

    print(f"\nPackaging to: {output_path}")
    file_count, total_size = _create_package(target, output_path)

    print(f"  ✓ Packaged {file_count} files")

    # Warn about large packages
    max_size = 1024 * 1024 * 1024  # 1 GiB
    if total_size > max_size:
        print(
            f"\n⚠ Warning: Package size ({total_size / 1024 / 1024:.1f} MB) exceeds limit, this will likely fail to install",
            file=sys.stderr,
        )

    print(f"\nDone! Package created: {output_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
