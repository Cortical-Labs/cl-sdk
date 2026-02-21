"""
## Cortical Labs Application Boilerplate Generator

This module is a command-line tool only and cannot be imported directly.

Usage:
    ```shell
    python -m cl.app.init [target_directory]
    ```

Without arguments, creates an application skeleton in the current directory.
With a path argument, creates the skeleton in that directory (creating it if needed).

The generated structure includes:

- `info.json` - Application metadata (name, version, author, etc.)
- `default.json` - Default configuration with `name` and `timeout_s` fields
- `src/__init__.py` - Application entry point with `application` instance
- `src/app.py` - Application and config class definitions
- `web/` - Directory for web visualisation files (empty)
- `presets/` - Directory for preset configurations (empty)

Example:
    Create a new application in `my-app/`:

        ```shell
        python -m cl.app.init my-app
        cd my-app
        # Edit files as needed...
        # Package the application
        python -m cl.app.pack
        # Creates ../my-app.zip
        # Upload the package to the device
        ```
"""

__all__: list[str] = []  # Nothing to export
