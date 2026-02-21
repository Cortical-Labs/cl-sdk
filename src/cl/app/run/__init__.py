"""
## Cortical Labs Application Runner

This module is a command-line tool only and cannot be imported directly. This tool is only intended for use in development of applications,
and should only be run on a local machine. This module does not perform any safety or validation checks.

Usage:
    ```shell
    python -m cl.app.run <target_directory> <configuration JSON path>
    ```

The target directory is treated as the base application folder. If the application has a custom visualiser implemented,
the visualiser will be launched in the default web browser upon starting the run.
A valid configuration JSON file must also be provided as an argument.

Example:
    Run an application located in `my-app/`:

        ```shell
        python -m cl.app.run my-app my-app/default.json
        ```
"""

__all__: list[str] = []  # Nothing to export
