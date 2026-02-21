"""
## Cortical Labs Application Packager

This module is a command-line tool only and cannot be imported directly.

Usage:
    ```shell
    python -m cl.app.pack [target_directory]
    ```

Without arguments, packages the application in the current directory.
With a path argument, packages the application at that path.

The packager performs the following validation steps:

1. **Folder name validation** - Verifies the application folder name:

   - Contains only alphanumeric characters, hyphens, underscores, and periods
   - Warns if using the `cl-` prefix (reserved for system applications)

2. **Structure validation** - Checks for required files:

   - `info.json`
   - `default.json`
   - `src/__init__.py`

3. **info.json validation** - Validates application metadata:

   - File is valid JSON
   - Contains required fields: `name`, `version`, `description`, `author`, `config_version`
   - Field types are correct (strings and integer for `config_version`)

4. **Web visualisation validation** - If `web/` directory exists:

   - Both `vis.html` and `vis.mjs` must be present together
   - `vis.mjs` must define `const dataStreams = [...]`
   - `vis.mjs` must define `function createVisualiser(uniqueId, div)`

5. **Import validation** - Verifies the application code:

   - The application can be imported without errors
   - An `application` variable is defined in `src/__init__.py`
   - The `application` is an instance of `cl.app.BaseApplication`
   - The application's config class can be retrieved

6. **Configuration validation** - Validates configuration files:

   - `default.json` is valid JSON with required fields (`name`, `timeout_s`)
   - All `presets/*.json` files are valid JSON with required fields
   - All configuration files can be deserialized using the application's config class
   - No duplicate config names across default and presets

7. **Packaging** - Creates a ZIP archive containing the application folder.
   The following are automatically excluded:

   - OS-generated files (`.DS_Store`, `Thumbs.db`, etc.)
   - Python cache (`__pycache__`, `*.pyc`)
   - Hidden directories (starting with `.`)
   - Version control directories (`.git`, `.svn`, `.hg`)
   - Editor/IDE files (`.vscode`, `.idea`)

The output ZIP is created in the parent directory of the target,
named `<folder_name>.zip`.

**Note:** A warning (not error) is issued if the uncompressed package size exceeds 100 MB.

Example:
    Package an application:
        ```shell
        cd my-app
        python -m cl.app.pack
        # Creates ../my-app.zip
        ```
"""

__all__: list[str] = []  # Nothing to export
