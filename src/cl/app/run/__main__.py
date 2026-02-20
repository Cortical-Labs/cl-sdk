import os

# Setup environment variables for app runner mode
os.environ["CL_SDK_WEBSOCKET"] = "1"

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from cl import Neurons
from cl.app import BaseApplication, BaseApplicationConfig
from cl.util import in_vscode

def _setup_visualiser(app_id: str, web_path: Path):
    html_path = web_path / "vis.html"
    js_path   = web_path / "vis.mjs"

    if not html_path.is_file() or not js_path.is_file():
        print(f"Warning: Visualiser files not found in {web_path}, skipping visualiser setup", file=sys.stderr)
        return

    css_path = web_path / "vis.css"  # Optional CSS file for the visualiser

    ssh_ip = None
    if in_vscode() and "SSH_CONNECTION" in os.environ:
        ssh_connection = os.environ["SSH_CONNECTION"]
        ssh_ip = ssh_connection.split(" ")[2]

    ws_port = os.environ.get("CL_SDK_WEBSOCKET_PORT", "1025")

    html_content = f"""
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{app_id} Visualiser</title>
    <link rel="stylesheet" href="/visualiser/visualiser-base.css">
    {f"<style>{css_path.read_text()}</style>" if css_path.is_file() else ""}
    <script src="/visualiser/msgpack-lite-numpy.js"></script>
</head>

<body>
    <div id="visualiser">
        {html_path.read_text(encoding="utf-8")}
    </div>
</body>

<script>
    {js_path.read_text(encoding="utf-8")}
</script>
<script>
    const uniqueId = "visualiser";
    const sshIp    = {json.dumps(ssh_ip)};
    const websocketPort = {ws_port};
</script>
<script src="/visualiser/engine.mjs"></script>

</html>
"""
    Neurons._app_html = html_content

def _main() -> int:
    parse = argparse.ArgumentParser(description="Cortical Labs Application Runner")
    parse.add_argument("target_dir", help="Path to the application directory (default: current directory)")
    parse.add_argument("config_json", help="Path to the configuration JSON file")
    parse.add_argument("output_dir", nargs="?", default=".", help="Optional path to an output directory for logs or other generated files (default: current directory)")
    args = parse.parse_args()

    app_path    = Path(args.target_dir).resolve()
    config_path = Path(args.config_json).resolve()

    # Validate inputs
    if not app_path.is_dir():
        print(f"Error: Target directory not found: {app_path}", file=sys.stderr)
        return 1

    if not config_path.is_file():
        print(f"Error: Configuration JSON file not found: {config_path}", file=sys.stderr)
        return 1

    # Deserialise the config JSON into a dict
    try:
        with config_path.open(encoding="utf-8") as f:
            config_dict = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse configuration JSON file: {e}", file=sys.stderr)
        return 1

    if not isinstance(config_dict, dict):
        print("Error: Configuration JSON must be an object at the top level", file=sys.stderr)
        return 1

    # Only do basic validation of the config here, the app itself will perform more detailed validation as needed.
    if "name" not in config_dict or not isinstance(config_dict["name"], str):
        print("Error: Configuration JSON must contain a 'name' field of type string", file=sys.stderr)
        return 1

    if "timeout_s" not in config_dict or not isinstance(config_dict["timeout_s"], int):
        print("Error: Configuration JSON must contain a 'timeout_s' field of type integer", file=sys.stderr)
        return 1

    # Try to import the app as a module
    init_py = app_path / "src" / "__init__.py"
    if not init_py.is_file():
        print(f"Error: Could not find application entry point at expected location: {init_py}", file=sys.stderr)
        return 1

    module_name = app_path.name
    spec = importlib.util.spec_from_file_location(module_name, init_py, submodule_search_locations=[str(app_path / "src")])

    if spec is None:
        print(f"Error: Could not create module spec for application at {init_py}", file=sys.stderr)
        return 1

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    if spec.loader is None:
        print(f"Error: Module spec for application at {init_py} does not have a loader", file=sys.stderr)
        return 1

    spec.loader.exec_module(module)

    if not hasattr(module, "application"):
        print(f"Error: Application module at {init_py} does not define an 'application' object", file=sys.stderr)
        return 1

    application = module.application

    if not isinstance(application, BaseApplication):
        print(f"Error: 'application' object in {init_py} is not an instance of BaseApplication", file=sys.stderr)
        return 1

    config_class: type[BaseApplicationConfig] = application.config_class()

    config = config_class.model_validate(config_dict)

    if not isinstance(config, BaseApplicationConfig):
        print("Error: Failed to validate configuration JSON against the application's config schema", file=sys.stderr)
        return 1

    web_path = app_path / "web"
    if web_path.is_dir():
        _setup_visualiser(app_path.name, web_path)

    print(f"Running application in {app_path} with configuration {config_path}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output = application.run(config=config, output_directory=str(output_dir))

    if output is not None:
        from pprint import pprint
        print("Application RunSummary output:")
        pprint(output)

    print("Application run completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(_main())
