"""
Cortical Labs Recording Playback Tool

Usage:
    python -m cl.playback <h5_recording_file> [app_directory]

Arguments:
    h5_recording_file: Path to the HDF5 recording file to play back
    app_directory:     Optional path to the application directory containing
                       the app's visualiser (web/ directory with vis.html, vis.mjs)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

# Enable WebSocket server for visualiser
os.environ["CL_SDK_WEBSOCKET"] = "1"

from cl.util import RecordingView, in_vscode
from cl.visualisation._websocket_subprocess import WebSocketProcessManager

from ._cli_controller import PlaybackController
from ._playback_producer import PlaybackProducer

_logger = logging.getLogger("cl.playback")

@dataclass
class RecordingMetadata:
    """Metadata extracted from a recording file."""
    channel_count          : int
    frames_per_second      : int
    duration_frames        : int
    start_timestamp        : int
    data_stream_attributes : dict[str, dict]

def _setup_visualiser_html(
    app_id  : str,
    html_str: str,
    js_str  : str,
    css_str : str | None = None,
) -> str:
    """
    Build the visualiser HTML content from the provided content strings.

    Args:
        app_id: The application ID (used for the page title)
        html_str: The HTML content for the visualiser body
        js_str: The JavaScript code for the visualiser
        css_str: Optional CSS content for styling

    Returns the complete HTML page content string.
    """
    ssh_ip = None
    if in_vscode() and "SSH_CONNECTION" in os.environ:
        ssh_connection = os.environ["SSH_CONNECTION"]
        ssh_ip = ssh_connection.split(" ")[2]

    ws_port = os.environ.get("CL_SDK_WEBSOCKET_PORT", "1025")

    css_block = f"<style>{css_str}</style>" if css_str else ""

    html_content = f"""
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{app_id} Playback</title>
    <link rel="stylesheet" href="/visualiser/visualiser-base.css">
    {css_block}
    <script src="/visualiser/msgpack-lite-numpy.js"></script>
</head>

<body>
    <div id="visualiser">
        {html_str}
    </div>
</body>

<script>
    {js_str}
</script>
<script>
    const uniqueId = "visualiser";
    const sshIp    = {json.dumps(ssh_ip)};
    const websocketPort = {ws_port};
</script>
<script src="/visualiser/engine.mjs"></script>

</html>
"""
    return html_content

def _load_app_visualiser(app_path: Path | None) -> str | None:
    """Load the app visualiser HTML from a directory or zip file.

    For directories: expects web/vis.html and web/vis.mjs
    For zip files: expects a single top-level directory containing web/vis.html and web/vis.mjs
    """
    if app_path is None:
        return None

    # Check if it's a zip file
    if app_path.suffix.lower() == ".zip" and app_path.is_file():
        try:
            with zipfile.ZipFile(app_path, 'r') as zf:
                # Find the top-level directory in the zip
                names = zf.namelist()
                top_dirs = set()
                for name in names:
                    parts = name.split('/')
                    if parts[0]:  # Skip empty parts
                        top_dirs.add(parts[0])

                if len(top_dirs) != 1:
                    _logger.error("Zip file must contain exactly one top-level directory, found: %s", top_dirs)
                    raise SystemExit(1)

                app_id     = next(iter(top_dirs))
                web_prefix = f"{app_id}/web/"

                # Check for required visualiser files
                html_path = f"{web_prefix}vis.html"
                js_path   = f"{web_prefix}vis.mjs"
                css_path  = f"{web_prefix}vis.css"

                if html_path not in zf.namelist() or js_path not in zf.namelist():
                    _logger.warning(
                        "Visualiser files not found in zip (need %s/web/vis.html and %s/web/vis.mjs)",
                        app_id, app_id
                    )
                    return None

                # Read files from zip into memory
                html_str = zf.read(html_path).decode('utf-8')
                js_str   = zf.read(js_path).decode('utf-8')
                css_str  = zf.read(css_path).decode("utf-8") if css_path in zf.namelist() else None

                app_html = _setup_visualiser_html(app_id, html_str, js_str, css_str)
                _logger.info("Loaded visualiser from zip %s", app_path)
                return app_html

        except zipfile.BadZipFile as e:
            _logger.error("Invalid zip file: %s", app_path)
            raise RuntimeError(f"Invalid zip file: {app_path}") from e
        except Exception as e:
            _logger.error("Error reading zip file: %s", e)
            raise RuntimeError(f"Error reading zip file: {e}") from e

    # Handle directory case
    if not app_path.is_dir():
        _logger.error("Application path '%s' does not exist or is not a zip file.", app_path)
        raise RuntimeError(f"Application path '{app_path}' does not exist or is not a zip file.")

    web_path = app_path / "web"
    if web_path.is_dir():
        html_file = web_path / "vis.html"
        js_file   = web_path / "vis.mjs"
        css_file  = web_path / "vis.css"

        if not html_file.is_file() or not js_file.is_file():
            _logger.warning("Visualiser files not found in %s (need vis.html and vis.mjs)", web_path)
            return None

        # Read files from disk
        html_str = html_file.read_text(encoding="utf-8")
        js_str   = js_file.read_text(encoding="utf-8")
        css_str  = css_file.read_text(encoding="utf-8") if css_file.is_file() else None

        app_html = _setup_visualiser_html(app_path.name, html_str, js_str, css_str)
        _logger.info("Loaded visualiser from %s", web_path)
        return app_html

    _logger.warning("No web/ directory found in %s", app_path)
    return None

def _load_recording_metadata(recording_file: Path) -> RecordingMetadata:
    """Load metadata from a recording file."""
    _logger.info("Opening recording: %s", recording_file)

    recording = RecordingView(str(recording_file))
    attrs     = recording.attributes

    # Extract data stream attributes
    data_stream_attributes: dict[str, dict] = {}
    if recording.data_streams is not None:
        for ds_name in recording.data_streams:
            ds = recording.data_streams[ds_name]
            # Convert AttributesView to dict
            data_stream_attributes[ds_name] = dict(ds.attributes.application)

    metadata = RecordingMetadata(
        channel_count          = attrs.get("channel_count", 64),
        frames_per_second      = attrs.get("frames_per_second", 25000),
        duration_frames        = attrs.get("duration_frames", 0),
        start_timestamp        = attrs.get("start_timestamp", 0),
        data_stream_attributes = data_stream_attributes,
    )

    recording.close()
    return metadata

def _run_playback(
    recording_file: Path,
    metadata      : RecordingMetadata,
    app_html      : str | None,
    host          : str,
    port          : int,
) -> int:
    """Run the playback session."""
    # Create and start the playback producer
    producer = PlaybackProducer(
        replay_file_path  = str(recording_file),
        channel_count     = metadata.channel_count,
        frames_per_second = metadata.frames_per_second,
        duration_frames   = metadata.duration_frames,
        start_timestamp   = metadata.start_timestamp,
    )

    _logger.info("Starting playback producer...")
    producer.start()

    if producer.name_prefix is None:
        producer.stop()
        raise RuntimeError("Playback producer did not initialize properly")

    # Start the WebSocket server subprocess
    _logger.info("Starting WebSocket server...")

    ws_manager = WebSocketProcessManager(
        buffer_name       = producer.name_prefix,
        frames_per_second = metadata.frames_per_second,
        channel_count     = metadata.channel_count,
        port              = port,
        host              = host,
        serve_vis         = True,
        app_html          = app_html,
    )

    try:
        ws_manager.start()
    except Exception:
        producer.stop()
        raise

    # Send data stream attributes to WebSocket server
    for ds_name, ds_attrs in metadata.data_stream_attributes.items():
        _logger.debug("Sending attributes for data stream '%s'", ds_name)
        ws_manager.send_attribute_reset(ds_name, ds_attrs)
        ws_manager.send_attribute_update(ds_name, ds_attrs)

    # Print URLs
    print()
    if ws_manager.web_url:
        print(f"Data visualisation: {ws_manager.web_url}")
    if ws_manager.app_url:
        print(f"App visualisation:  {ws_manager.app_url}")
    print()

    # Set up signal handler for graceful shutdown
    def signal_handler(_signum: int, _frame) -> None:
        pass  # Just let the exception propagate

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create and run the CLI controller
    controller = PlaybackController(
        producer          = producer,
        frames_per_second = metadata.frames_per_second,
        duration_frames   = metadata.duration_frames,
    )

    try:
        controller.run()
    except KeyboardInterrupt:
        pass
    finally:
        _logger.info("Shutting down...")
        ws_manager.stop()
        producer.stop()

    print("Playback finished.")
    return 0

def _main() -> int:
    parser = argparse.ArgumentParser(
        description     = "Cortical Labs Recording Playback Tool",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog          = """
Controls:
  SPACE                                   Toggle pause/play
  Left/Right                              Skip ±5 seconds
  Shift (Ctrl on Windows) + Left/Right    Skip ±1 minute
  g                                       Go to specific timestamp
  r                                       Restart
  h / ?                                   Show help
  q                                       Quit
"""
    )
    parser.add_argument(
        "recording_file",
        type = Path,
        help = "Path to the HDF5 recording file to play back"
    )
    parser.add_argument(
        "app_directory",
        type    = Path,
        nargs   = "?",
        default = None,
        help    = "Optional path to the application directory or zip file containing the visualiser"
    )
    parser.add_argument(
        "--port",
        type    = int,
        default = 1025,
        help    = "WebSocket server port (default: 1025)"
    )
    parser.add_argument(
        "--host",
        type    = str,
        default = "127.0.0.1",
        help    = "WebSocket server host (default: 127.0.0.1)"
    )
    args = parser.parse_args()

    recording_file: Path = args.recording_file

    os.environ["CL_SDK_WEBSOCKET_PORT"] = str(args.port)  # Pass port to WebSocket server via environment variable

    # Validate recording file
    if not recording_file.is_file():
        print(f"Error: Recording file '{recording_file}' does not exist.", file=sys.stderr)
        return 1

    try:
        app_html = _load_app_visualiser(args.app_directory)
        metadata = _load_recording_metadata(recording_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if metadata.duration_frames == 0:
        print("Error: Recording has no frames.", file=sys.stderr)
        return 1

    duration_seconds = metadata.duration_frames / metadata.frames_per_second
    _logger.info("Recording: %d channels, %d Hz, %.1f seconds (%d frames)",
                 metadata.channel_count, metadata.frames_per_second,
                 duration_seconds, metadata.duration_frames)

    try:
        return _run_playback(recording_file, metadata, app_html, args.host, args.port)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(_main())
