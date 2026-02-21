from __future__ import annotations

import http.server
import json
import logging
import socket
import socketserver
import threading
from pathlib import Path
from typing import ClassVar, override
from urllib.parse import unquote

_logger = logging.getLogger("cl.http_server")

def _find_available_port(host: str = "127.0.0.1", start_port: int = 8000, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find an available port in range {start_port}-{start_port + max_attempts}")

class WebStaticHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler that serves web visualization files under /visualiser."""

    # Class-level configuration
    web_directory  : Path | None = None
    websocket_port : int         = 1025
    websocket_host : str         = "127.0.0.1"

    app_html: ClassVar[str | None]  = None

    def __init__(self, *args, **kwargs):
        # Set the directory for SimpleHTTPRequestHandler
        super().__init__(*args, directory=str(self.web_directory.parent) if self.web_directory else None, **kwargs)

    def translate_path(self, path: str) -> str:
        """Translate URL path to filesystem path, mapping /visualiser to the web directory."""
        # Decode URL encoding
        path = unquote(path)

        # Handle /visualiser prefix
        if path.startswith("/visualiser"):
            # Remove /visualiser prefix and map to web directory
            relative_path = path[11:].lstrip("/")  # Remove '/visualiser' and any leading slash

            if not relative_path:
                relative_path = "index.html"

            if self.web_directory:
                return str(self.web_directory / relative_path)

        # Return empty for non-visualiser paths (will result in 404)
        return ""

    @override
    def do_GET(self) -> None:
        """Handle GET requests."""
        # Handle the special /_/config endpoint to provide WebSocket configuration
        if self.path in {"/_/config", "/_/config.json"}:
            self._serve_config()
            return

        # Handle the /app endpoint for app visualiser
        if self.path in {"/app", "/app/"}:
            self._serve_app_visualiser()
            return

        # Redirect root to /visualiser
        if self.path in {"/", ""}:
            self.send_response(302)
            self.send_header("Location", "/visualiser")
            self.end_headers()
            return

        # Only serve files under /visualiser
        if not self.path.startswith("/visualiser"):
            self.send_error(404, "Not Found")
            return

        # Use parent implementation for static files
        super().do_GET()

    @override
    def end_headers(self) -> None:
        """Add CORS headers to all responses."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    @override
    def log_message(self, format: str, *args) -> None:
        """Log HTTP requests to the cl.websocket logger."""
        _logger.debug("HTTP: %s", args[0])

    def _serve_config(self) -> None:
        """Serve WebSocket configuration as JSON."""
        config = {
            "websocket_host"    : self.websocket_host,
            "websocket_port"    : self.websocket_port,
            "websocket_url"     : f"ws://{self.websocket_host}:{self.websocket_port}",
            "overview_url"      : f"ws://{self.websocket_host}:{self.websocket_port}/_/ws/overview",
            "live_streaming_url": f"ws://{self.websocket_host}:{self.websocket_port}/_/ws/live_streaming",
        }
        content = json.dumps(config).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def _serve_app_visualiser(self) -> None:
        """Serve the app visualiser HTML."""
        if self.app_html is None:
            self.send_error(404, "App visualiser not configured")
            return

        content = self.app_html.encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """HTTP server that handles each request in a separate thread."""
    allow_reuse_address = True
    daemon_threads      = True

class StaticHttpServer:
    """
    HTTP server that serves the MEA visualization static files.

    Runs on a random available port and serves files from the
    visualisation/web directory.
    """

    def __init__(
        self,
        websocket_host: str        = "127.0.0.1",
        websocket_port: int        = 1025,
        host          : str        = "127.0.0.1",
        app_html      : str | None = None,
    ):
        self._host           = host
        self._websocket_host = websocket_host
        self._websocket_port = websocket_port
        self._app_html       = app_html

        self._port  : int                | None = None
        self._server: ThreadedHTTPServer | None = None
        self._thread: threading.Thread   | None = None

        self._running = False

        # Find the web directory relative to this file
        self._web_directory = Path(__file__).parent / "web"
        if not self._web_directory.exists():
            _logger.warning("Web visualization directory not found: %s", self._web_directory)

    @property
    def port(self) -> int | None:
        """The port the server is running on, or None if not started."""
        return self._port

    @property
    def url(self) -> str | None:
        """The full URL to access the visualization, or None if not started."""
        if self._port is None:
            return None
        return f"http://{self._host}:{self._port}/visualiser"

    @property
    def app_url(self) -> str | None:
        """The URL for the application visualiser, or None if not configured."""
        if self._port is None:
            return None
        return f"http://{self._host}:{self._port}/app" if self._app_html else None

    def start(self) -> None:
        """Start the HTTP server in a background thread."""
        if self._running:
            return

        if not self._web_directory.exists():
            _logger.warning("Cannot start MEA server: visualization directory not found")
            return

        # Find an available port
        self._port = _find_available_port(self._host)

        # Configure the handler class
        WebStaticHandler.web_directory  = self._web_directory
        WebStaticHandler.websocket_host = self._websocket_host
        WebStaticHandler.websocket_port = self._websocket_port
        WebStaticHandler.app_html       = self._app_html

        # Create and start the server
        self._server  = ThreadedHTTPServer((self._host, self._port), WebStaticHandler)
        self._running = True

        self._thread = threading.Thread(
            target = self._server.serve_forever,
            daemon = True,
            name   = "cl-mea-http-server"
        )
        self._thread.start()

        _logger.info("MEA visualization server started at %s", self.url)

    def stop(self) -> None:
        """Stop the HTTP server."""
        if not self._running:
            return

        self._running = False

        if self._server:
            self._server.shutdown()
            self._server = None

        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        _logger.info("MEA visualization server stopped at %s", self.url)
