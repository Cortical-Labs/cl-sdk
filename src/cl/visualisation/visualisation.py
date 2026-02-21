import json
import os
import uuid
from pathlib import Path

from cl.util import in_vscode

def _generate_iframe(
    unique_id    : str,
    *,
    iframe_content: str | None = None,
    iframe_src    : str | None = None,
):
    if iframe_content is not None and iframe_src is not None:
        raise ValueError("Only one of iframe_content or iframe_src should be provided.")

    if iframe_content is not None:
        # Escape the iframe content for srcdoc attribute
        iframe_content = iframe_content.replace('&', '&amp;').replace('"', '&quot;')

    if iframe_src is not None and in_vscode():
        # Try to extract the device IP that the VSCode server is accessing to via the 'SSH_CONNECTION' env var
        device_ip = 'localhost'
        if 'SSH_CONNECTION' in os.environ:
            ssh_connection = os.environ['SSH_CONNECTION']
            device_ip      = ssh_connection.split(' ')[2]

        iframe_src = f"http://{device_ip}/{iframe_src.lstrip('/')}"

    return f"""
<iframe
    id="iframe-{unique_id}"
    {f'srcdoc="{iframe_content}"' if iframe_content is not None else ""}
    {f'src="{iframe_src}"' if iframe_src is not None else ""}
    style="width: 100%; border: none; display: block;"
    scrolling="no"
></iframe>
"""

def _generate_aspect_ratio_observer(unique_id: str, aspect_ratio: float) -> str:
    """Generate JavaScript to maintain aspect ratio in sidebar mode."""
    return f"""
// Maintain aspect ratio in sidebar mode
const iframe = document.getElementById('iframe-{unique_id}');
if (iframe) {{
    const aspectRatio = {aspect_ratio};
    const updateHeight = () => {{
        const width = iframe.offsetWidth;
        iframe.style.height = (width / aspectRatio) + 'px';
    }};

    // Initial height
    updateHeight();

    // Watch for width changes
    const resizeObserver = new ResizeObserver(updateHeight);
    resizeObserver.observe(iframe);
}}
"""

def _generate_message_handlers(unique_id: str) -> str:
    """Generate JavaScript message handlers for resize and scroll events."""
    return f"""
// Listen for resize messages from the iframe
window.addEventListener('message', function(event) {{
    if (event.data && event.data.type === 'resize' && event.data.id === '{unique_id}') {{
        const iframe = document.getElementById('iframe-{unique_id}');
        if (iframe) {{
            iframe.style.height = event.data.height + 'px';
        }}
    }}
    if (event.data && event.data.type === 'scroll' && event.data.id === '{unique_id}') {{
        const iframe = document.getElementById('iframe-{unique_id}');
        if (!iframe) return;

        // Find the scrollable parent by walking up from the iframe
        function findScrollableAncestor(element) {{
            let current = element.parentElement;
            while (current) {{
                const style = window.getComputedStyle(current);
                const overflowY = style.overflowY;
                const isScrollable = (overflowY === 'auto' || overflowY === 'scroll') &&
                                        current.scrollHeight > current.clientHeight;
                if (isScrollable) {{
                    return current;
                }}
                current = current.parentElement;
            }}
            return null;
        }}

        // Find scrollable ancestor starting from iframe
        let scrollTarget = findScrollableAncestor(iframe);

        // If not found, check for sidebar panel
        if (!scrollTarget) {{
            const panel = document.getElementById('cl-visualiser-panel');
            if (panel && panel.contains(iframe) && panel.scrollHeight > panel.clientHeight) {{
                scrollTarget = panel;
            }}
        }}

        // Fallback to document scrolling element
        if (!scrollTarget) {{
            scrollTarget = document.scrollingElement || document.documentElement;
        }}

        // Apply scroll - handle deltaMode for line/page scrolling
        let deltaY = event.data.deltaY || 0;
        let deltaX = event.data.deltaX || 0;
        if (event.data.deltaMode === 1) {{ // DOM_DELTA_LINE
            deltaY *= 16;
            deltaX *= 16;
        }} else if (event.data.deltaMode === 2) {{ // DOM_DELTA_PAGE
            deltaY *= scrollTarget.clientHeight || 400;
            deltaX *= scrollTarget.clientWidth || 400;
        }}

        scrollTarget.scrollBy({{ left: deltaX, top: deltaY }});
    }}
}});
"""

def _generate_wrapper_script(
    unique_id: str,
    use_sidebar: bool,
    aspect_ratio: float | None
) -> str:
    """Generate the wrapper script that handles layout and event forwarding."""
    script_dir            = Path(__file__).parent
    layout_handler        = Path(script_dir / 'layout_handler.mjs').read_text(encoding='utf-8') if use_sidebar and not in_vscode() else ""
    aspect_ratio_observer = _generate_aspect_ratio_observer(unique_id, aspect_ratio) if (aspect_ratio is not None) else ""

    return f"""
        <script type="module">
            const iframeId = 'iframe-{unique_id}';
            {layout_handler}
            {aspect_ratio_observer}
{_generate_message_handlers(unique_id)}
        </script>
        """

def create_iframe_visualiser(iframe_url: str, use_sidebar: bool = True, aspect_ratio: float | None = None) -> str:
    """
    Create the HTML needed to display an iframe in a Jupyter notebook.

    Args:
        iframe_url: URL to load in the iframe for the visualiser.
        use_sidebar: Whether to enable the sidebar layout for the visualiser, when used in Jupyter notebook/lab.
        aspect_ratio: Optional aspect ratio (width/height) for the visualiser. Only applies in sidebar mode.
    """
    unique_id = str(uuid.uuid4())

    return f"""
{_generate_iframe(unique_id, iframe_src=iframe_url)}
{_generate_wrapper_script(unique_id, use_sidebar, aspect_ratio)}
"""


def create_visualiser(
    javascript_file: str | Path,
    html_file      : str | Path | None = None,
    data_streams   : list[str]  | None = None,
    use_sidebar    : bool              = True,
    aspect_ratio   : float | None      = None,
) -> str:
    """
    Create the HTML needed to display a custom visualiser in a Jupyter notebook, using the built-in data stream Javascript engine.

    Args:
        javascript_file: Path to the visualiser's JavaScript module file.
        html_file: Optional path to an HTML file to include in the visualiser.
        data_streams: Optional list of data stream names to connect to the visualiser.
        use_sidebar: Whether to enable the sidebar layout for the visualiser, when used in Jupyter notebook/lab.
        aspect_ratio: Optional aspect ratio (width/height) for the visualiser. Only applies in sidebar mode.
    """

    # Create a unique id for the parent div. It is also used
    # to defeat the js import cache within vscode.
    unique_id = str(uuid.uuid4())

    #
    # We need access to the engine code. Normally this would be
    # a simple js module import, however, we are in a notebook
    # which might be running in vscode and it's not clear what the
    # path to the file will be. So for now, the simple
    # solution is to read the engine code from a file and inject
    # it into the html each time it is needed.
    #
    # We do the same with msgpack-lite-numpy.js.
    #
    # There might be a jupyter way to ensure that we only get one
    # copy of this code on the page, one that works when the page
    # is saved and then reloaded.
    #

    script_dir     = Path(__file__).parent
    msgpack_source = Path(script_dir / 'web' / 'msgpack-lite-numpy.js').read_text(encoding='utf-8')
    engine         = Path(script_dir / 'web' / 'engine.mjs').read_text(encoding='utf-8')

    # Load the visualiser code.
    mjs = Path(javascript_file).read_text(encoding='utf-8')

    html = ""
    if html_file is not None:
        html = Path(html_file).read_text(encoding='utf-8')

    ssh_ip = None
    if in_vscode() and 'SSH_CONNECTION' in os.environ:
        ssh_connection = os.environ['SSH_CONNECTION']
        ssh_ip = ssh_connection.split(" ")[2]

    ws_port = os.environ.get("CL_SDK_WEBSOCKET_PORT", "1025")

    iframe_content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            width: 100%;
            min-height: 100%;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        }}
        #{unique_id} {{
            width: 100%;
            min-height: 100%;
        }}

        @media (prefers-color-scheme: dark) {{

            body {{
                color: #ddd;
            }}
        }}
    </style>
</head>
<body>
    <div id="{unique_id}">
        {html}
    </div>
    <script type="module">
        {msgpack_source}
        const uniqueId = '{unique_id}';
        const dataStreams = {json.dumps(data_streams or [])};
        const sshIp = {json.dumps(ssh_ip)};
        const websocketPort = {ws_port};
        {mjs}
        {engine}

        // Notify parent of size changes for dynamic iframe height
        function notifyParentOfSize() {{
            const height = document.body.scrollHeight;
            window.parent.postMessage({{ type: 'resize', id: '{unique_id}', height: height }}, '*');
        }}

        // Observe size changes
        const resizeObserver = new ResizeObserver(notifyParentOfSize);
        resizeObserver.observe(document.body);

        // Initial size notification
        notifyParentOfSize();

        // Forward scroll events to parent
        window.addEventListener('wheel', (e) => {{
            window.parent.postMessage({{
                type: 'scroll',
                id: '{unique_id}',
                deltaY: e.deltaY,
                deltaX: e.deltaX,
                deltaMode: e.deltaMode
            }}, '*');
            // Don't preventDefault - let the event bubble naturally
            // The parent will handle scrolling its container
        }}, {{ passive: true }});
    </script>
</body>
</html>
"""

    return f"""
{_generate_iframe(unique_id, iframe_content=iframe_content)}
{_generate_wrapper_script(unique_id, use_sidebar, aspect_ratio)}
"""
