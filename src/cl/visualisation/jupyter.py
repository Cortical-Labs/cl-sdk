"""
# Visualisation utilities for Jupyter

This module provides utility methods for displaying visualisations in Jupyter notebooks.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from cl import ChannelSet, is_simulator

from .visualisation import create_iframe_visualiser, create_visualiser

# Jupyter's 'HTML' will create a widget that supports arbitrary HTML content,
# and will also execute any JavaScript code included in the HTML content.

def display_visualiser(
    javascript_file: str | Path,
    html_file      : str | Path | None = None,
    data_streams   : list[str]  | None = None,
    use_sidebar    : bool              = True,
    aspect_ratio   : float | None      = None,
):
    """
    Display a custom visualiser in a Jupyter notebook.

    Args:
        javascript_file: Path to the visualiser's JavaScript module file.
        html_file: Optional path to an HTML file to include in the visualiser.
        data_streams: Optional list of data stream names to connect to the visualiser.
        use_sidebar: Whether to enable the sidebar layout for the visualiser, when used in Jupyter notebook/lab environment.
        aspect_ratio: Optional aspect ratio (width / height) for the visualiser display area. If not provided, height is determined from the content.
    """
    from IPython.display import HTML, display
    display(
        HTML(
            create_visualiser(
                javascript_file = javascript_file,
                html_file       = html_file,
                data_streams    = data_streams,
                use_sidebar     = use_sidebar,
                aspect_ratio    = aspect_ratio,
            )
        )
    )

def show_activity(
    mode             : Literal["2d", "3d"]                     = "2d",
    use_sidebar      : bool                                    = True,
    focus_on_channels: int | Sequence[int] | ChannelSet | None = None,
    **kwargs,
):
    """
    Show the activity visualiser in a Jupyter notebook, supporting both 2D and 3D modes.

    Args:
        mode: The visualisation mode, either "2d" or "3d".
        use_sidebar: Whether to enable the sidebar layout for the visualiser, when used in Jupyter notebook/lab.
        focus_on_channels: Channel or list of channels to focus on initially.
        **kwargs: Additional query parameters to pass to the visualiser.
    """

    # TODO: Add proper support
    # Handle focus_on_channels parameter by adding to kwargs
    if focus_on_channels is not None:
        if isinstance(focus_on_channels, int):
            focus_on_channels = [focus_on_channels]
        elif isinstance(focus_on_channels, ChannelSet):
            focus_on_channels = list(focus_on_channels)
        kwargs['focusOnChannels'] = ",".join(str(ch) for ch in focus_on_channels)

    endpoint     = "visualiser"
    query_params = "&".join(f"{key}={value}" for key, value in kwargs.items())

    if is_simulator():
        from cl import Neurons
        iframe_url = f"/{endpoint}?jupyterMode=1&sidebarMode={int(use_sidebar)}&plotMode={mode}&{query_params}"
        neurons    = Neurons._get_instance()
        neurons._start_simulator_services()
    else:
        endpoint   = "vis" if mode == "2d" else "mea"
        iframe_url = f"/{endpoint}/?jupyterMode=1&sidebarMode={int(use_sidebar)}&{query_params}"

    from IPython.display import HTML, display
    display(
        HTML(
            create_iframe_visualiser(
                iframe_url   = iframe_url,
                use_sidebar  = use_sidebar,
                aspect_ratio = 16 / 9,
            )
        )
    )