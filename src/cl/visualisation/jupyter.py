"""
# Visualisation utilities for Jupyter

This module provides utility methods for displaying visualisations in Jupyter notebooks.
"""

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from IPython.display import HTML, display

from cl import ChannelSet

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

    print("Warning: show_activity() is currently not supported in cl-sdk", file=sys.stderr)
