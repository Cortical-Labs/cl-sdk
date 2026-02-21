"""
# Cortical Labs Visualisation Module

This module provides tools for visualising data from the system, when used within a Jupyter notebook. It includes functions for
creating interactive visualisations that can be embedded in Jupyter notebooks. If used within a Jupyter notebook or Jupyter Lab,
the visualisations can be rendered in "sidebar" mode, which displays each visualisation on the side of the notebook instead of
inline.
"""

from . import jupyter
from .visualisation import create_iframe_visualiser, create_visualiser

__all__ = [
    "create_iframe_visualiser",
    "create_visualiser",
    "jupyter",
]
