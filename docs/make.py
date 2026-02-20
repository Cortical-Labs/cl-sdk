"""
This builds the Cortical Labs API documentation using source code from src/cl
and places it in docs/html.
"""
import sys
import shutil
from pathlib import Path
from typing import Mapping, cast

import pdoc
from pdoc.render_helpers import edit_url as _edit_url

# Add cl-sdk source to sys.path FIRST to ensure we document our source, not installed packages
here = Path(__file__).parent
_src_path = str((here / ".." / "src").resolve())
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from ._pdoc_pydantic_patch import *

if __name__ == "__main__":

    module_path    = here / ".." / "src" / "cl"
    output_path    = here / "html"
    assets_to_copy = \
        [
            "favicon.png",
            "doc-handler.js",
            "images"
        ]

    # Clean output path
    if output_path.exists():
        shutil.rmtree(output_path)

    # Render docs
    pdoc.render.configure(
        docformat          = "google",
        favicon            = "/favicon.png",
        logo               = "/images/logo.svg",
        logo_link          = "https://corticallabs.com",
        math               = True,
        show_source        = False,
        template_directory = here
        )

    # Force root_module_name=None so index uses full nav (not a redirect)
    # and module pages get the "← Module Index" link back
    def _patched_html_index(all_modules):
        return pdoc.render.env.get_template("index.html.jinja2").render(
            all_modules      = all_modules,
            root_module_name = None,
        )
    pdoc.render.html_index = _patched_html_index

    def _patched_html_module(module, all_modules, mtime=None):
        return pdoc.render.env.get_template("module.html.jinja2").render(
            module=module,
            all_modules=all_modules,
            root_module_name=None,
            edit_url=_edit_url(
                module.modulename,
                module.is_package,
                cast("Mapping[str, str]", pdoc.render.env.globals["edit_url_map"]),
            ),
            mtime=mtime,
        )
    pdoc.render.html_module = _patched_html_module

    def _build_module_tree(all_modules: dict) -> dict:
        """Convert flat module name list into nested dict for tree rendering."""
        tree = {}
        for name in sorted(all_modules.keys()):
            node = tree
            for part in name.split("."):
                node = node.setdefault(part, {})
        return tree

    index_md = (here / "index.md").read_text()
    pdoc.render.env.globals["preamble"] = index_md
    pdoc.render.env.globals["build_module_tree"] = _build_module_tree

    pdoc.pdoc(module_path, output_directory=output_path)

    # Post-process: build all_modules from generated files, render our custom index
    all_modules = {
        ".".join(f.with_suffix("").relative_to(output_path).parts): None
        for f in sorted(output_path.rglob("*.html"))
        if f.name != "index.html"
    }
    index_html = pdoc.render.env.get_template("index.html.jinja2").render(
        all_modules      = all_modules,
        root_module_name = None,
    )
    (output_path / "index.html").write_text(index_html)

    # Copy assets
    for asset_path in assets_to_copy:
        src_path = here / asset_path
        dst_path = output_path / asset_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if not src_path.exists():
            continue
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy(src_path, dst_path)