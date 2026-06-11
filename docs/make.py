"""
This builds the Cortical Labs API documentation using source code from src/cl
and places it in docs/html.
"""
import os

os.environ["PDOC_ALLOW_EXEC"] = "1"  # Allow pdoc to run subprocesses (as cl-sdk does for its data producer etc.)

import argparse
import re
import sys
import ast
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

    parser = argparse.ArgumentParser(description="Build CL SDK documentation")
    parser.add_argument("--base-url", default="", help="URL prefix for hosted docs (e.g., /sdk-docs)")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

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
        favicon            = f"{base_url}/favicon.png",
        logo               = f"{base_url}/images/logo.svg",
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

    def _link_index_to_module(index_md):
        """Allow module declarations in index.md to link to module docs."""

        # Build a dynamic registry of API endpoints by parsing Python files
        api_links = {"cl": "cl.html"}

        def _add_link(current_path):
            # pdoc generates anchors relative to the module, so we strip the 'cl.' prefix
            anchor = current_path[3:] if current_path.startswith("cl.") else current_path
            link   = f"cl.html#{anchor}"

            api_links[current_path] = link

            # Register short name (e.g., 'Spike' -> 'cl.Spike')
            short_name = current_path.split('.')[-1]
            if short_name not in api_links:
                api_links[short_name] = link

            # Register ClassName.member (e.g., 'Spike.samples')
            if current_path.count('.') >= 2:
                class_member = ".".join(current_path.split('.')[-2:])
                if class_member not in api_links:
                    api_links[class_member] = link

        for py_file in module_path.rglob("*.py"):
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"))
            except Exception:
                continue

            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    if not node.name.startswith("_"):
                        class_path = f"cl.{node.name}"
                        _add_link(class_path)
                        for child in node.body:
                            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not child.name.startswith("_"):
                                _add_link(f"{class_path}.{child.name}")
                            elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                                if not child.target.id.startswith("_"):
                                    _add_link(f"{class_path}.{child.target.id}")
                            elif isinstance(child, ast.Assign):
                                for target in child.targets:
                                    if isinstance(target, ast.Name) and not target.id.startswith("_"):
                                        _add_link(f"{class_path}.{target.id}")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not node.name.startswith("_"):
                        _add_link(f"cl.{node.name}")
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    if not node.target.id.startswith("_"):
                        _add_link(f"cl.{node.target.id}")
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and not target.id.startswith("_"):
                            _add_link(f"cl.{target.id}")

        # Register subpackage links (e.g. cl.sim → cl/sim.html)
        for init_file in sorted(module_path.rglob("__init__.py")):
            if init_file.parent == module_path:
                continue  # skip the root cl/__init__.py
            rel_parts   = init_file.parent.relative_to(module_path.parent).parts
            module_name = ".".join(rel_parts)
            html_path   = "/".join(rel_parts) + ".html"
            api_links[module_name] = html_path
            short_name = rel_parts[-1]
            if short_name not in api_links:
                api_links[short_name] = html_path

        def _linkify_backticks(match):
            text   = match.group(1)
            lookup = text.replace('()', '')

            # 1. Exact match (e.g., 'cl', 'cl.Loop', 'Spike')
            if lookup in api_links:
                return f"[`{text}`]({api_links[lookup]})"

            # 2. Fallback for instance chains (e.g., 'tick.analysis.spikes' -> 'spikes')
            parts = lookup.split('.')
            for i in range(1, len(parts)):
                suffix = ".".join(parts[i:])
                if suffix in api_links:
                    return f"[`{text}`]({api_links[suffix]})"

            return match.group(0)

        index_md = re.sub(r"`([a-zA-Z0-9_.]+(?:\(\))?)`", _linkify_backticks, index_md)

        return index_md

    index_md = _link_index_to_module(index_md)

    pdoc.render.env.globals["preamble"] = index_md
    pdoc.render.env.globals["build_module_tree"] = _build_module_tree
    pdoc.render.env.globals["base_url"] = base_url

    pdoc.pdoc(module_path, output_directory=output_path)

    # Strip noisy PydanticUndefined default values from rendered HTML
    pattern = re.compile(
        r"\s*=\s*<span\s+class=\"default_value\">\s*PydanticUndefined\s*</span>",
        re.DOTALL,
    )

    for html_file in output_path.rglob("*.html"):
        html_text = html_file.read_text(encoding="utf-8")
        cleaned_html = pattern.sub("", html_text)
        if base_url:
            cleaned_html = cleaned_html.replace('url("/images/', f'url("{base_url}/images/')
        if cleaned_html != html_text:
            html_file.write_text(cleaned_html, encoding="utf-8")

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