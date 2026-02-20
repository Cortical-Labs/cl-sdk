
"""
This patches pdoc to unwrap Pydantic type Annotations to remove documentation
clutter and to show just the intended Type
"""
import inspect
import warnings
from typing import Annotated, get_args, get_origin

import pdoc._compat
import pdoc._pydantic
import pdoc.doc

# Store original formatannotation
_original_formatannotation = inspect.formatannotation

def _unwrap_annotated(annotation):
    """Recursively unwrap Annotated types to get the base type."""
    while get_origin(annotation) is Annotated:
        args = get_args(annotation)
        if args:
            annotation = args[0]
        else:
            break
    return annotation

def _simplified_formatannotation(annotation, base_module=None) -> str:
    """
    Custom formatannotation that simplifies Annotated types.

    For Annotated[BaseType, metadata...], this returns just the formatted BaseType,
    avoiding the verbose FieldInfo repr from Pydantic.
    """
    # Unwrap Annotated types to get the base type
    unwrapped = _unwrap_annotated(annotation)
    return _original_formatannotation(unwrapped, base_module)

# Monkey-patch pdoc's formatannotation in all relevant modules
pdoc._compat.formatannotation = _simplified_formatannotation
pdoc.doc.formatannotation = _simplified_formatannotation

# Patch Variable.annotation_str to properly handle TypeAliasType inside Annotated
_original_annotation_str_func = pdoc.doc.Variable.annotation_str.func

def _patched_annotation_str(self) -> str:
    """The variable's type annotation as a pretty-printed str, with Annotated unwrapping."""
    if self.annotation is not pdoc.doc.empty:
        # Unwrap Annotated to get the inner type for TypeAliasType detection
        unwrapped = _unwrap_annotated(self.annotation)
        formatted = _simplified_formatannotation(self.annotation)

        # Check if the unwrapped type is a TypeAliasType (needs module prefix for linking)
        if isinstance(unwrapped, pdoc.doc.TypeAliasType):
            formatted = f"{unwrapped.__module__}.{formatted}"

        return f": {pdoc.doc._remove_collections_abc(formatted)}"
    else:
        return ""

# Replace the cached_property with a new one using our patched function
_new_annotation_str = pdoc.doc.cached_property(_patched_annotation_str)
_new_annotation_str.__set_name__(pdoc.doc.Variable, "annotation_str")
pdoc.doc.Variable.annotation_str = _new_annotation_str

# Patch get_field_docstring to return None if a proper docstring comment exists
# This makes docstring comments take priority over Field(description=...)
_original_get_field_docstring = pdoc._pydantic.get_field_docstring

def _patched_get_field_docstring(parent, field_name):
    """
    Return None to let docstring comments take priority over Field descriptions.
    The original pydantic Field description is only used as a fallback.
    """
    # We return None here so pdoc falls through to check _var_docstrings first
    # The actual Field description will be used elsewhere if needed
    return None

pdoc._pydantic.get_field_docstring = _patched_get_field_docstring

# Patch Variable.default_value_str to simplify Annotated types for type alias assignments
_original_default_value_str_func = pdoc.doc.Variable.default_value_str.func

def _patched_default_value_str(self) -> str:
    """The variable's default value as a pretty-printed str, with Annotated unwrapping."""
    if self.default_value is pdoc.doc.empty:
        return ""

    # Handle TypeAliasType (Python 3.12+ type statement)
    if isinstance(self.default_value, pdoc.doc.TypeAliasType):
        formatted = _simplified_formatannotation(self.default_value.__value__)
        return pdoc.doc._remove_collections_abc(formatted)

    # Handle TypeAlias annotation
    from typing import TypeAlias
    if self.annotation == TypeAlias:
        formatted = _simplified_formatannotation(self.default_value)
        return pdoc.doc._remove_collections_abc(formatted)

    # Handle Annotated types used as type alias (assignment style)
    if get_origin(self.default_value) is Annotated:
        formatted = _simplified_formatannotation(self.default_value)
        return pdoc.doc._remove_collections_abc(formatted)

    # Fall back to original implementation
    return _original_default_value_str_func(self)

# Replace the cached_property with a new one using our patched function
_new_default_value_str = pdoc.doc.cached_property(_patched_default_value_str)
_new_default_value_str.__set_name__(pdoc.doc.Variable, "default_value_str")
pdoc.doc.Variable.default_value_str = _new_default_value_str

# Patch Namespace.members to fetch docstrings for re-exported symbols from their source modules
import pdoc.doc_ast as doc_ast

# Patch get_source to handle classes with modified __module__ (e.g., from __init__.py reassignment)
_original_get_source = doc_ast._get_source.__wrapped__

def _patched_get_source(obj) -> str:
    """
    Enhanced get_source that handles classes with modified __module__.

    When a class has __module__ reassigned (e.g., in __init__.py for nicer import paths),
    inspect.getsource fails because it looks in the wrong file. This patch tries to
    find the source by looking at the class's actual definition location.
    """
    try:
        return inspect.getsource(obj)
    except Exception:
        pass

    # For classes, try to find the source file from __qualname__ and module hierarchy
    if isinstance(obj, type):
        try:
            module_name = getattr(obj, "__module__", None)
            qualname = getattr(obj, "__qualname__", None)

            if module_name and qualname and "." not in qualname:
                # Try module.model (common pattern: package/__init__.py re-exports from package/model.py)
                import importlib
                for suffix in ["model", "models", "_model", "_models", qualname.lower()]:
                    try:
                        submodule_name = f"{module_name}.{suffix}"
                        submodule = importlib.import_module(submodule_name)
                        # Check if the class exists in this submodule
                        if hasattr(submodule, qualname):
                            cls_in_submodule = getattr(submodule, qualname)
                            if cls_in_submodule is obj:
                                # Found it - get source from the submodule's file
                                submodule_file = inspect.getsourcefile(submodule)
                                if submodule_file:
                                    source_content = Path(submodule_file).read_text()
                                    # Find the class definition in that file
                                    import ast
                                    tree = ast.parse(source_content)
                                    for node in ast.walk(tree):
                                        if isinstance(node, ast.ClassDef) and node.name == qualname:
                                            lines = source_content.splitlines(keepends=True)
                                            start_line = node.lineno - 1
                                            end_line = node.end_lineno if hasattr(node, 'end_lineno') else len(lines)
                                            return "".join(lines[start_line:end_line])
                    except (ImportError, OSError):
                        continue
        except Exception:
            pass

    return ""

# Replace the cached version
doc_ast._get_source = doc_ast.cache(_patched_get_source)
doc_ast.get_source = lambda obj: doc_ast._get_source(obj)

_original_members_func = pdoc.doc.Namespace.members.func

def _patched_members(self) -> dict:
    """Enhanced members property that fetches docstrings for re-exported symbols."""
    members = _original_members_func(self)

    # For each variable that was taken from another module, try to get its docstring
    for name, doc_obj in members.items():
        if isinstance(doc_obj, pdoc.doc.Variable) and not doc_obj.docstring:
            src_mod, src_qual = doc_obj.taken_from
            # If taken from a different module, look up the docstring there
            if src_mod and src_mod != self.modulename:
                try:
                    # Import the source module
                    import importlib
                    source_module = importlib.import_module(src_mod)
                    # Parse its AST to get var_docstrings
                    ast_info = doc_ast.walk_tree(source_module)
                    # The source qualname might be "module.VarName", extract just the var name
                    var_name = src_qual.rsplit(".", 1)[-1] if "." in src_qual else src_qual
                    if var_name in ast_info.var_docstrings:
                        doc_obj.docstring = ast_info.var_docstrings[var_name]
                except Exception:
                    pass  # If anything fails, just keep the empty docstring

    return members

# Replace the cached_property
_new_members = pdoc.doc.cached_property(_patched_members)
_new_members.__set_name__(pdoc.doc.Namespace, "members")
pdoc.doc.Namespace.members = _new_members

# Suppress pdoc warnings about Pydantic model field origins (harmless noise)
warnings.filterwarnings("ignore", message="Cannot determine where .* is taken from")