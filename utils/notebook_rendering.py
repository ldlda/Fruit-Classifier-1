import nbformat
import streamlit as st
from nbconvert import HTMLExporter

__all__ = ["render_notebook_to_html", "render_notebook_to_pdf"]


@st.cache_data(show_spinner=True)
def render_notebook_to_html(path: str, _mtime: float, theme: str) -> str:
    """Render a .ipynb to HTML. Cached by file modification time."""
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    exporter = HTMLExporter()
    exporter.exclude_input = False
    exporter.exclude_output = False
    # Align notebook theme with the Streamlit page theme; include theme in cache key above
    if hasattr(exporter, "theme"):
        exporter.theme = "dark" if theme == "dark" else "light"
    body, _resources = exporter.from_notebook_node(nb)
    return body


def render_notebook_to_pdf(): ...
