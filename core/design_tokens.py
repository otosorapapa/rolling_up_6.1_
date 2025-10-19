"""Utilities for loading and working with design tokens defined in YAML."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml


_TOKENS_PATH = Path(__file__).resolve().parent.parent / "design_tokens.yaml"


@lru_cache(maxsize=1)
def load_design_tokens() -> Dict[str, Any]:
    """Return the parsed design tokens defined in :mod:`design_tokens.yaml`."""

    if not _TOKENS_PATH.exists():
        raise FileNotFoundError(f"Design token file not found: {_TOKENS_PATH}")
    with _TOKENS_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def _resolve_section(path: Sequence[str]) -> Any:
    tokens = load_design_tokens()
    node: Any = tokens
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise KeyError("/".join(path))
        node = node[key]
    return node


def _normalise_hex(color: str) -> str:
    color = color.strip()
    if not color.startswith("#"):
        raise ValueError(f"Expected hex colour (got {color!r})")
    if len(color) == 7:
        return color.upper()
    if len(color) == 4:
        return "#" + "".join(ch * 2 for ch in color[1:]).upper()
    raise ValueError(f"Unsupported colour format: {color!r}")


def get_color(name: str, variant: str | None = None, default: str | None = None) -> str:
    """Return the hexadecimal colour value for ``name``.

    ``variant`` can be used to fetch nested values such as ``accent.soft``.
    """

    entry = _resolve_section(["colors", name])
    if isinstance(entry, dict):
        if variant and variant in entry:
            return _normalise_hex(entry[variant])
        if "value" in entry:
            return _normalise_hex(entry["value"])
        if default is not None:
            return default
        raise KeyError(f"Variant {variant!r} not found for colour {name!r}")
    if isinstance(entry, str):
        return _normalise_hex(entry)
    if default is not None:
        return default
    raise TypeError(f"Unsupported colour entry for {name!r}: {entry!r}")


def get_color_rgb(name: str, variant: str | None = None) -> str:
    """Return an ``r,g,b`` string for the given colour token."""

    hex_color = get_color(name, variant)
    r, g, b = hex_to_rgb_tuple(hex_color)
    return f"{r},{g},{b}"


def get_font_stack(section: str) -> str:
    """Return a CSS font stack for the given typography ``section``."""

    entry = _resolve_section(["typography", section])
    families = entry.get("font_family") if isinstance(entry, dict) else None
    if not families:
        raise KeyError(f"Font family not defined for section {section!r}")

    def wrap(font: str) -> str:
        font = font.strip()
        if font.lower() in {"serif", "sans-serif", "monospace"}:
            return font
        if " " in font or "-" in font:
            return f"'{font}'"
        return font

    return ", ".join(wrap(font) for font in families)


def get_typography(section: str) -> Dict[str, Any]:
    """Return the typography tokens for ``section``."""

    entry = _resolve_section(["typography", section])
    if not isinstance(entry, dict):
        raise TypeError(f"Typography section {section!r} is not a mapping")
    return entry


def get_layout_token(*path: str) -> Any:
    """Return a layout token from the ``layout`` section."""

    return _resolve_section(["layout", *path])


def hex_to_rgb_tuple(color: str) -> tuple[int, int, int]:
    """Convert a hex colour to an ``(r, g, b)`` tuple."""

    color = _normalise_hex(color)
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    return r, g, b


def rgba(color: str, alpha: float) -> str:
    """Return an ``rgba()`` string for ``color`` with the provided ``alpha``."""

    alpha = max(0.0, min(1.0, float(alpha)))
    r, g, b = hex_to_rgb_tuple(color)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def mix(color: str, other: str, ratio: float) -> str:
    """Mix ``color`` with ``other`` using ``ratio`` (0-1) and return hex."""

    ratio = max(0.0, min(1.0, float(ratio)))
    r1, g1, b1 = hex_to_rgb_tuple(color)
    r2, g2, b2 = hex_to_rgb_tuple(other)
    r = round(r1 * (1.0 - ratio) + r2 * ratio)
    g = round(g1 * (1.0 - ratio) + g2 * ratio)
    b = round(b1 * (1.0 - ratio) + b2 * ratio)
    return f"#{r:02X}{g:02X}{b:02X}"


def lighten(color: str, ratio: float) -> str:
    """Return ``color`` mixed with white by ``ratio``."""

    return mix(color, "#FFFFFF", ratio)


def darken(color: str, ratio: float) -> str:
    """Return ``color`` mixed with black by ``ratio``."""

    return mix(color, "#000000", ratio)


def get_plotly_palette() -> List[str]:
    """Return the default qualitative palette for Plotly figures."""

    return [
        get_color("accent"),
        get_color("primary"),
        get_color("success"),
        get_color("secondary"),
        get_color("warning"),
        get_color("error"),
    ]


def spacing_scale() -> List[int]:
    """Return the spacing scale defined under ``layout.spacing``."""

    spacing = _resolve_section(["layout", "spacing"])
    scale = spacing.get("scale") if isinstance(spacing, dict) else None
    if not scale:
        return []
    return [int(val) for val in scale]

