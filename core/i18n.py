"""Internationalization helpers for the Streamlit app."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import yaml

DEFAULT_LANGUAGE = "ja"
_TRANSLATION_FILE = Path(__file__).with_name("translations.yaml")


@lru_cache()
def _load_translations() -> Dict[str, Any]:
    """Load the translation dictionary from the YAML file."""
    if not _TRANSLATION_FILE.exists():
        return {}
    with _TRANSLATION_FILE.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        return {}
    return data


def _resolve_key(key: str) -> Any:
    """Resolve a dotted key path inside the translation dictionary."""
    data = _load_translations()
    node: Any = data
    for part in key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def init_language(default: str = DEFAULT_LANGUAGE) -> str:
    """Ensure the session state has a language value."""
    available = get_available_languages()
    if default not in available and available:
        default = available[0]
    if "language" not in st.session_state:
        st.session_state["language"] = default
    return st.session_state["language"]


def get_current_language() -> str:
    """Return the current language stored in the session state."""
    lang = st.session_state.get("language", DEFAULT_LANGUAGE)
    available = get_available_languages()
    if available and lang not in available:
        return available[0]
    return lang


def get_available_languages() -> List[str]:
    """Return the supported language codes defined in the translation file."""
    data = _load_translations()
    languages = data.get("languages")
    if isinstance(languages, list):
        return [str(code) for code in languages if code]
    if isinstance(languages, dict):
        return [str(code) for code in languages.keys()]
    language_names = data.get("language_names")
    if isinstance(language_names, dict):
        return [str(code) for code in language_names.keys()]
    return [DEFAULT_LANGUAGE]


def translate(key: str, *, language: str | None = None, default: str | None = None) -> str:
    """Fetch a translated string.

    Parameters
    ----------
    key:
        Dotted path identifying the entry in the translation dictionary.
    language:
        Override language code. When ``None`` the session state's language is used.
    default:
        Fallback text when the key or translation is missing. If omitted, the
        key itself is returned.
    """

    lang = language or get_current_language()
    node = _resolve_key(key)
    if node is None:
        return default if default is not None else key
    if isinstance(node, dict):
        if lang in node and node[lang] is not None:
            return str(node[lang])
        if DEFAULT_LANGUAGE in node and node[DEFAULT_LANGUAGE] is not None:
            return str(node[DEFAULT_LANGUAGE])
        for value in node.values():
            if isinstance(value, str):
                return value
        return default if default is not None else key
    return str(node)


def language_name(code: str, *, language: str | None = None) -> str:
    """Return the localized display name for a language code."""
    return translate(f"language_names.{code}", language=language, default=code)


# Shorthand alias used in the app.
t = translate
