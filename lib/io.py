# lib/io.py
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict

import pandas as pd

# Optional Streamlit caching – if Streamlit isn't running, this degrades gracefully.
try:  # pragma: no cover
    import streamlit as st

    _cache = st.cache_data(show_spinner=False)
except Exception:  # pragma: no cover
    def _cache(func):  # no-op decorator
        return func


def _candidate_data_dirs() -> list[str]:
    """
    Try common repo layouts. First existing directory wins.

    Priority order:
      1) env var LUE_DATA_DIR
      2) ./data
      3) ./Streamlit/data
      4) ../data (useful when pages/ is cwd)
    """
    env = os.getenv("LUE_DATA_DIR")
    candidates = [env] if env else []
    candidates += ["data", os.path.join("Streamlit", "data"), os.path.join("..", "data")]
    return [c for c in candidates if c]


def _resolve_path(name: str) -> str:
    file = name if name.endswith(".parquet") else f"{name}.parquet"
    for root in _candidate_data_dirs():
        path = os.path.abspath(os.path.join(root, file))
        if os.path.exists(path):
            return path
    # Fallback: return first candidate even if it doesn't exist (clear error)
    first = os.path.abspath(os.path.join(_candidate_data_dirs()[0], file))
    return first


@_cache
def load_parquet(name: str) -> pd.DataFrame:
    """
    Load a parquet file from the detected data directory with caching.
    Example: load_parquet("pubs_final") or load_parquet("pubs_final.parquet")
    """
    path = _resolve_path(name)
    return pd.read_parquet(path)


@_cache
def load_all_core() -> Dict[str, pd.DataFrame]:
    """
    Load all core datasets used across pages.
    Keys: pubs, authors, fields, domains, partners, units, topics
    """
    return {
        "pubs": load_parquet("pubs_final"),
        "authors": load_parquet("ul_authors_indicators"),
        "fields": load_parquet("ul_fields_indicators"),
        "domains": load_parquet("ul_domains_indicators"),
        "partners": load_parquet("ul_partners_indicators"),
        "units": load_parquet("ul_units_indicators"),
        "topics": load_parquet("all_topics"),
    }