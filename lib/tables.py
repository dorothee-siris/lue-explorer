# lib/tables.py
from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd

# Functions here are safe to import without Streamlit.
# We only touch st.* inside functions so non-Streamlit contexts won't import-fail.

def progressify(df: pd.DataFrame, cols: Iterable[str]) -> Dict[str, "object"]:
    """
    Build a Streamlit column_config dict with ProgressColumn for the given columns.
    Values are assumed to be 0..100 (percent). Max is set to column max (or 100 if empty).
    """
    try:
        import streamlit as st  # lazy import for safety
    except Exception:  # pragma: no cover
        return {}

    cfg = {}
    for c in cols:
        if c not in df.columns:
            continue
        try:
            max_val = float(pd.to_numeric(df[c], errors="coerce").max())
            if not (max_val > 0):
                max_val = 100.0
        except Exception:
            max_val = 100.0
        cfg[c] = st.column_config.ProgressColumn(
            c, format="%0.1f%%", min_value=0.0, max_value=max_val
        )
    return cfg


def show_table(
    df: pd.DataFrame,
    height: int = 420,
    use_container_width: bool = True,
    hide_index: bool = True,
    column_config: dict | None = None,
):
    """Thin wrapper over st.dataframe with sensible defaults."""
    import streamlit as st  # local import
    st.dataframe(
        df,
        use_container_width=use_container_width,
        height=height,
        hide_index=hide_index,
        column_config=column_config or {},
    )
