# lib/charts.py
from __future__ import annotations

from math import ceil
from typing import Dict, List, Optional

import altair as alt
import pandas as pd

from .taxonomy import get_domain_color

# Streamlit renders Altair with container_width, so width/height are hints.
_LEFT_GUTTER_PX = 80  # space for count labels (left of bars)
_BAR_WIDTH_PX = 720   # default bar area width


def _dynamic_height(n_items: int) -> int:
    """Compute chart pixel height so all labels are always visible."""
    return int(ceil(48 * max(n_items, 1) + 80))


def _apply_domain_colors(df: pd.DataFrame, domain_col: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if domain_col and domain_col in out.columns:
        out["__color__"] = out[domain_col].map(get_domain_color)
    else:
        out["__color__"] = "#7f7f7f"
    return out


def _order_categorical(df: pd.DataFrame, label_col: str, order: Optional[List[str]]) -> pd.DataFrame:
    if not order:
        return df
    out = df.copy()
    # Keep only items that exist; preserve user order
    ordered = [x for x in order if x in out[label_col].astype(str).tolist()]
    out[label_col] = pd.Categorical(out[label_col], categories=ordered, ordered=True)
    out = out.sort_values(label_col)
    return out


def plot_bar_with_counts(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    count_col: str,
    domain_col: Optional[str],
    title: str,
    order: Optional[List[str]] = None,
) -> alt.Chart:
    """
    Horizontal bar chart with a left text gutter for counts.
    - label_col: y-axis labels (all shown)
    - value_col: numeric value (e.g., %)
    - count_col: integer counts shown in left gutter
    - domain_col: used to color bars with domain palette (via get_domain_color)
    - order: explicit label order (kept even for labels with value 0)
    """
    base = df.copy()
    # Normalize columns to strings/numerics
    base[label_col] = base[label_col].astype(str)
    base[value_col] = pd.to_numeric(base[value_col], errors="coerce").fillna(0.0)
    base[count_col] = pd.to_numeric(base[count_col], errors="coerce").fillna(0)

    base = _order_categorical(base, label_col, order)
    base = _apply_domain_colors(base, domain_col)
    height = _dynamic_height(len(base))

    # Left gutter counts (right-aligned)
    txt = (
        alt.Chart(base, height=height, width=_LEFT_GUTTER_PX)
        .mark_text(align="right", dx=-6)
        .encode(
            y=alt.Y(f"{label_col}:N", sort=None, title=""),
            text=alt.Text(f"{count_col}:Q", format=",.0f"),
            color=alt.value("#444"),
            tooltip=[count_col],
        )
    )

    # Bars
    bars = (
        alt.Chart(base, height=height, width=_BAR_WIDTH_PX)
        .mark_bar()
        .encode(
            y=alt.Y(f"{label_col}:N", sort=None, title=""),
            x=alt.X(f"{value_col}:Q", title=title),
            color=alt.Color("__color__:N", scale=None, legend=None),
            tooltip=list(base.columns),
        )
    )

    return txt | bars  # horizontal concat


def plot_whisker(
    df: pd.DataFrame,
    label_col: str,
    qcols: Dict[str, str],
    domain_col: Optional[str],
    title: str,
    order: Optional[List[str]] = None,
) -> alt.Chart:
    """
    Whisker plot with min/Q1/median/Q3/max.
    qcols keys: {'min','q1','q2','q3','max'} -> column names
    """
    base = df.copy()
    base[label_col] = base[label_col].astype(str)
    for key in ("min", "q1", "q2", "q3", "max"):
        col = qcols[key]
        base[col] = pd.to_numeric(base[col], errors="coerce")
    base = _order_categorical(base, label_col, order)
    base = _apply_domain_colors(base, domain_col)
    height = _dynamic_height(len(base))

    # No log scale, using linear scale by default
    xscale = alt.Scale()

    # Whisker (min..max)
    rules = (
        alt.Chart(base, height=height, width=_BAR_WIDTH_PX)
        .mark_rule()
        .encode(
            y=alt.Y(f"{label_col}:N", sort=None, title=""),
            x=alt.X(f"{qcols['min']}:Q", scale=xscale, title=title),
            x2=f"{qcols['max']}:Q",
            color=alt.Color("__color__:N", scale=None, legend=None),
            tooltip = [alt.Tooltip(c, type="nominal") for c in base.columns],
        )
    )

    # IQR box (Q1..Q3)
    boxes = (
        alt.Chart(base, height=height, width=_BAR_WIDTH_PX)
        .mark_bar(opacity=0.35)
        .encode(
            y=alt.Y(f"{label_col}:N", sort=None, title=""),
            x=alt.X(f"{qcols['q1']}:Q", scale=xscale, title=title),
            x2=f"{qcols['q3']}:Q",
            color=alt.Color("__color__:N", scale=None, legend=None),
        )
    )

    # Median tick (Q2)
    median = (
        alt.Chart(base, height=height, width=_BAR_WIDTH_PX)
        .mark_tick(size=18, thickness=2)
        .encode(
            y=alt.Y(f"{label_col}:N", sort=None, title=""),
            x=f"{qcols['q2']}:Q",
            color=alt.Color("__color__:N", scale=None, legend=None),
        )
    )

    return rules + boxes + median

