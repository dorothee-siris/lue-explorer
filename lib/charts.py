from __future__ import annotations

import pandas as pd
import altair as alt

from lib.constants import DOMAIN_COLORS
from lib.transforms import map_field_to_domain, darken, field_order

alt.data_transformers.disable_max_rows()


def _color_for_row(domain: str, in_lue: bool) -> str:
    base = DOMAIN_COLORS.get(domain, DOMAIN_COLORS["Other"])
    return darken(base, 0.7 if in_lue else 1.0)


def field_mix_bars(
    df: pd.DataFrame,
    value_col: str = "count",
    percent: bool = False,
    height_per_field: int = 24,
    xmin: float | None = 0.0,
    xmax: float | None = None,
    enforce_order_from: list[str] | None = None,
    show_y_labels: bool = True,
):
    """
    Horizontal stacked bars of field distribution, colored by domain.
    Darker shade shows the ISITE segment.
    Expects columns: field, count, in_lue_count.
    """
    if df.empty:
        return alt.Chart(pd.DataFrame({"field": [], "value": []})).mark_bar()

    d = df.copy()
    d["domain"] = d["field"].map(map_field_to_domain)
    d["in_lue_count"] = pd.to_numeric(d["in_lue_count"], errors="coerce").fillna(0).astype(int)
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce").fillna(0)

    # Build segments with friendly names
    d["Not ISITE"] = d[value_col] - d["in_lue_count"]
    d["ISITE"] = d["in_lue_count"]
    d = d.melt(
        id_vars=["field", "domain"],
        value_vars=["Not ISITE", "ISITE"],
        var_name="segment",
        value_name="value",
    )
    d["is_lue"] = d["segment"].eq("ISITE")

    if percent:
        totals = d.groupby("field")["value"].transform(lambda s: s.sum() if s.sum() else 1)
        d["value"] = d["value"] / totals

    d["color"] = d.apply(lambda r: _color_for_row(r["domain"], bool(r["is_lue"])), axis=1)

    # Fixed order by domain then alphabetical – include fields not present for this lab
    present_fields = d["field"].dropna().unique().tolist()
    order = field_order(enforce_order_from or present_fields)

    # Dynamic height so every label is visible
    chart_height = max(180, int(len(order) * height_per_field))

    # X axis and scale
    x_axis = (alt.Axis(format="%", tickCount=5) if percent else alt.Axis())
    if percent:
        x_scale = alt.Scale(domain=[0, 1])
    else:
        # Shared domain across charts if xmax provided
        lo = 0 if xmin is None else xmin
        x_scale = alt.Scale(domain=[lo, xmax] if xmax is not None else None)

    y_axis = alt.Axis(labelLimit=2000, labels=show_y_labels, ticks=show_y_labels)

    tooltip = [
        alt.Tooltip("field:N", title="Field"),
        alt.Tooltip("domain:N", title="Domain"),
        alt.Tooltip("segment:N", title="Segment"),
        alt.Tooltip("value:Q", title=("Share" if percent else "Count"), format=(".0%" if percent else ",")),
    ]

    chart = (
        alt.Chart(d)
        .mark_bar()
        .encode(
            y=alt.Y("field:N", sort=order, title=None, axis=y_axis),
            x=alt.X("value:Q", title=("Share of works" if percent else "Works"), axis=x_axis, scale=x_scale),
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=tooltip,
        )
        .properties(height=chart_height)
    )
    return chart
