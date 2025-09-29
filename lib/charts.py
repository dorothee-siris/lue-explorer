from __future__ import annotations
import pandas as pd
import altair as alt

from lib.constants import DOMAIN_COLORS
from lib.transforms import map_field_to_domain, darken, field_order

# Streamlit sometimes limits rows in Altair; disable that.
alt.data_transformers.disable_max_rows()


def _color_for_row(domain: str, in_lue: bool) -> str:
    base = DOMAIN_COLORS.get(domain, DOMAIN_COLORS["Other"])
    return darken(base, 0.7 if in_lue else 1.0)


def field_mix_bars(
    df: pd.DataFrame,
    value_col: str = "count",
    percent: bool = False,
    height_per_field: int = 18,
    xmin: float | None = 0.0,
    xmax: float | None = None,
    enforce_order_from: list[str] | None = None,   # pass FULL field catalogue here
    show_y_labels: bool = True,
    width: int | None = None,                      # <-- accept width
):
    """
    Horizontal stacked bars of field distribution by domain.
    Segments: "ISITE" (darker) and "Not ISITE".
    Ensures EVERY field in `enforce_order_from` appears (zeros if absent).
    """
    if df is None or df.empty:
        # Build an empty chart with the right encoding so Streamlit won't crash.
        return alt.Chart(pd.DataFrame({"field": [], "value": []})).mark_bar()

    # ---- ensure full field list + zeros for missing fields ----
    fields_full = enforce_order_from or sorted(df["field"].dropna().unique().tolist())
    base = pd.DataFrame({"field": fields_full})
    base["domain"] = base["field"].map(map_field_to_domain)

    d = df.copy()
    d["in_lue_count"] = pd.to_numeric(d.get("in_lue_count", 0), errors="coerce").fillna(0).astype(int)
    d[value_col] = pd.to_numeric(d.get(value_col, 0), errors="coerce").fillna(0)

    d = base.merge(d[["field", value_col, "in_lue_count"]], on="field", how="left")
    d[[value_col, "in_lue_count"]] = d[[value_col, "in_lue_count"]].fillna(0)

    # friendly segment names
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

    # fixed order (domain buckets -> A→Z)
    order = field_order(fields_full)
    chart_height = max(220, int(len(order) * height_per_field))

    # X axis & scale
    x_axis = alt.Axis(format="%", tickCount=5) if percent else alt.Axis()
    if percent:
        x_scale = alt.Scale(domain=[0, 1])
    else:
        lo = 0 if xmin is None else xmin
        x_scale = alt.Scale(domain=[lo, xmax] if (xmax is not None and xmax > 0) else None)

    # Force the full y-domain so labels always show (even when all zeros)
    y_axis = alt.Axis(labels=show_y_labels, ticks=False, labelFontSize=11, labelPadding=2, labelLimit=9999)
    y_scale = alt.Scale(domain=order)

    # Slimmer margins to reduce wasted space; fixed left padding keeps the plotting
    # width consistent between the two side-by-side charts.
    padding = {"left": 80, "right": 6, "top": 2, "bottom": 4}

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
            y=alt.Y("field:N", scale=y_scale, title=None, axis=y_axis),
            x=alt.X("value:Q", title=("Share of works" if percent else "Works"), axis=x_axis, scale=x_scale),
            color=alt.Color("color:N", legend=None, scale=None),
            tooltip=tooltip,
        )
        .properties(height=chart_height, padding=padding)
    )
    if width is not None:
        chart = chart.properties(width=width)

    return chart

def simple_field_bars(
    df: pd.DataFrame,
    value_col: str = "count",
    percent: bool = False,
    enforce_order_from: list[str] | None = None,
    show_counts: bool = True,
    width: int | None = 560,
    height_per_field: int = 18,
):
    """
    One-segment horizontal bars by field (colored by domain), optional % mode.
    Prints the integer count next to the y-label (volume only).
    Ensures every field in `enforce_order_from` is shown (zeros if missing).
    """
    if df is None:
        df = pd.DataFrame(columns=["field", value_col])

    d = df.copy()
    d["field"] = d.get("field")
    d[value_col] = pd.to_numeric(d.get(value_col, 0), errors="coerce").fillna(0)

    # Full field list and domain mapping
    fields_full = enforce_order_from or sorted(d["field"].dropna().unique().tolist())
    base = pd.DataFrame({"field": fields_full})
    base["domain"] = base["field"].map(map_field_to_domain)

    # merge so missing fields exist with zeros
    d = base.merge(d[["field", value_col]], on="field", how="left")
    d[value_col] = d[value_col].fillna(0)

    # compute value / percent
    if percent:
        total = float(d[value_col].sum() or 1.0)
        d["value"] = d[value_col] / total
    else:
        d["value"] = d[value_col]

    # fixed order + size
    order = field_order(fields_full)
    h = max(220, int(len(order) * height_per_field))

    # axes
    x_axis = alt.Axis(format="%", tickCount=5) if percent else alt.Axis()
    x_scale = alt.Scale(domain=[0, 1]) if percent else alt.Undefined
    y_axis = alt.Axis(labelLimit=9999, labelPadding=2, labelFontSize=11)
    y_scale = alt.Scale(domain=order)  # force all labels

    base_ch = alt.Chart(d)
    bar = base_ch.mark_bar().encode(
        y=alt.Y("field:N", scale=y_scale, title=None, axis=y_axis),
        x=alt.X("value:Q", title=("Share of works" if percent else "Works"),
                scale=x_scale, axis=x_axis),
        color=alt.Color("domain:N", legend=None,
                        scale=alt.Scale(domain=list(DOMAIN_COLORS),
                                        range=[DOMAIN_COLORS[k] for k in DOMAIN_COLORS])),
        tooltip=[
            alt.Tooltip("field:N", title="Field"),
            alt.Tooltip("domain:N", title="Domain"),
            alt.Tooltip("value:Q", title=("Share" if percent else "Count"),
                        format=(".0%" if percent else ",")),
        ],
    )

    layers = [bar]
    if show_counts and not percent:
        txt = base_ch.mark_text(align="left", baseline="middle", dx=6).encode(
            y=alt.Y("field:N", scale=y_scale, title=None),
            x=alt.value(0),
            text=alt.Text(f"{value_col}:Q", format=","),
        )
        layers.append(txt)

    return alt.layer(*layers).properties(
        height=h, width=width, padding={"left": 80, "right": 6, "top": 2, "bottom": 4}
    )

