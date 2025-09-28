# pages/1_🏭_Lab_View.py
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib.constants import YEAR_START, YEAR_END
from lib.data_io import (
    load_core,
    load_internal,
    topline_metrics,
    lab_summary_table_from_internal,
    lab_field_counts,
)
from lib.charts import field_mix_bars

st.set_page_config(page_title="Lab View — LUE Portfolio Explorer", page_icon="🏭", layout="wide")

st.title("🏭 Lab View")
st.caption(f"Default period: {YEAR_START}–{YEAR_END}")

# Load data
with st.spinner("Loading data…"):
    try:
        pubs = load_core()
    except Exception as e:
        st.error(f"Could not load pubs_final.parquet — {e}")
        st.stop()
    try:
        internal = load_internal()
    except Exception as e:
        st.error(f"Could not load dict_internal.parquet — {e}")
        st.stop()

# Topline metrics
m = topline_metrics(pubs, internal)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Number of labs", f"{m['n_labs']:,}")
k2.metric("Publications (2019–2023)", f"{m['n_pubs_total_19_23']:,}")
k3.metric("Publications with a lab (2019–2023)", f"{m['n_pubs_lab_19_23']:,}")
k4.metric("% covered by labs", f"{m['%_covered_by_labs']*100:.1f}%")

st.divider()

# ---- Year filter (applies to plots + OpenAlex link) --------------------------------
years_all = list(range(YEAR_START, YEAR_END + 1))
years_sel = st.multiselect("Filter years (affects the plots and OpenAlex links)", years_all, default=years_all)
if not years_sel:
    st.warning("Select at least one year.")
    st.stop()
year_min, year_max = min(years_sel), max(years_sel)

# ---- Per-lab table (from dict_internal) --------------------------------------------
st.subheader("Per-lab overview (2019–2023)")

summary = lab_summary_table_from_internal(internal, year_min=year_min, year_max=year_max)

# Progress bars with max set to highest lab value (so the leader shows a full bar)
max_share = float(summary["share_of_dataset_works"].max() or 1)
max_lue = float(summary["lue_pct"].max() or 1)
max_intl = float(summary["intl_pct"].max() or 1)
max_comp = float(summary["company_pct"].max() or 1)

st.dataframe(
    summary.rename(columns={"openalex_ui_url": "openalex_ui"}),
    use_container_width=True,
    hide_index=True,
    column_config={
        "lab_name": st.column_config.TextColumn("Lab"),
        "pubs_19_23": st.column_config.NumberColumn("Publications", format=","),

        # bar max = highest lab so the top one is 'full'
        "share_of_dataset_works": st.column_config.ProgressColumn(
            "% of all publications (2019–2023)", format="%.1f%%", min_value=0.0, max_value=max_share
        ),
        "lue_pct": st.column_config.ProgressColumn("% LUE publications", format="%.1f%%", min_value=0.0, max_value=max_lue),
        "intl_pct": st.column_config.ProgressColumn("% international", format="%.1f%%", min_value=0.0, max_value=max_intl),
        "company_pct": st.column_config.ProgressColumn("% with company", format="%.1f%%", min_value=0.0, max_value=max_comp),

        "avg_fwci": st.column_config.NumberColumn("Avg FWCI", format="%.2f"),
        "openalex_ui": st.column_config.LinkColumn("Open in OpenAlex (UI)"),
        "ror_url": st.column_config.LinkColumn("ROR page"),

        # hide backend columns
        "unit_ror": None,
        "institution_id": None,
    },
)

st.divider()

# ---- Compare two labs ---------------------------------------------------------------
st.subheader("Compare two labs")

# Options from internal so we get clean names
lab_options = internal[["laboratoire", "unit_ror"]].drop_duplicates().rename(
    columns={"laboratoire": "lab_name", "unit_ror": "lab_ror"}
)
labels = lab_options["lab_name"].tolist()
name_to_ror = dict(zip(lab_options["lab_name"], lab_options["lab_ror"]))

c1, c2 = st.columns(2)
with c1:
    left_label = st.selectbox("Left lab", labels, index=0 if labels else None)
with c2:
    right_label = st.selectbox("Right lab", labels, index=(1 if len(labels) > 1 else 0) if labels else None)

if not labels:
    st.info("No labs available.")
    st.stop()

left_ror = name_to_ror[left_label]
right_ror = name_to_ror[right_label]

# Precompute distributions for the selected years
lfc = lab_field_counts(pubs, years=years_sel)

pL, pR = st.columns(2, gap="large")
left_df = lfc[lfc["lab_ror"].eq(left_ror)].copy()
right_df = lfc[lfc["lab_ror"].eq(right_ror)].copy()

# Shared xmax for volume chart (max field count across both labs)
xmax = float(pd.concat([left_df["count"], right_df["count"]]).max() or 0)

for side, title, df_lab in [(pL, left_label, left_df), (pR, right_label, right_df)]:
    with side:
        st.markdown(f"### {title}")
        if df_lab.empty:
            st.info("No publications for this lab in the selected period.")
            continue

        st.markdown("**Field distribution (volume)**")
        st.altair_chart(
            field_mix_bars(df_lab, value_col="count", percent=False, xmax=xmax, enforce_order_from=list(lfc["field"].unique())),
            use_container_width=True,
        )

        st.markdown("**Field distribution (% of lab works)**")
        st.altair_chart(
            field_mix_bars(df_lab, value_col="count", percent=True, xmax=1.0, enforce_order_from=list(lfc["field"].unique())),
            use_container_width=True,
        )

        total = int(df_lab["count"].sum())
        lue = int(df_lab["in_lue_count"].sum())
        st.caption(
            f"Works counted across fields: {total:,}. "
            f"In_LUE portions are shown in darker shades ({lue:,} field-assignments from In_LUE works)."
        )

st.divider()
st.markdown("Use the sidebar to return to **Home** and switch dashboards.")
