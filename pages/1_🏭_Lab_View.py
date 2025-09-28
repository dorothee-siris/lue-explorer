# pages/1_🏭_Lab_View.py
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib.constants import YEAR_START, YEAR_END
from lib.data_io import (
    load_core,
    load_internal,
    topline_metrics,
    lab_summary_table,
    lab_field_counts,
)
from lib.charts import field_mix_bars

st.set_page_config(page_title="Lab View — LUE Portfolio Explorer", page_icon="🏭", layout="wide")

st.title("🏭 Lab View")
st.caption(f"Period: {YEAR_START}–{YEAR_END}")

# Load data
with st.spinner("Loading data…"):
    try:
        pubs = load_core()
    except Exception as e:
        st.error(f"Could not load pubs_final.parquet — {e}")
        st.stop()

    try:
        internal = load_internal()
    except Exception:
        internal = None

# Topline metrics
m = topline_metrics(pubs, internal)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Number of labs", f"{m['n_labs']:,}")
k2.metric("Publications (2019–2023)", f"{m['n_pubs_total_19_23']:,}")
k3.metric("Publications with a lab (2019–2023)", f"{m['n_pubs_lab_19_23']:,}")
k4.metric("% covered by labs", f"{m['%_covered_by_labs']*100:.1f}%")

st.divider()

# Per-lab table
st.subheader("Per-lab overview (2019–2023)")
summary = lab_summary_table(pubs, internal)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
    column_config={
        "lab_name": st.column_config.TextColumn("Lab"),
        "pubs_19_23": st.column_config.NumberColumn("Publications", format=","),
        "share_among_lab_works": st.column_config.ProgressColumn(
            "% of publications covered by labs", format="%.1f%%", min_value=0.0, max_value=1.0
        ),
        "avg_fwci": st.column_config.NumberColumn("Avg FWCI", format="%.2f"),
        "lab_ror": st.column_config.TextColumn("ROR ID"),
        "ror_url": st.column_config.LinkColumn("ROR page"),
        "openalex_url": st.column_config.LinkColumn("OpenAlex works (by ROR)"),
    },
)

st.divider()

# Compare two labs
st.subheader("Compare two labs")

lab_options = summary[["lab_name", "lab_ror"]].copy()
lab_options["label"] = lab_options.apply(
    lambda r: f"{r['lab_name']} ({r['lab_ror']})" if pd.notna(r["lab_name"]) and r["lab_name"] else r["lab_ror"],
    axis=1,
)

labels = lab_options["label"].tolist()
left_default = 0 if labels else None
right_default = (1 if len(labels) > 1 else 0) if labels else None

c1, c2 = st.columns(2)
with c1:
    left_choice = st.selectbox("Left lab", labels, index=left_default)
with c2:
    right_choice = st.selectbox("Right lab", labels, index=right_default)

if not labels:
    st.info("No labs available.")
    st.stop()

lab_map = dict(zip(lab_options["label"], lab_options["lab_ror"]))
left_ror = lab_map[left_choice]
right_ror = lab_map[right_choice]

lfc = lab_field_counts(pubs)

pL, pR = st.columns(2, gap="large")
for side, ror in zip([pL, pR], [left_ror, right_ror]):
    with side:
        title = lab_options.loc[lab_options["lab_ror"].eq(ror), "label"].iloc[0]
        st.markdown(f"### {title}")
        df_lab = lfc[lfc["lab_ror"].eq(ror)].copy()
        if df_lab.empty:
            st.info("No publications for this lab in the selected period.")
            continue

        st.markdown("**Field distribution (volume)**")
        st.altair_chart(field_mix_bars(df_lab, value_col="count", percent=False), use_container_width=True)

        st.markdown("**Field distribution (% of lab works)**")
        st.altair_chart(field_mix_bars(df_lab, value_col="count", percent=True), use_container_width=True)

        total = int(df_lab["count"].sum())
        lue = int(df_lab["in_lue_count"].sum())
        st.caption(
            f"Works counted across fields: {total:,}. "
            f"In_LUE portions are shown in darker shades ({lue:,} field-assignments from In_LUE works)."
        )

st.divider()
st.markdown("Tip: Use the sidebar to return to **Home** and switch dashboards.")
