# pages/4_🤝_Partners_View.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import re

from lib.constants import YEAR_START, YEAR_END
from lib.charts import simple_field_bars
from lib.transforms import all_fields_order
from lib.data_io import (
    load_core,
    load_internal,
    load_partners_ext,
    explode_institutions,
    explode_field_details,
    ul_field_counts_from_internal,
)

UL_ROR = "04vfs2w97"  # Université de Lorraine ROR (used for co-pub filters)

st.set_page_config(page_title="Partners View — LUE Portfolio Explorer", page_icon="🤝", layout="wide")
st.title("🤝 Partners View")
st.caption("Explore external partners, their field profiles, and collaboration patterns with Université de Lorraine.")

# ------------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------------
with st.spinner("Loading data…"):
    pubs = load_core()
    internal = load_internal()
    partners = load_partners_ext()

# ------------------------------------------------------------------------------------
# Landscape
# ------------------------------------------------------------------------------------
st.subheader("Landscape (2019–2023)")

n_partners = len(partners)
avg_partner_fwci = float(pd.to_numeric(partners.get("avg_fwci"), errors="coerce").mean() or 0.0)
n_companies = int((partners.get("partner_type", pd.Series(dtype=str)).str.lower() == "company").sum())

m1, m3, m4 = st.columns(3)
m1.metric("External partners (≥5 pubs)", f"{n_partners:,}")
m3.metric("Avg. FWCI across partners", f"{avg_partner_fwci:.2f}")
m4.metric("Company partners", f"{n_companies:,}")

# ------------------- Scatter (interactive) -------------------
st.markdown("### Partner bubble map")

# Controls
c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
with c1:
    scope = st.radio("Geo scope", ["All", "France only", "International only"], horizontal=True)
with c2:
    min_copubs_scatter = st.number_input("Min co-pubs (scatter)", min_value=0, max_value=2000, value=50, step=10)
with c3:
    size_scale_log = st.toggle("Log size", value=True, help="Use log1p(count) for bubble size.")
with c4:
    st.caption("Tip: click legend items to include/exclude partner types.")

# Prepare data
p_bubble = partners.copy()
p_bubble["pubs_partner_ul"] = pd.to_numeric(p_bubble["pubs_partner_ul"], errors="coerce").fillna(0)
p_bubble["relative_weight_to_total"] = pd.to_numeric(p_bubble["relative_weight_to_total"], errors="coerce")
p_bubble["avg_fwci"] = pd.to_numeric(p_bubble["avg_fwci"], errors="coerce")

# Scope filter
cc = p_bubble.get("country")
if scope == "France only":
    p_bubble = p_bubble[cc.str.upper().eq("FR")]
elif scope == "International only":
    p_bubble = p_bubble[cc.str.upper().ne("FR")]

# Min co-pubs filter
p_bubble = p_bubble[p_bubble["pubs_partner_ul"] >= min_copubs_scatter]

# Drop rows without the axes
p_bubble = p_bubble.dropna(subset=["relative_weight_to_total", "avg_fwci"])

if p_bubble.empty:
    st.info("No partners match your filters. Loosen the filters (scope or min co-pubs).")
else:
    # Size field (linear vs log)
    if size_scale_log:
        p_bubble["pubs_partner_ul_log"] = np.log1p(p_bubble["pubs_partner_ul"])
        size_field = "pubs_partner_ul_log"
        size_title = "Co-pubs (log scale)"
    else:
        size_field = "pubs_partner_ul"
        size_title = "Co-pubs"

    # Safe x-domain
    xmax = p_bubble["relative_weight_to_total"].max()
    if not np.isfinite(xmax) or xmax <= 0:
        x_scale = alt.Scale(domain=[0, 1])
    else:
        x_scale = alt.Scale(domain=[0, float(xmax)])

    # Legend-click selection
    sel_type = alt.selection_point(fields=["partner_type"], bind="legend")

    bubble = (
        alt.Chart(p_bubble)
        .add_params(sel_type)
        .transform_filter(sel_type)
        .mark_circle(opacity=0.75)
        .encode(
            x=alt.X(
                "relative_weight_to_total:Q",
                title="Share of partner output (2019–2023)",
                axis=alt.Axis(format="%"),
                scale=x_scale,
            ),
            y=alt.Y("avg_fwci:Q", title="Avg. FWCI (UL × partner)"),
            size=alt.Size(f"{size_field}:Q", title=size_title, scale=alt.Scale(range=[10, 1200])),
            color=alt.Color("partner_type:N", title="Partner type"),
            tooltip=[
                alt.Tooltip("partner_name:N", title="Partner"),
                alt.Tooltip("partner_type:N", title="Type"),
                alt.Tooltip("country:N", title="Country"),
                alt.Tooltip("pubs_partner_ul:Q", title="Co-pubs", format=","),
                alt.Tooltip("relative_weight_to_total:Q", title="Share of partner output", format=".1%"),
                alt.Tooltip("avg_fwci:Q", title="Avg. FWCI", format=".2f"),
            ],
        )
        .properties(height=380)
    )
    st.altair_chart(bubble, use_container_width=True)

# Ranked table with quick filters
st.markdown("### Top partners")

t1, t2, t3 = st.columns(3)
with t1:
    type_filter = st.selectbox("Filter by type", ["(all)"] + sorted(partners["partner_type"].dropna().unique().tolist()))
with t2:
    country_filter = st.selectbox("Filter by country", ["(all)"] + sorted(partners["country"].dropna().unique().tolist()))
with t3:
    min_pubs_table = st.number_input("Min co-publications (table)", min_value=5, max_value=2000, value=100, step=5)

ptab = partners.copy()
if type_filter != "(all)":
    ptab = ptab[ptab["partner_type"].eq(type_filter)]
if country_filter != "(all)":
    ptab = ptab[ptab["country"].eq(country_filter)]
ptab = ptab[pd.to_numeric(ptab["pubs_partner_ul"], errors="coerce").fillna(0).ge(min_pubs_table)]

ptab["ul_share_display"] = pd.to_numeric(ptab["share_of_ul"], errors="coerce").fillna(0) * 100.0
ptab["rel_weight_display"] = pd.to_numeric(ptab["relative_weight_to_total"], errors="coerce").fillna(0) * 100.0

st.dataframe(
    ptab.sort_values(["pubs_partner_ul", "avg_fwci"], ascending=[False, False])[
        ["partner_name", "partner_type", "country", "pubs_partner_ul", "avg_fwci", "ul_share_display", "partner_total_works", "rel_weight_display"]
    ],
    use_container_width=True,
    hide_index=True,
    column_config={
        "partner_name": st.column_config.TextColumn("Partner"),
        "partner_type": st.column_config.TextColumn("Type"),
        "country": st.column_config.TextColumn("Country"),
        "pubs_partner_ul": st.column_config.NumberColumn("Co-pubs", format="%.0f"),
        "avg_fwci": st.column_config.NumberColumn("Avg. FWCI", format="%.2f"),
        "ul_share_display": st.column_config.ProgressColumn("% of UL output", format="%.1f %%",
            min_value=0.0, max_value=float(ptab["ul_share_display"].max() or 1.0)),
        "partner_total_works": st.column_config.NumberColumn("Partner total works (19–23)", format="%.0f"),
        # renamed column here ↓
        "rel_weight_display": st.column_config.ProgressColumn("Share of partner output", format="%.1f %%",
            min_value=0.0, max_value=float(ptab["rel_weight_display"].max() or 1.0)),
    },
)

st.divider()

# ------------------------------------------------------------------------------------
# Compare a partner to Université de Lorraine (field mix)
# ------------------------------------------------------------------------------------
st.subheader("Compare a partner to Université de Lorraine")

query = st.text_input("Search partner (type a few letters of the name/country/type)", "")
candidates = partners.copy()
if query.strip():
    q = query.strip().lower()
    candidates = candidates[
        candidates["partner_name"].str.lower().str.contains(q, na=False)
        | candidates["country"].str.lower().str.contains(q, na=False)
        | candidates["partner_type"].str.lower().str.contains(q, na=False)
    ]
candidates = candidates.sort_values("pubs_partner_ul", ascending=False).head(50)

if candidates.empty:
    st.info("Type a few characters to find a partner.")
else:
    labels = [f"{r.partner_name} — {r.partner_type}, {r.country}" for _, r in candidates.iterrows()]
    pick = st.selectbox("Pick a partner", options=list(range(len(labels))), format_func=lambda i: labels[i])
    partner_row = candidates.iloc[pick]

    # Partner field mix (count) from the compact string
    partner_fields = explode_field_details(partner_row.get("fields_details"), "count", "fwci", a_is_int=True)[["field", "count"]]

    # UL field mix (count) using the robust helper (fixes your KeyError)
    ul_fields = ul_field_counts_from_internal(internal, prefer="institution_ROR")

    catalogue = all_fields_order()
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(f"### {partner_row['partner_name']}")
        st.altair_chart(
            simple_field_bars(partner_fields, value_col="count", percent=False,
                              enforce_order_from=catalogue, show_counts=True, width=560),
            use_container_width=True,
        )
        st.altair_chart(
            simple_field_bars(partner_fields, value_col="count", percent=True,
                              enforce_order_from=catalogue, show_counts=False, width=560),
            use_container_width=True,
        )
    with right:
        st.markdown("### Université de Lorraine")
        st.altair_chart(
            simple_field_bars(ul_fields, value_col="count", percent=False,
                              enforce_order_from=catalogue, show_counts=True, width=560),
            use_container_width=True,
        )
        st.altair_chart(
            simple_field_bars(ul_fields, value_col="count", percent=True,
                              enforce_order_from=catalogue, show_counts=False, width=560),
            use_container_width=True,
        )

    # --------------------------------------------------------------------------------
    # Collaboration details (UL × selected partner)
    # --------------------------------------------------------------------------------
    st.divider()
    st.subheader("Collaboration with UL — details")

    inst_ror = partner_row.get("inst_ror")
    inst_id = partner_row.get("inst_id")

    e = explode_institutions(pubs)
    mask_partner = False
    if pd.notna(inst_ror) and str(inst_ror).strip():
        mask_partner = (e["inst_ror"] == str(inst_ror).strip())
    if pd.notna(inst_id) and str(inst_id).strip():
        mask_partner = mask_partner | (e["inst_id"] == str(inst_id).strip())

    ids_with_partner = set(e.loc[mask_partner, "openalex_id"])
    ids_with_ul = set(e.loc[e["inst_ror"] == UL_ROR, "openalex_id"])
    collab_ids = ids_with_partner.intersection(ids_with_ul)

    years_all = list(range(YEAR_START, YEAR_END + 1))
    years_sel = st.multiselect("Filter years for the collaboration section", years_all, default=years_all, key="years_partner")
    collab = pubs[(pubs["OpenAlex ID"].isin(collab_ids)) & (pubs["Publication Year"].isin(years_sel))].copy()

    if collab.empty:
        st.info("No co-publications for the selected years.")
    else:
        total = int(collab["OpenAlex ID"].nunique())
        lue_count = int(collab.get("In_LUE", pd.Series([False]*len(collab))).fillna(False).astype(bool).sum())
        lue_pct = (lue_count / total) if total else 0.0
        avg_fwci = float(pd.to_numeric(collab.get("Field-Weighted Citation Impact"), errors="coerce").mean() or 0.0)

        def _has_non_fr(c):
            if c is None: return False
            toks = [t.strip().upper() for t in str(c).replace(";", "|").split("|") if t.strip()]
            return any(t and t != "FR" for t in toks)

        def _has_company(t):
            if t is None: return False
            toks = [t.strip().lower() for t in str(t).replace(";", "|").split("|") if t.strip()]
            return any("company" in t for t in toks)

        intl = int(collab.get("Institution Countries", pd.Series([None]*len(collab))).map(_has_non_fr).sum())
        intl_pct = (intl / total) if total else 0.0
        comp = int(collab.get("Institution Types", pd.Series([None]*len(collab))).map(_has_company).sum())
        comp_pct = (comp / total) if total else 0.0

        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("Co-publications", f"{total:,}")
        t2.metric("ISITE (count / %)", f"{lue_count:,} / {lue_pct*100:.1f}%")
        t3.metric("Avg. FWCI", f"{avg_fwci:.2f}")
        t4.metric("% international", f"{intl_pct*100:.1f}%")
        t5.metric("% with company", f"{comp_pct*100:.1f}%")

        type_col = "Publication Type"
        cop = collab.copy()
        cop["doc_type"] = cop[type_col].str.lower().map({
            "article": "article",
            "review": "review",
            "book-chapter": "book-chapter",
            "chapter": "book-chapter",
            "book": "book",
        }).fillna("other")
        doc_order = ["article", "review", "book-chapter", "book", "other"]

        year_counts = cop.groupby(["Publication Year", "doc_type"])["OpenAlex ID"].nunique().reset_index(name="count")
        chart = (
            alt.Chart(year_counts)
            .mark_bar()
            .encode(
                x=alt.X("Publication Year:O", title="Year", sort=sorted(years_sel)),
                y=alt.Y("count:Q", title="Co-publications"),
                color=alt.Color("doc_type:N", title="Type", sort=doc_order),
                tooltip=[alt.Tooltip("Publication Year:O"), alt.Tooltip("doc_type:N"), alt.Tooltip("count:Q", format=",")],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)

        # Co-publications table (top 100 by default)
        cols, rename_out = [], {}
        wanted = [
            ("OpenAlex ID", "OpenAlex ID"),
            ("DOI", "DOI"),
            ("Publication Year", "Publication Year"),
            ("Publication Type", "Publication Type"),
            ("Title", "Title"),
            ("Citation Count", "Citation Count"),
            ("Field-Weighted Citation Impact", "Field-Weighted Citation Impact"),
            ("In_LUE", "In LUE"),
            ("All Topics", "All Topics"),
            ("All Subfields", "All Subfields"),
            ("All Fields", "All Fields"),
            ("All Domains", "All Domains"),
        ]
        for c_in, c_out in wanted:
            if c_in in collab.columns:
                cols.append(c_in); rename_out[c_in] = c_out

        detail = collab[cols].rename(columns=rename_out).drop_duplicates()
        detail = detail.sort_values(["Citation Count", "Field-Weighted Citation Impact"], ascending=[False, False])

        show_all = st.toggle("Show full list (may be long)", value=False)
        display_df = detail if show_all else detail.head(100)

        st.markdown("**Co-publications**")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV",
            data=detail.to_csv(index=False).encode("utf-8"),
            file_name=f"partner_collab_{partner_row['partner_name']}_{min(years_sel)}-{max(years_sel)}.csv",
            mime="text/csv",
        )

st.divider()

# ------------------------------------------------------------------------------------
# Focus on companies
# ------------------------------------------------------------------------------------
st.subheader("Focus on companies")

companies = partners[partners["partner_type"].str.lower().eq("company")]
if companies.empty:
    st.info("No company partners detected in the dictionary.")
else:
    comp_bubble = (
        alt.Chart(companies)
        .mark_circle(opacity=0.75)
        .encode(
            x=alt.X("relative_weight_to_total:Q", title="Share of partner output", axis=alt.Axis(format="%")),
            y=alt.Y("avg_fwci:Q", title="Avg. FWCI (UL × company)"),
            size=alt.Size("pubs_partner_ul:Q", title="Co-pubs", scale=alt.Scale(range=[10, 1200])),
            color=alt.Color("country:N", title="Country"),
            tooltip=[
                alt.Tooltip("partner_name:N", title="Company"),
                alt.Tooltip("country:N", title="Country"),
                alt.Tooltip("pubs_partner_ul:Q", title="Co-pubs", format=","),
                alt.Tooltip("relative_weight_to_total:Q", title="Share of partner output", format=".1%"),
                alt.Tooltip("avg_fwci:Q", title="Avg. FWCI", format=".2f"),
            ],
        )
        .properties(height=360)
    )
    st.altair_chart(comp_bubble, use_container_width=True)

    st.markdown("**Top companies**")
    st.dataframe(
        companies.sort_values(["pubs_partner_ul", "avg_fwci"], ascending=[False, False])[
            ["partner_name", "country", "pubs_partner_ul", "avg_fwci", "partner_total_works", "relative_weight_to_total"]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "partner_name": st.column_config.TextColumn("Company"),
            "country": st.column_config.TextColumn("Country"),
            "pubs_partner_ul": st.column_config.NumberColumn("Co-pubs", format="%.0f"),
            "avg_fwci": st.column_config.NumberColumn("Avg. FWCI", format="%.2f"),
            "partner_total_works": st.column_config.NumberColumn("Company total works (19–23)", format="%.0f"),
            "relative_weight_to_total": st.column_config.ProgressColumn("Share of partner output", format="%.1f %%",
                min_value=0.0, max_value=float((companies["relative_weight_to_total"]*100).max() or 1.0)),
        },
    )

st.divider()
st.markdown("Use the sidebar to return to **Home** or other views.")
