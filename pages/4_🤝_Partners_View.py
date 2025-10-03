# pages/4_🤝_Partners_View.py
from __future__ import annotations

import re
from typing import List

import altair as alt
import pandas as pd
import streamlit as st

from lib.constants import YEAR_START, YEAR_END
from lib.data_io import (
    load_core,
    partners_joined,
    load_topics,
    ul_field_counts_from_fields_table,
    explode_institutions,
    explode_labs,
    load_authors_lookup,
    author_global_metrics,
    explode_field_details,  # parser for "Fields distribution (count ; FWCI_FR ; top10 ; top1)"
)

st.set_page_config(page_title="Partners View — LUE Portfolio Explorer", page_icon="🤝", layout="wide")

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def safe_num(x, default=0):
    try:
        v = float(x)
        if pd.isna(v):
            return default
        return v
    except Exception:
        return default

def canonical_field_order(topics: pd.DataFrame) -> List[str]:
    """
    Create a stable order for the 26 Fields: by domain name, then field name.
    """
    look = topics.drop_duplicates(["field_id", "field_name", "domain_name"])
    look = look.sort_values(["domain_name", "field_name"])
    return look["field_name"].tolist()

def complete_fields_frame(df_counts: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    """
    Ensure all fields in 'order' appear with zero counts if missing.
    Expects columns: field_name, domain_name, count
    """
    left = pd.DataFrame({"field_name": order})
    out = left.merge(df_counts, on="field_name", how="left")
    out["count"] = pd.to_numeric(out["count"], errors="coerce").fillna(0).astype(int)
    # carry domain_name forward if missing
    out["domain_name"] = out["domain_name"].fillna(method="ffill").fillna("Other")
    return out

def field_bar_chart(df_counts: pd.DataFrame, title: str, y_order: List[str], show_counts=True, width=560):
    """
    Horizontal bar chart with optional count labels near axis origin.
    Requires columns: field_name, domain_name, count
    - Always shows ALL field labels (including zero bars) by completing with y_order.
    - Fixed width to keep two panels visually comparable.
    """
    data = complete_fields_frame(df_counts[["field_name", "domain_name", "count"]].copy(), y_order)
    base = alt.Chart(data).encode(
        y=alt.Y("field_name:N", sort=y_order, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("count:Q", title="Publications"),
        color=alt.Color("domain_name:N", title="Domain"),
        tooltip=[
            alt.Tooltip("field_name:N", title="Field"),
            alt.Tooltip("domain_name:N", title="Domain"),
            alt.Tooltip("count:Q", title="Publications", format=","),
        ],
    )

    bars = base.mark_bar()
    if show_counts:
        # print counts just to the right of zero (works even for 0 with dx)
        labels = base.mark_text(align="left", dx=4).encode(text=alt.Text("count:Q", format=","))
        chart = (bars + labels)
    else:
        chart = bars

    # height proportional to number of fields (keeps labels readable)
    height = max(26 * 18, 350)
    return chart.properties(title=title, width=width, height=height)

def search_select_partner(df_partners: pd.DataFrame):
    """
    Text search box + selectbox of matching partners (by name or ROR).
    Returns the selected row (as Series) or None.
    """
    st.markdown("### Compare a partner with Université de Lorraine")
    q = st.text_input("Search a partner (type a few letters of the name or paste a ROR ID):", "")
    matches = df_partners.copy()

    if q.strip():
        qs = re.escape(q.strip())
        name_mask = matches["partner_name"].fillna("").str.contains(qs, case=False, regex=True)
        ror_mask = matches["inst_ror"].fillna("").str.contains(qs, case=False, regex=True)
        matches = matches[name_mask | ror_mask]

    matches = matches.sort_values(["copubs", "avg_fwci_fr"], ascending=[False, False]).head(50)
    label_map = {
        f"{r.partner_name} — {r.country or 'Unknown'} [{r.partner_type or 'unknown'}]": r.inst_ror
        for _, r in matches.iterrows()
    }

    if not label_map:
        st.info("No matching partners. Refine your search.")
        return None

    sel_label = st.selectbox("Select a partner from search results:", list(label_map.keys()))
    sel_ror = label_map.get(sel_label)
    if not sel_ror:
        return None
    row = df_partners[df_partners["inst_ror"] == sel_ror].head(1)
    return None if row.empty else row.iloc[0]

# --------------------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------------------
with st.spinner("Loading data…"):
    pubs = load_core()
    topics = load_topics()
    ul_fields = ul_field_counts_from_fields_table()
    partners = partners_joined()

# Canonical field order for all field charts
FIELD_ORDER = canonical_field_order(topics)

st.title("🤝 Partners View")

# --------------------------------------------------------------------------------------
# Landscape KPIs (high level)
# --------------------------------------------------------------------------------------
st.subheader("Landscape")
# (Removed the “UL × partner co-publications (sum)” KPI per your request.)
n_partners = int(partners["inst_ror"].nunique())
n_countries = int(partners["country"].dropna().nunique())
n_companies = int((partners["partner_type"].fillna("").str.lower().str.contains("company")).sum())

median_fwci = float(partners["avg_fwci_fr"].median(skipna=True) or 0.0)
median_share_partner = float(partners["share_partner_output"].median(skipna=True) or 0.0)  # share of partner output

m1, m2, m3, m4 = st.columns(4)
m1.metric("Top partners (≥5 co-pubs)", f"{n_partners:,}")
m2.metric("Countries represented", f"{n_countries:,}")
m3.metric("Company partners", f"{n_companies:,}")
m4.metric("Median FWCI (FR)", f"{median_fwci:.2f}")

st.divider()

# --------------------------------------------------------------------------------------
# Scatter — strategic landscape
# --------------------------------------------------------------------------------------
st.subheader("Where are our strongest partners?")

# Controls
c1, c2, c3, c4 = st.columns([1, 1, 1, 1.5])
with c1:
    partner_types = partners["partner_type"].fillna("unknown").str.lower().replace("", "unknown")
    unique_types = sorted(partner_types.unique().tolist())
    sel_types = st.multiselect("Partner type(s)", unique_types, default=unique_types)

with c2:
    scope = st.radio("Geography", ["All", "France only", "International only"], index=0, horizontal=False)

with c3:
    min_copubs = st.slider("Min co-publications (scatter)", min_value=0, max_value=500, value=50, step=10)

with c4:
    y_log = st.toggle("Log scale on FWCI", value=False, help="Apply logarithmic scale to the Y axis (FWCI)")

# Prepare
df_scatter = partners.copy()
df_scatter["partner_type"] = df_scatter["partner_type"].fillna("unknown").str.lower().replace("", "unknown")
df_scatter["country"] = df_scatter["country"].fillna("Unknown")
df_scatter["copubs"] = pd.to_numeric(df_scatter["copubs"], errors="coerce").fillna(0)
df_scatter["share_partner_output"] = pd.to_numeric(df_scatter["share_partner_output"], errors="coerce").fillna(0.0)
df_scatter["avg_fwci_fr"] = pd.to_numeric(df_scatter["avg_fwci_fr"], errors="coerce").fillna(0.0)

# Filters
df_scatter = df_scatter[df_scatter["partner_type"].isin(sel_types)]
if scope == "France only":
    df_scatter = df_scatter[df_scatter["country"].eq("France")]
elif scope == "International only":
    df_scatter = df_scatter[df_scatter["country"].ne("France") & df_scatter["country"].notna()]

df_scatter = df_scatter[df_scatter["copubs"] >= min_copubs]
# avoid NaNs/inf for Vega — drop rows missing core measures
df_scatter = df_scatter[pd.notna(df_scatter["share_partner_output"]) & pd.notna(df_scatter["avg_fwci_fr"])]

if df_scatter.empty:
    st.info("No partners match your filters.")
else:
    sel = alt.selection_point(fields=["partner_type"], bind="legend")
    y_scale = alt.Scale(type="log") if y_log else alt.Scale()

    scatter = (
        alt.Chart(df_scatter)
        .mark_circle(opacity=0.75)
        .encode(
            x=alt.X("share_partner_output:Q", title="Share of partner output"),
            y=alt.Y("avg_fwci_fr:Q", title="Average FWCI (FR)", scale=y_scale),
            size=alt.Size("copubs:Q", title="Co-publications", scale=alt.Scale(range=[30, 1200])),
            color=alt.Color("partner_type:N", title="Type"),
            tooltip=[
                alt.Tooltip("partner_name:N", title="Partner"),
                alt.Tooltip("country:N", title="Country"),
                alt.Tooltip("partner_type:N", title="Type"),
                alt.Tooltip("copubs:Q", title="Co-pubs", format=","),
                alt.Tooltip("share_partner_output:Q", title="Share of partner output", format=".2%"),
                alt.Tooltip("avg_fwci_fr:Q", title="FWCI (FR)", format=".2f"),
            ],
        )
        .add_params(sel)
        .transform_filter(sel)
        .properties(height=700, use_container_width=True)
    )
    st.altair_chart(scatter, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------------------
# Top partners table
# --------------------------------------------------------------------------------------
st.subheader("Top partners table")

# Default filter min copubs 100 (per your request)
min_table_copubs = st.slider("Minimum co-publications (table)", 0, 2000, 100, step=25)
df_table = partners.copy()
df_table["copubs"] = pd.to_numeric(df_table["copubs"], errors="coerce").fillna(0).astype(int)
df_table = df_table[df_table["copubs"] >= min_table_copubs]

# Display-friendly columns
show = df_table[
    [
        "partner_name",
        "country",
        "partner_type",
        "copubs",
        "share_partner_output",
        "avg_fwci_fr",
        "partner_total_works",
    ]
].rename(
    columns={
        "partner_name": "Partner",
        "country": "Country",
        "partner_type": "Type",
        "copubs": "Co-publications",
        "share_partner_output": "Share of partner output",
        "avg_fwci_fr": "Avg. FWCI (FR)",
        "partner_total_works": "Partner works (2019–2023)",
    }
)

st.dataframe(
    show,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Co-publications": st.column_config.NumberColumn(format="%.0f"),
        "Share of partner output": st.column_config.NumberColumn(format="%.2f %%", help="(UL co-pubs) / (Partner total works) × 100"),
        "Avg. FWCI (FR)": st.column_config.NumberColumn(format="%.2f"),
        "Partner works (2019–2023)": st.column_config.NumberColumn(format="%.0f"),
    },
)

st.divider()

# --------------------------------------------------------------------------------------
# Comparison: UL vs Selected Partner — Field distribution
# --------------------------------------------------------------------------------------
sel_row = search_select_partner(partners)
if sel_row is not None:
    p_left, p_right = st.columns(2, gap="large")

    # Partner fields
    fields_det = explode_field_details(sel_row.get("fields_details"))
    if fields_det is None or fields_det.empty:
        partner_fields = pd.DataFrame(columns=["field_id", "field_name", "domain_name", "count"])
    else:
        # map id -> names
        partner_fields = fields_det.rename(columns={"id": "field_id", "count": "count"})
        partner_fields = partner_fields.merge(
            topics.drop_duplicates("field_id")[["field_id", "field_name", "domain_name"]],
            on="field_id",
            how="left",
        )
        partner_fields["field_name"] = partner_fields["field_name"].fillna("Unknown")
        partner_fields["domain_name"] = partner_fields["domain_name"].fillna("Other")

    # UL fields
    ul_counts = ul_fields.rename(columns={"count": "count"})[["field_id", "field_name", "domain_name", "count"]]

    with p_left:
        st.markdown(f"### {sel_row.partner_name} — Field distribution (count)")
        st.altair_chart(
            field_bar_chart(partner_fields, title="Partner", y_order=FIELD_ORDER, show_counts=True, width=560),
            use_container_width=True,
        )
    with p_right:
        st.markdown("### Université de Lorraine — Field distribution (count)")
        st.altair_chart(
            field_bar_chart(ul_counts, title="Université de Lorraine", y_order=FIELD_ORDER, show_counts=True, width=560),
            use_container_width=True,
        )

    st.divider()

    # ----------------------------------------------------------------------------------
    # Collaboration detail: UL × Partner
    # ----------------------------------------------------------------------------------
    st.subheader(f"Collaboration details — UL × {sel_row.partner_name}")

    # Identify co-publications: intersection of (works with any UL lab) and (works with this partner ROR)
    elabs = explode_labs(pubs)
    ul_work_ids = set(elabs["openalex_id"].unique())

    eins = explode_institutions(pubs)
    partner_work_ids = set(eins.loc[eins["inst_ror"] == sel_row.inst_ror, "openalex_id"].unique())

    copub_ids = ul_work_ids.intersection(partner_work_ids)
    copubs = pubs[pubs["openalex_id"].isin(copub_ids) & pubs["year"].between(YEAR_START, YEAR_END)].copy()

    if copubs.empty:
        st.info("No co-publications in the selected period.")
    else:
        # KPIs
        total = int(copubs["openalex_id"].nunique())
        lue_count = int(copubs["in_lue"].fillna(False).astype(bool).sum()) if "in_lue" in copubs.columns else 0
        lue_pct = (lue_count / total) if total else 0.0
        avg_fwci = float(pd.to_numeric(copubs.get("fwci_fr", copubs.get("fwci_all")), errors="coerce").mean() or 0.0)

        # International/company flags via exploded institutions for *these works only*
        e_cop = eins[eins["openalex_id"].isin(copubs["openalex_id"])].copy()
        # International = any non-France country among co-authors
        intl_works = (
            e_cop.assign(is_non_fr=e_cop["inst_country"].fillna("").astype(str).str.upper().ne("FR"))
            .groupby("openalex_id")["is_non_fr"]
            .any()
        )
        pct_intl = float(intl_works.mean()) if len(intl_works) else 0.0
        # Company = any institution type contains 'company'
        comp_works = (
            e_cop.assign(is_company=e_cop["inst_type"].fillna("").str.lower().str.contains("company"))
            .groupby("openalex_id")["is_company"]
            .any()
        )
        pct_company = float(comp_works.mean()) if len(comp_works) else 0.0

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Co-publications", f"{total:,}")
        k2.metric("ISITE (count / %)", f"{lue_count:,} / {lue_pct*100:.1f}%")
        k3.metric("Avg. FWCI (FR)", f"{avg_fwci:.2f}")
        k4.metric("% international", f"{pct_intl*100:.1f}%")
        k5.metric("% with company", f"{pct_company*100:.1f}%")

        # Stacked bars: document type per year
        type_col = "type"  # normalized on load_core
        cop = copubs.copy()
        cop["doc_type"] = cop[type_col].str.lower().map(
            {
                "article": "article",
                "review": "review",
                "book-chapter": "book-chapter",
                "chapter": "book-chapter",
                "book": "book",
            }
        ).fillna("other")
        doc_order = ["article", "review", "book-chapter", "book", "other"]

        year_counts = cop.groupby(["year", "doc_type"])["openalex_id"].nunique().reset_index(name="count")

        chart = (
            alt.Chart(year_counts)
            .mark_bar()
            .encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("count:Q", title="Co-publications"),
                color=alt.Color("doc_type:N", title="Type", sort=doc_order),
                tooltip=[
                    alt.Tooltip("year:O"),
                    alt.Tooltip("doc_type:N", title="Type"),
                    alt.Tooltip("count:Q", title="Co-pubs", format=","),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)

        # Top authors in this collaboration (enriched)
        st.markdown("### Top authors in these co-publications")
        # Build author rows from (Authors, Authors ID) — robust without relying on an extra helper
        rows = []
        LEAD = re.compile(r"^\[\d+\]\s*")
        for _, r in copubs[["openalex_id", "authors", "authors_id"]].iterrows():
            names = [LEAD.sub("", x).strip() for x in str(r["authors"] or "").split("|") if x.strip()]
            ids = [LEAD.sub("", x).strip() for x in str(r["authors_id"] or "").split("|") if x.strip()]
            if len(names) < len(ids):
                names += [""] * (len(ids) - len(names))
            if len(ids) < len(names):
                ids += [""] * (len(names) - len(ids))
            for nm, aid in zip(names, ids):
                if not aid and not nm:
                    continue
                rows.append({"author_id": aid, "Author": nm, "openalex_id": r["openalex_id"]})
        ea = pd.DataFrame(rows)

        top_counts = (
            ea.groupby(["author_id", "Author"], as_index=False)
            .agg(Copubs=("openalex_id", "nunique"))
            .sort_values("Copubs", ascending=False)
        )

        g = author_global_metrics(pubs).rename(
            columns={"author_id": "author_id", "author_name": "Author", "total_pubs": "Total publications", "avg_fwci_overall": "Avg. FWCI (overall)"}
        )

        lk = load_authors_lookup()
        if lk is not None and not lk.empty:
            lk = lk.rename(
                columns={
                    "author_id": "author_id",
                    "orcid": "ORCID",
                    "is_lorraine": "Is Lorraine",
                    "labs_from_dict": "Lab(s)",
                }
            )[["author_id", "ORCID", "Is Lorraine", "Lab(s)"]]
        else:
            lk = pd.DataFrame(columns=["author_id", "ORCID", "Is Lorraine", "Lab(s)"])

        top_authors = top_counts.merge(g, on=["author_id", "Author"], how="left").merge(lk, on="author_id", how="left")
        top_authors = top_authors.sort_values(["Copubs", "Avg. FWCI (overall)"], ascending=[False, False]).head(25)

        st.dataframe(
            top_authors[["Author", "author_id", "ORCID", "Copubs", "Total publications", "Avg. FWCI (overall)", "Is Lorraine", "Lab(s)"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "author_id": st.column_config.TextColumn("Author ID"),
                "Copubs": st.column_config.NumberColumn(format="%.0f"),
                "Total publications": st.column_config.NumberColumn(format="%.0f"),
                "Avg. FWCI (overall)": st.column_config.NumberColumn(format="%.2f"),
            },
        )

        # All co-publications (top 100 with option to expand)
        st.markdown("### All co-publications (exportable)")
        want_cols = {
            "openalex_id": "OpenAlex ID",
            "doi": "DOI",
            "type": "Publication Type",
            "year": "Publication Year",
            "title": "Title",
            "citation_count": "Citation Count",
            "fwci_fr": "Field-Weighted Citation Impact (FR)",
            "in_lue": "In LUE",
            "primary_topic_id": "Primary Topic",
            "primary_subfield_id": "Primary Subfield ID",
            "primary_field_id": "Primary Field ID",
            "primary_domain_id": "Primary Domain ID",
        }
        copub_table = copubs[[c for c in want_cols if c in copubs.columns]].rename(columns=want_cols)

        # Show 100 by default
        show_all = st.toggle("Show full list", value=False)
        view_tbl = copub_table if show_all else copub_table.head(100)
        st.dataframe(view_tbl, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV",
            data=copub_table.to_csv(index=False).encode("utf-8"),
            file_name=f"copubs_UL_x_{sel_row.partner_name}_{YEAR_START}-{YEAR_END}.csv",
            mime="text/csv",
        )

st.divider()

# --------------------------------------------------------------------------------------
# Focus on companies
# --------------------------------------------------------------------------------------
st.subheader("Focus on companies")
min_copubs_comp = st.slider("Minimum co-publications (companies table)", 0, 1000, 25, step=5)
companies = partners.copy()
companies["partner_type"] = companies["partner_type"].fillna("").str.lower()
companies = companies[companies["partner_type"].str.contains("company")]

if companies.empty:
    st.info("No company partners found.")
else:
    companies["copubs"] = pd.to_numeric(companies["copubs"], errors="coerce").fillna(0).astype(int)
    companies["share_partner_output"] = pd.to_numeric(companies["share_partner_output"], errors="coerce").fillna(0.0)
    companies["avg_fwci_fr"] = pd.to_numeric(companies["avg_fwci_fr"], errors="coerce").fillna(0.0)

    comp_table = companies[companies["copubs"] >= min_copubs_comp][
        ["partner_name", "country", "copubs", "share_partner_output", "avg_fwci_fr", "partner_total_works"]
    ].rename(
        columns={
            "partner_name": "Company",
            "country": "Country",
            "copubs": "Co-publications",
            "share_partner_output": "Share of partner output",
            "avg_fwci_fr": "Avg. FWCI (FR)",
            "partner_total_works": "Company works (2019–2023)",
        }
    ).sort_values(["Co-publications", "Avg. FWCI (FR)"], ascending=[False, False])

    st.dataframe(
        comp_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Co-publications": st.column_config.NumberColumn(format="%.0f"),
            "Share of partner output": st.column_config.NumberColumn(format="%.2f %%"),
            "Avg. FWCI (FR)": st.column_config.NumberColumn(format="%.2f"),
            "Company works (2019–2023)": st.column_config.NumberColumn(format="%.0f"),
        },
    )

    # Optional small scatter for companies only
    st.markdown("**Companies landscape (optional)**")
    comp_scatter = (
        alt.Chart(companies[companies["copubs"] >= min_copubs_comp])
        .mark_circle(opacity=0.75)
        .encode(
            x=alt.X("share_partner_output:Q", title="Share of partner output"),
            y=alt.Y("avg_fwci_fr:Q", title="Average FWCI (FR)"),
            size=alt.Size("copubs:Q", title="Co-publications", scale=alt.Scale(range=[30, 1200])),
            color=alt.Color("country:N", title="Country"),
            tooltip=[
                alt.Tooltip("partner_name:N", title="Company"),
                alt.Tooltip("country:N", title="Country"),
                alt.Tooltip("copubs:Q", title="Co-pubs", format=","),
                alt.Tooltip("share_partner_output:Q", title="Share of partner output", format=".2%"),
                alt.Tooltip("avg_fwci_fr:Q", title="FWCI (FR)", format=".2f"),
            ],
        )
        .properties(height=500)
    )
    st.altair_chart(comp_scatter, use_container_width=True)

st.caption("Use the sidebar to return to **Home** or switch dashboards.")
