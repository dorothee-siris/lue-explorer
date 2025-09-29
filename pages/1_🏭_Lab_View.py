# pages/1_🏭_Lab_View.py
from __future__ import annotations

import re
import pandas as pd
import streamlit as st
import altair as alt

from lib.constants import YEAR_START, YEAR_END
from lib.charts import field_mix_bars
from lib.transforms import all_fields_order
from lib.data_io import (
    load_core,
    load_internal,
    topline_metrics,
    lab_summary_table_from_internal,
    lab_field_counts,
    explode_labs,
    author_global_metrics,
    load_authors_lookup,
)

# ---------------------------------------------------------------------
# Page config (must be before any output)
# ---------------------------------------------------------------------
st.set_page_config(page_title="Lab View — LUE Portfolio Explorer", page_icon="🏭", layout="wide")

st.title("🏭 Lab View")
st.caption(f"Default period: {YEAR_START}–{YEAR_END}")

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
with st.spinner("Loading data…"):
    try:
        pubs = load_core()      # normalized columns (e.g., pub_type, inst_types, inst_countries, fwci, etc.)
    except Exception as e:
        st.error(f"Could not load pubs_final.parquet — {e}")
        st.stop()

    try:
        internal = load_internal()
    except Exception as e:
        st.error(f"Could not load dict_internal.parquet — {e}")
        st.stop()

# ---------------------------------------------------------------------
# Topline metrics
# ---------------------------------------------------------------------
m = topline_metrics(pubs, internal)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Number of labs", f"{m['n_labs']:,}")
k2.metric("Publications (2019–2023)", f"{m['n_pubs_total_19_23']:,}")
k3.metric("Publications with a lab (2019–2023)", f"{m['n_pubs_lab_19_23']:,}")
k4.metric("% covered by labs", f"{m['%_covered_by_labs']*100:.1f}%")

st.divider()

# ---------------------------------------------------------------------
# Per-lab overview table (2019–2023 baseline)
# ---------------------------------------------------------------------
st.subheader("Per-lab overview (2019–2023)")

summary = lab_summary_table_from_internal(internal, year_min=YEAR_START, year_max=YEAR_END).copy()

# display-friendly %
summary["share_pct_display"]   = summary["share_of_dataset_works"] * 100.0
summary["lue_pct_display"]     = summary["lue_pct"] * 100.0
summary["intl_pct_display"]    = summary["intl_pct"] * 100.0
summary["company_pct_display"] = summary["company_pct"] * 100.0

max_share = float(summary["share_pct_display"].max() or 1.0)
max_lue   = float(summary["lue_pct_display"].max() or 1.0)
max_intl  = float(summary["intl_pct_display"].max() or 1.0)
max_comp  = float(summary["company_pct_display"].max() or 1.0)

default_cols = [
    "lab_name", "pubs_19_23", "share_pct_display",
    "lue_pct_display", "intl_pct_display", "company_pct_display",
    "avg_fwci",
]
column_order = default_cols + ["openalex_ui_url", "ror_url"]  # visible in column menu (hidden by default)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
    column_order=column_order,
    column_config={
        "lab_name": st.column_config.TextColumn("Lab"),
        "pubs_19_23": st.column_config.NumberColumn("Publications", format="%.0f"),
        "share_pct_display":    st.column_config.ProgressColumn("% Université de Lorraine", format="%.1f %%", min_value=0.0, max_value=max_share),
        "lue_pct_display":      st.column_config.ProgressColumn("% of pubs LUE",            format="%.1f %%", min_value=0.0, max_value=max_lue),
        "intl_pct_display":     st.column_config.ProgressColumn("% international",           format="%.1f %%", min_value=0.0, max_value=max_intl),
        "company_pct_display":  st.column_config.ProgressColumn("% with company",            format="%.1f %%", min_value=0.0, max_value=max_comp),
        "avg_fwci": st.column_config.NumberColumn("Avg. FWCI", format="%.3f"),
        "openalex_ui_url": st.column_config.LinkColumn("See in OpenAlex"),
        "ror_url":         st.column_config.LinkColumn("See in ROR"),
    },
)

# ---------------------------------------------------------------------
# Year filter (for plots + OpenAlex links)
# ---------------------------------------------------------------------
st.markdown("### Year filter")
years_all = list(range(YEAR_START, YEAR_END + 1))
years_sel = st.multiselect("Filter years (affects the plots and OpenAlex links)", years_all, default=years_all)
if not years_sel:
    st.warning("Select at least one year.")
    st.stop()
year_min, year_max = min(years_sel), max(years_sel)

# rebuild OpenAlex links for the chosen window
summary_links = lab_summary_table_from_internal(internal, year_min=year_min, year_max=year_max)[["lab_name", "openalex_ui_url"]]
summary = summary.drop(columns=["openalex_ui_url"]).merge(summary_links, on="lab_name", how="left")

st.divider()

# ---------------------------------------------------------------------
# Compare two labs
# ---------------------------------------------------------------------
st.subheader("Compare two labs")

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

# field distributions for selected years
lfc = lab_field_counts(pubs, years=years_sel)
left_df  = lfc[lfc["lab_ror"].eq(left_ror)].copy()
right_df = lfc[lfc["lab_ror"].eq(right_ror)].copy()

catalogue = all_fields_order()  # full fixed list from constants
xmax = float(pd.concat([left_df["count"], right_df["count"]]).max() or 0)  # shared x max for volume

pL, pR = st.columns(2, gap="large")
for side, title, df_lab in [(pL, left_label, left_df), (pR, right_label, right_df)]:
    with side:
        st.markdown(f"### {title}")
        if df_lab.empty:
            st.info("No publications for this lab in the selected period.")
            continue

        st.markdown("**Field distribution (volume)**")
        st.altair_chart(
            field_mix_bars(
                df_lab, value_col="count", percent=False,
                xmax=xmax, enforce_order_from=catalogue, show_y_labels=True, width=560
            ),
            use_container_width=True,
        )

        st.markdown("**Field distribution (% of lab works)**")
        st.altair_chart(
            field_mix_bars(
                df_lab, value_col="count", percent=True,
                xmax=1.0, enforce_order_from=catalogue, show_y_labels=True, width=560
            ),
            use_container_width=True,
        )

# ---------------------------------------------------------------------
# Collaboration between the two labs
# ---------------------------------------------------------------------
st.divider()
st.subheader("Collaboration between the selected labs")

elabs = explode_labs(pubs)
e2 = elabs[elabs["year"].isin(years_sel) & elabs["lab_ror"].isin([left_ror, right_ror])]
both_works = e2.groupby("openalex_id")["lab_ror"].nunique().reset_index(name="n_labs")
both_ids = set(both_works.loc[both_works["n_labs"] == 2, "openalex_id"])
copubs = pubs[pubs["openalex_id"].isin(both_ids) & pubs["year"].isin(years_sel)].copy()

if copubs.empty:
    st.info("No co-publications between these labs for the selected years.")
else:
    # --- KPIs
    total = int(copubs["openalex_id"].nunique())
    lue_count = int(copubs.get("in_lue", pd.Series([False]*len(copubs))).fillna(False).astype(bool).sum())
    lue_pct = (lue_count / total) if total else 0.0
    avg_fwci = float(pd.to_numeric(copubs.get("fwci"), errors="coerce").mean() or 0.0)

    # International / company flags from normalized columns (inst_countries / inst_types)
    def _has_non_fr(c):
        if c is None:
            return False
        toks = [t.strip().upper() for t in str(c).replace(";", "|").split("|") if t.strip()]
        return any(t and t != "FR" for t in toks)

    def _has_company(t):
        if t is None:
            return False
        toks = [t.strip().lower() for t in str(t).replace(";", "|").split("|") if t.strip()]
        return any("company" in t for t in toks)

    intl = int(copubs.get("inst_countries", pd.Series([None]*len(copubs))).map(_has_non_fr).sum())
    intl_pct = (intl / total) if total else 0.0
    comp = int(copubs.get("inst_types", pd.Series([None]*len(copubs))).map(_has_company).sum())
    comp_pct = (comp / total) if total else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Co-publications", f"{total:,}")
    k2.metric("ISITE (count / %)", f"{lue_count:,} / {lue_pct*100:.1f}%")
    k3.metric("Avg. FWCI", f"{avg_fwci:.2f}")
    k4.metric("% international", f"{intl_pct*100:.1f}%")
    k5.metric("% with company", f"{comp_pct*100:.1f}%")

    # --- Stacked vertical bars: document types by year
    # Use normalized 'pub_type' if available
    type_col = "pub_type" if "pub_type" in copubs.columns else ("Publication Type" if "Publication Type" in copubs.columns else None)
    cop = copubs.copy()
    if type_col is not None:
        cop["doc_type"] = cop[type_col].str.lower().map({
            "article": "article",
            "review": "review",
            "book-chapter": "book-chapter",
            "chapter": "book-chapter",
            "book": "book",
        }).fillna("other")
    else:
        cop["doc_type"] = "other"

    doc_order = ["article", "review", "book-chapter", "book", "other"]
    year_counts = cop.groupby(["year", "doc_type"])["openalex_id"].nunique().reset_index(name="count")

    chart = (
        alt.Chart(year_counts)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Year", sort=sorted(years_sel)),
            y=alt.Y("count:Q", title="Co-publications"),
            color=alt.Color("doc_type:N", title="Type", sort=doc_order),
            tooltip=[alt.Tooltip("year:O"), alt.Tooltip("doc_type:N"), alt.Tooltip("count:Q", format=",")],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)

    # ---------- TOP AUTHORS (co-pubs) + enrich ----------
    rows = []
    LEAD = re.compile(r"^\[\d+\]\s*")
    for _, r in copubs[["openalex_id", "authors", "authors_id", "fwci"]].iterrows():
        names = [LEAD.sub("", x).strip() for x in str(r["authors"] or "").split("|") if x.strip()]
        ids   = [LEAD.sub("", x).strip() for x in str(r["authors_id"] or "").split("|") if x.strip()]
        if len(names) < len(ids):
            names += [""] * (len(ids) - len(names))
        if len(ids) < len(names):
            ids += [""] * (len(names) - len(ids))
        for nm, aid in zip(names, ids):
            if not aid and not nm:
                continue
            rows.append({"author_id": aid, "Author": nm, "openalex_id": r["openalex_id"], "fwci": r["fwci"]})
    ea = pd.DataFrame(rows)

    top_counts = (
        ea.groupby(["author_id", "Author"], as_index=False)
          .agg(Publications=("openalex_id", "nunique"))
          .sort_values("Publications", ascending=False)
    )

    g = author_global_metrics(pubs).rename(columns={
        "author_id": "author_id",
        "author_name": "Author",
        "total_pubs": "Total publications",
        "avg_fwci_overall": "Avg. FWCI (overall)",
        "labs_concat": "Lab(s)",
    })

    lk = load_authors_lookup()
    if lk is not None and not lk.empty:
        lk = lk.rename(columns={
            "author_id": "author_id",
            "orcid": "ORCID",
            "is_lorraine": "Is Lorraine",
            "labs_from_dict": "Lab(s) (dict)",
        })[["author_id", "ORCID", "Is Lorraine", "Lab(s) (dict)"]]
    else:
        lk = pd.DataFrame(columns=["author_id", "ORCID", "Is Lorraine", "Lab(s) (dict)"])

    top_authors = top_counts.merge(g, on=["author_id", "Author"], how="left").merge(lk, on="author_id", how="left")

    # prefer dict labs if present; otherwise keep computed labs
    if "Lab(s) (dict)" in top_authors.columns:
        top_authors["Lab(s)"] = top_authors["Lab(s) (dict)"].fillna(top_authors.get("Lab(s)", ""))
        top_authors = top_authors.drop(columns=["Lab(s) (dict)"], errors="ignore")

    top_authors = top_authors.sort_values(["Publications", "Avg. FWCI (overall)"], ascending=[False, False]).head(25)

    st.markdown("**Top authors in these co-publications**")
    st.dataframe(
        top_authors[["Author", "author_id", "ORCID", "Publications", "Total publications", "Avg. FWCI (overall)", "Is Lorraine", "Lab(s)"]],
        use_container_width=True, hide_index=True,
        column_config={
            "author_id": st.column_config.TextColumn("Author ID"),
            "Publications": st.column_config.NumberColumn(format="%.0f"),
            "Total publications": st.column_config.NumberColumn(format="%.0f"),
            "Avg. FWCI (overall)": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    # ---------- ALL CO-PUBLICATIONS (rich, exportable) ----------
    want = [
        ("openalex_id", "OpenAlex ID"),
        ("DOI", "DOI"),
        ("pub_type", "Publication Type"),
        ("year", "Publication Year"),
        ("title", "Title"),
        ("citation_count", "Citation Count"),
        ("fwci", "Field-Weighted Citation Impact"),
        ("in_lue", "In LUE"),
        ("All Topics", "All Topics"),
        ("All Subfields", "All Subfields"),
        ("all_fields", "All Fields"),
        ("All Domains", "All Domains"),
    ]
    cols = [c for c, _ in want if c in copubs.columns]
    rename = {c: new for c, new in want if c in copubs.columns}
    copub_table = copubs[cols].rename(columns=rename).drop_duplicates()

    st.markdown("**All co-publications (exportable)**")
    st.dataframe(copub_table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        data=copub_table.to_csv(index=False).encode("utf-8"),
        file_name=f"copubs_{left_label}_{right_label}_{year_min}-{year_max}.csv",
        mime="text/csv",
    )

st.divider()
st.markdown("Use the sidebar to return to **Home** and switch dashboards.")
