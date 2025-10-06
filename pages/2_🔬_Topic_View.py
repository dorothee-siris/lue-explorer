# pages/2_🔬_Topic_View.py
from __future__ import annotations

import math
import pandas as pd
import streamlit as st

from lib.io import load_all_core
from lib.taxonomy import build_taxonomy_lookups
from lib.charts import plot_whisker, plot_bar_with_counts
from lib.tables import show_table, progressify
from lib.exports import download_csv_button
from lib.transforms import parse_id_count_series


st.set_page_config(page_title="🔬 Topic View", layout="wide")


# ---------------------------- small helpers ----------------------------
def _pct_0_100(x) -> float:
    try:
        return float(x) * 100.0
    except Exception:
        return float("nan")


def _explode(cell: str, sep="|"):
    if pd.isna(cell) or cell is None:
        return []
    return [c.strip() for c in str(cell).replace(";", sep).split(sep) if c.strip()]


def _parse_pairs_to(df_or_cell, id_type=str, val_type=float) -> pd.DataFrame:
    """Accept a single cell or a Series of 'id (value) | id (value)' -> DataFrame[id,value]."""
    s = df_or_cell if isinstance(df_or_cell, pd.Series) else pd.Series([df_or_cell])
    out = parse_id_count_series(s, id_type=id_type, count_type=val_type)
    if out.empty:
        return pd.DataFrame(columns=["id", "value"])
    return out.rename(columns={"count": "value"})[["id", "value"]].copy()


def _field_domain(field_name: str, lookups: dict) -> str:
    for d, fields in lookups["fields_by_domain"].items():
        if field_name in fields:
            return d
    return "Other"


def _join_lab_names(labs_df: pd.DataFrame, units_df: pd.DataFrame) -> pd.DataFrame:
    name_map = units_df.set_index("ROR")["Unit Name"].to_dict()
    out = labs_df.copy()
    out["Lab"] = out["ROR"].map(name_map).fillna(out["ROR"])
    return out


def _pad_to(lst, n, fill=None):
    lst = list(lst or [])
    return (lst + [fill] * max(0, n - len(lst)))[:n]


# ---------------------------- load core ----------------------------
core = load_all_core()  # pubs, authors, fields, domains, partners, units, topics
domains = core["domains"].copy()
fields = core["fields"].copy()
partners = core["partners"].copy()
units = core["units"].copy()
authors = core["authors"].copy()
topics = core["topics"].copy()
lookups = build_taxonomy_lookups()  # domain_order, fields_by_domain, subfields_by_field, etc.

st.title("🔬 Topic View")
st.caption("Domain palette drives colors. Labels always visible. % charts show counts in the left gutter. FWCI whiskers use min/Q1/median/Q3/max.")


# ============================================================================
# 1) DOMAIN OVERVIEW TABLE  (renames + visibility rules)
# ============================================================================
st.subheader("Domain overview")

dom_df = pd.DataFrame({
    "Domain": domains["Domain name"],
    "Publications": domains["Pubs"],
    "% UL": domains["% Pubs (uni level)"].apply(_pct_0_100),
    "Pubs LUE": domains.get("Pubs LUE", pd.Series([math.nan] * len(domains))),
    "% Pubs LUE": domains["% Pubs LUE (domain level)"].apply(_pct_0_100),
    "% PPtop10%": domains["% PPtop10% (domain level)"].apply(_pct_0_100),
    "% PPtop1%": domains["% PPtop1% (domain level)"].apply(_pct_0_100),
    "% internal collaboration": domains["% internal collaboration"].apply(_pct_0_100),
    "% international": domains["% international"].apply(_pct_0_100),
    # hidden by default (available behind the toggle)
    "Avg. FWCI (France)": domains.get("Avg FWCI (France)", pd.Series([math.nan] * len(domains))),
    "% Pubs LUE (uni level)": domains["% Pubs LUE (uni level)"].apply(_pct_0_100),
    "% PPtop10% (uni level)": domains["% PPtop10% (uni level)"].apply(_pct_0_100),
    "% PPtop1% (uni level)": domains["% PPtop1% (uni level)"].apply(_pct_0_100),
    "See in OpenAlex": domains.get("See in OpenAlex", ""),
})

# Canonical order — keep only domains present; no extra blank rows
dom_df["__key"] = dom_df["Domain"].astype(str).str.strip()
present = set(dom_df["__key"])
order_clean = [d.strip() for d in lookups.get("domain_order", []) if d and d.strip() in present]
if order_clean:
    dom_df = dom_df.set_index("__key").reindex(order_clean).reset_index(drop=True)

# This list drives both the table and the selectbox below
domain_options = dom_df["Domain"].tolist()

visible_cols = ["Domain", "Publications", "% UL", "Pubs LUE", "% Pubs LUE",
                "% PPtop10%", "% PPtop1%", "% internal collaboration", "% international"]
advanced_cols = visible_cols + ["Avg. FWCI (France)", "% Pubs LUE (uni level)",
                                "% PPtop10% (uni level)", "% PPtop1% (uni level)", "See in OpenAlex"]

show_adv = st.toggle("Show advanced columns", False)
cfg = progressify(dom_df, [
    "% UL", "% Pubs LUE", "% PPtop10%", "% PPtop1%", "% internal collaboration", "% international",
    "% Pubs LUE (uni level)", "% PPtop10% (uni level)", "% PPtop1% (uni level)"
])

show_table(dom_df[advanced_cols if show_adv else visible_cols], column_config=cfg)
download_csv_button(dom_df, "Download domain overview (CSV)", "domains_overview.csv")


# ============================================================================
# 2) DOMAIN FWCI WHISKERS (no log scale)
# ============================================================================
st.subheader("FWCI (France) distribution by domain")
qcols = {"min": "FWCI_FR min", "q1": "FWCI_FR Q1", "q2": "FWCI_FR Q2", "q3": "FWCI_FR Q3", "max": "FWCI_FR max"}

# Pass only the columns the chart needs (avoid colon headers in tooltips)
_whisker_df = domains[["Domain name", qcols["min"], qcols["q1"], qcols["q2"], qcols["q3"], qcols["max"]]].copy()

st.altair_chart(
    plot_whisker(
        _whisker_df,
        label_col="Domain name",
        qcols=qcols,
        domain_col="Domain name",
        title="FWCI (France) — min / Q1 / median / Q3 / max",
        order=domain_options,  # same order used in the table
    ),
    use_container_width=True,
)


# ============================================================================
# 3) DRILLDOWN BY DOMAIN
# ============================================================================
st.subheader("Drill down by domain")
sel_domain = st.selectbox("Pick a domain", options=domain_options, index=0)
drow = domains.loc[domains["Domain name"] == sel_domain].iloc[0]

# --- Labs contribution (left) + FWCI whiskers for the same labs (right)
st.markdown("##### Labs contribution and impact")
c1, c2 = st.columns(2)

labs_count = _parse_pairs_to(drow["By lab: count"], id_type=str, val_type=int).rename(columns={"id": "ROR", "value": "count"})
labs_share = _parse_pairs_to(drow["By lab: % of domain pubs"], id_type=str, val_type=float).rename(columns={"id": "ROR", "value": "share"})
labs = labs_count.merge(labs_share, on="ROR", how="outer").fillna(0.0)
labs["share_pct"] = labs["share"].apply(_pct_0_100)
labs = labs[labs["share_pct"] >= 2.0].copy()  # only labs > 2%
labs = _join_lab_names(labs, units)
labs["domain_name"] = sel_domain
labs = labs.sort_values("share", ascending=False).reset_index(drop=True)
lab_order = labs["Lab"].tolist()

with c1:
    st.altair_chart(
        plot_bar_with_counts(
            labs.rename(columns={"Lab": "Label"}),
            label_col="Label",
            value_col="share_pct",
            count_col="count",
            domain_col="domain_name",
            title="% of domain publications",
            order=lab_order,
        ),
        use_container_width=True,
    )

# Lab FWCI whiskers (same lab order)
def _lab_q(col):
    return _parse_pairs_to(drow[col], id_type=str, val_type=float).rename(columns={"id": "ROR"}).set_index("ROR")["value"]

if all(k in drow for k in ["By lab: FWCI_FR min", "By lab: FWCI_FR Q1", "By lab: FWCI_FR Q2", "By lab: FWCI_FR Q3", "By lab: FWCI_FR max"]):
    qdf = pd.DataFrame({
        "ROR": labs["ROR"].tolist(),
        "min": _lab_q("By lab: FWCI_FR min").reindex(labs["ROR"]).values,
        "q1":  _lab_q("By lab: FWCI_FR Q1").reindex(labs["ROR"]).values,
        "q2":  _lab_q("By lab: FWCI_FR Q2").reindex(labs["ROR"]).values,
        "q3":  _lab_q("By lab: FWCI_FR Q3").reindex(labs["ROR"]).values,
        "max": _lab_q("By lab: FWCI_FR max").reindex(labs["ROR"]).values,
    })
    qdf = _join_lab_names(qdf, units)
    qdf["domain_name"] = sel_domain
    with c2:
        st.altair_chart(
            plot_whisker(
                qdf.rename(columns={"Lab": "Label"}),
                label_col="Label",
                qcols={"min": "min", "q1": "q1", "q2": "q2", "q3": "q3", "max": "max"},
                domain_col="domain_name",
                title="FWCI (France) by lab",
                order=lab_order,
            ),
            use_container_width=True,
        )
else:
    with c2:
        st.info("FWCI whiskers by lab unavailable for this domain in the source file.")


# --- Top partners (FR first, then International — not side-by-side)
st.markdown("##### Top partners")

# Top 20 French partners (no parent institution)
st.markdown("**Top 20 French partners (no parent institution)**")
fr_names = _explode(drow["Top 20 FR partners (name)"])
fr_types = _explode(drow["Top 20 FR partners (type)"])
fr_copubs = _pad_to([int(x) if x else 0 for x in _explode(drow["Top 20 FR partners (totals copubs in this domain)"])], len(fr_names), 0)
# Correct % column to use: “% of UL total copubs” for this partner (render as progress)
fr_pct_ul = _pad_to([float(x) * 100.0 for x in _explode(drow["Top 20 FR partners (% of UL total copubs)"])], len(fr_names), 0.0)

fr_df = pd.DataFrame({
    "Partner": fr_names,
    "Type": fr_types[:len(fr_names)],
    "Copubs in this domain": fr_copubs,
    "Share of UL–partner copubs (this domain)": fr_pct_ul,  # renamed column
})
show_table(fr_df, column_config=progressify(fr_df, ["Share of UL–partner copubs (this domain)"]))

# Top 20 International partners
st.markdown("**Top 20 international partners**")
int_names = _explode(drow["Top 20 int partners (name)"])
int_types = _explode(drow["Top 20 int partners (type)"])
int_ctry  = _explode(drow["Top 20 int partners (country)"])
int_copubs = _pad_to([int(x) if x else 0 for x in _explode(drow["Top 20 int partners (totals copubs in this domain)"])], len(int_names), 0)
int_pct_ul = _pad_to([float(x) * 100.0 for x in _explode(drow["Top 20 int partners (% of UL total copubs)"])], len(int_names), 0.0)

int_df = pd.DataFrame({
    "Partner": int_names,
    "Type": int_types[:len(int_names)],
    "Country": int_ctry[:len(int_names)],
    "Copubs in this domain": int_copubs,
    "Share of UL–partner copubs (this domain)": int_pct_ul,
})
show_table(int_df, column_config=progressify(int_df, ["Share of UL–partner copubs (this domain)"]))


# --- Top 20 authors (with added “Total pubs (at UL)” and derived %; ORCID/ID hidden by default)
st.markdown("##### Top 20 authors")

auth_names = _explode(drow["Top 20 authors (name)"])
if not auth_names:
    st.info("No authors listed for this domain.")
else:
    n = len(auth_names)

    auth_pubs        = _pad_to([int(x) if x else 0 for x in _explode(drow["Top 20 authors (pubs)"])], n, 0)
    auth_fwci        = _pad_to([float(x) if x else float("nan") for x in _explode(drow["Top 20 authors (Average FWCI_FR)"])], n, float("nan"))
    auth_top10       = _pad_to([int(x) if x else 0 for x in _explode(drow["Top 20 authors (PPtop10% Count)"])], n, 0)
    auth_top1        = _pad_to([int(x) if x else 0 for x in _explode(drow["Top 20 authors (PPtop1% Count)"])], n, 0)
    auth_is_lorraine = _pad_to([str(x).strip().lower() == "true" for x in _explode(drow["Top 20 authors (Is Lorraine)"])], n, False)
    auth_orcid       = _pad_to(_explode(drow.get("Top 20 authors (Orcid)", "")), n, "")
    auth_ids         = _pad_to(_explode(drow.get("Top 20 authors (ID)", "")), n, "")

    top_df = pd.DataFrame({
        "Author": auth_names,
        "Pubs in this domain": auth_pubs,
        "Avg. FWCI (France)": auth_fwci,
        "PPtop10% Count": auth_top10,
        "PPtop1% Count": auth_top1,
        "Is Lorraine": auth_is_lorraine,
        "ORCID": auth_orcid,
        "Author ID": auth_ids,
    })

    # Bring “Total pubs (at UL)” from authors table via ORCID first, then Author ID
    authors_long = []
    if "ORCID" in authors.columns:
        for _, r in authors.iterrows():
            for o in _explode(r["ORCID"], sep="|"):
                authors_long.append({
                    "Key": o,
                    "Kind": "ORCID",
                    "Total pubs (at UL)": r.get("Publications (unique)", pd.NA)
                })
    if "Author ID" in authors.columns:
        for _, r in authors.iterrows():
            for a in _explode(r["Author ID"], sep="|"):
                authors_long.append({
                    "Key": a,
                    "Kind": "Author ID",
                    "Total pubs (at UL)": r.get("Publications (unique)", pd.NA)
                })
    authors_long = pd.DataFrame(authors_long)

    enriched = top_df.copy()
    enriched["Total pubs (at UL)"] = pd.NA
    if not authors_long.empty:
        # ORCID match
        m_orcid = enriched.merge(
            authors_long[authors_long["Kind"] == "ORCID"][["Key", "Total pubs (at UL)"]],
            left_on="ORCID", right_on="Key", how="left"
        ).drop(columns=["Key"])
        enriched["Total pubs (at UL)"] = m_orcid["Total pubs (at UL)"]
        # Backfill by Author ID
        miss = enriched["Total pubs (at UL)"].isna()
        if miss.any():
            m_id = enriched[miss].merge(
                authors_long[authors_long["Kind"] == "Author ID"][["Key", "Total pubs (at UL)"]],
                left_on="Author ID", right_on="Key", how="left"
            ).drop(columns=["Key"])
            enriched.loc[miss, "Total pubs (at UL)"] = m_id["Total pubs (at UL)"].values

    enriched["Total pubs (at UL)"] = pd.to_numeric(enriched["Total pubs (at UL)"], errors="coerce")
    denom = enriched["Total pubs (at UL)"].replace(0, pd.NA)
    enriched["% Pubs in this domain"] = ((enriched["Pubs in this domain"] / denom).fillna(0.0) * 100.0).clip(lower=0)

    auth_basic = ["Author", "Pubs in this domain", "Total pubs (at UL)", "% Pubs in this domain",
                  "Avg. FWCI (France)", "PPtop10% Count", "PPtop1% Count", "Is Lorraine"]
    auth_all = auth_basic + ["ORCID", "Author ID"]

    show_ids = st.toggle("Show ORCID and Author ID", False, key="authors_ids_toggle")
    show_table(enriched[auth_all if show_ids else auth_basic],
               column_config=progressify(enriched, ["% Pubs in this domain"]))


# --- Field distribution inside the selected domain
st.markdown("##### Thematic shape — fields within this domain")

fld_counts = _parse_pairs_to(drow["By field: count"], id_type=str, val_type=int).rename(columns={"id": "Field ID", "value": "count"})
fld_pct    = _parse_pairs_to(drow["By field: % of domain pubs"], id_type=str, val_type=float).rename(columns={"id": "Field ID", "value": "pct"})
# Map Field ID -> Field name
id2name_field = topics.drop_duplicates(["field_id", "field_name"]).set_index("field_id")["field_name"].to_dict()
fld = fld_counts.merge(fld_pct, on="Field ID", how="outer").fillna(0.0)
fld["Field"] = fld["Field ID"].apply(lambda x: id2name_field.get(int(str(x)), str(x)))
fld["pct"] = fld["pct"].apply(_pct_0_100)
fld["domain_name"] = sel_domain

# Canonical field order for this domain
canon_fields = [f for f in lookups["fields_by_domain"].get(sel_domain, []) if f in fld["Field"].tolist()]
st.altair_chart(
    plot_bar_with_counts(
        fld.rename(columns={"Field": "Label"}),
        label_col="Label",
        value_col="pct",
        count_col="count",
        domain_col="domain_name",
        title="% of domain publications by field",
        order=canon_fields,
    ),
    use_container_width=True,
)


# ============================================================================
# 4) SUBFIELD SHAPE COMPARISON (field-level) — UL vs another institution
# ============================================================================
st.subheader("Compare subfield shape within a field")

# Field list (sorted by field_id)
all_fields_sorted = topics.drop_duplicates(["field_id", "field_name"]).sort_values("field_id")["field_name"].tolist()
sel_field = st.selectbox("Select a field", options=all_fields_sorted, index=0)

left_options = ["Université de Lorraine"] + sorted(partners["Institution name"].dropna().unique().tolist())
right_default = "Université de Strasbourg"
right_index = left_options.index(right_default) if right_default in left_options else 0

cL, cR = st.columns(2)
with cL:
    inst_left = st.selectbox("Institution (left)", options=left_options, index=0, key="inst_left")
with cR:
    inst_right = st.selectbox("Institution (right)", options=left_options, index=right_index, key="inst_right")

mode = st.radio("Scale", ["Relative to selected field", "Absolute (institution-level)"], index=0, horizontal=True)


def ul_subfields_for_field(field_name: str, as_relative: bool) -> pd.DataFrame:
    row = fields.loc[fields["Field name"] == field_name]
    if row.empty:
        return pd.DataFrame(columns=["Subfield", "count", "pct"])
    r = row.iloc[0]
    counts = _parse_pairs_to(r["By subfield: count"], id_type=str, val_type=int).rename(columns={"id": "subfield_id", "value": "count"})
    pct_rel = _parse_pairs_to(r["By subfield: % of field pubs"], id_type=str, val_type=float).rename(columns={"id": "subfield_id", "value": "pct_rel"})
    df = counts.merge(pct_rel, on="subfield_id", how="outer").fillna(0.0)
    sid2name = topics.drop_duplicates(["subfield_id", "subfield_name"]).set_index("subfield_id")["subfield_name"].to_dict()
    df["Subfield"] = df["subfield_id"].apply(lambda x: sid2name.get(int(str(x)), str(x)))
    if as_relative:
        df["pct"] = df["pct_rel"].apply(_pct_0_100)
    else:
        total_ul = int(domains["Pubs"].sum())
        df["pct"] = df["count"].astype(float) / max(total_ul, 1) * 100.0
    return df[["Subfield", "count", "pct"]]


def partner_subfields_for_field(partner_name: str, field_name: str, as_relative: bool) -> pd.DataFrame:
    prow = partners.loc[partners["Institution name"] == partner_name]
    if prow.empty:
        return pd.DataFrame(columns=["Subfield", "count", "pct"])
    pr = prow.iloc[0]
    sub_counts = _parse_pairs_to(pr["By subfield: count"], id_type=str, val_type=int).rename(columns={"id": "subfield_id", "value": "count"})
    field_sid = topics.loc[topics["field_name"] == field_name, "subfield_id"].unique().tolist()
    sub_counts = sub_counts[sub_counts["subfield_id"].astype(int).isin([int(x) for x in field_sid])].copy()
    if as_relative:
        denom = max(int(sub_counts["count"].sum()), 1)
    else:
        denom = max(int(pr["Copublications"]), 1)
    sub_counts["pct"] = sub_counts["count"].astype(float) / denom * 100.0
    sid2name = topics.drop_duplicates(["subfield_id", "subfield_name"]).set_index("subfield_id")["subfield_name"].to_dict()
    sub_counts["Subfield"] = sub_counts["subfield_id"].apply(lambda x: sid2name.get(int(str(x)), str(x)))
    return sub_counts[["Subfield", "count", "pct"]]


def _subfield_chart(df: pd.DataFrame, field_name: str):
    dname = _field_domain(field_name, lookups)
    df = df.copy()
    df["domain_name"] = dname
    # canonical subfield order for this field
    sf_canon = [s for s in lookups["subfields_by_field"].get(field_name, []) if s in df["Subfield"].tolist()]
    return plot_bar_with_counts(
        df.rename(columns={"Subfield": "Label"}),
        label_col="Label",
        value_col="pct",
        count_col="count",
        domain_col="domain_name",
        title=f"{field_name} — subfield distribution (%)",
        order=sf_canon,
    )


left_df = (
    ul_subfields_for_field(sel_field, as_relative=(mode == "Relative to selected field"))
    if inst_left == "Université de Lorraine"
    else partner_subfields_for_field(inst_left, sel_field, as_relative=(mode == "Relative to selected field"))
)
right_df = (
    ul_subfields_for_field(sel_field, as_relative=(mode == "Relative to selected field"))
    if inst_right == "Université de Lorraine"
    else partner_subfields_for_field(inst_right, sel_field, as_relative=(mode == "Relative to selected field"))
)

cc1, cc2 = st.columns(2)
with cc1:
    st.subheader(inst_left)
    st.altair_chart(_subfield_chart(left_df, sel_field), use_container_width=True)
with cc2:
    st.subheader(inst_right)
    st.altair_chart(_subfield_chart(right_df, sel_field), use_container_width=True)

# Mini collaboration snapshot (UL, selected field) — optional
st.caption("Collaboration snapshot (UL, selected field)")
frow = fields.loc[fields["Field name"] == sel_field]
if not frow.empty:
    rr = frow.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("% internal collaboration", f"{_pct_0_100(rr['% internal collaboration']):.1f}%")
    c2.metric("% international", f"{_pct_0_100(rr['% international']):.1f}%")
    c3.metric("% industrial", f"{_pct_0_100(rr['% industrial']):.1f}%")
else:
    st.info("Field-level collaboration indicators unavailable for this field.")
