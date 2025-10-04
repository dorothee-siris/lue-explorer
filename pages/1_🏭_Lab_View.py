# pages/1_🏭_Lab_View.py
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from lib.data_io import (
    load_core,
    load_internal,
    load_topics,
    explode_labs,
    explode_authors,
    author_global_metrics,
)

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(page_title="Lab View", page_icon="🏭", layout="wide")
st.title("🏭 Lab View")

YEAR_MIN, YEAR_MAX = 2019, 2023
YEARS_1923 = list(range(YEAR_MIN, YEAR_MAX + 1))

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
LEAD = re.compile(r"^\[\d+\]\s*")

def split_positions(s: object) -> List[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    toks = [LEAD.sub("", x).strip() for x in str(s).split("|")]
    return [t for t in toks if t]

def build_topics_hierarchy(topics: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    keep = ["domain_id","domain_name","field_id","field_name","subfield_id","subfield_name"]
    t = topics[keep].drop_duplicates().copy()
    domains = t[["domain_id","domain_name"]].drop_duplicates().sort_values("domain_name")
    domain_order = domains["domain_name"].tolist()
    fields = t[["domain_name","field_id","field_name"]].drop_duplicates().sort_values(["domain_name","field_name"])
    field_order = fields["field_name"].tolist()
    return t, domain_order, field_order

def make_domain_palette(domain_names: List[str]) -> Dict[str, str]:
    palette = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
        "#4c78a8","#f58518","#54a24b","#e45756","#b279a2",
    ]
    return {dn: palette[i % len(palette)] for i, dn in enumerate(domain_names)}

def prepare_core() -> pd.DataFrame:
    """Load UL core and standardise dtypes. Keep *all* works; we’ll filter later."""
    pubs = load_core().copy()

    # numeric/boolean coercions without future warnings
    if "year" in pubs: pubs["year"] = pd.to_numeric(pubs["year"], errors="coerce")
    if "citation_count" in pubs: pubs["citation_count"] = pd.to_numeric(pubs["citation_count"], errors="coerce")
    if "fwci_fr" in pubs: pubs["fwci_fr"] = pd.to_numeric(pubs["fwci_fr"], errors="coerce")

    for c in ["in_lue", "is_pp10_field", "is_pp1_field"]:
        if c in pubs.columns:
            pubs[c] = pubs[c].astype("boolean").fillna(False).astype(bool)

    return pubs

def pubs_with_labs(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["labs_rors"].fillna("").astype(str).str.strip().ne("") if "labs_rors" in df.columns else False
    return df.loc[mask].copy()

def filter_years(df: pd.DataFrame, years: List[int]) -> pd.DataFrame:
    return df[df["year"].isin(years)].copy() if "year" in df.columns else df.iloc[0:0].copy()

def compute_pub_level_flags(pubs: pd.DataFrame) -> pd.DataFrame:
    """Compute 'has_international' and 'n_labs' flags per work."""
    df = pubs.copy()

    if "inst_countries" in df.columns:
        def has_foreign(countries: object) -> bool:
            cs = {x.upper() for x in split_positions(countries)}
            cs.discard("")
            return any(c and c != "FR" for c in cs)
        df["has_international"] = df["inst_countries"].map(has_foreign)
    else:
        df["has_international"] = False

    def n_labs(s: object) -> int:
        labs = {x for x in split_positions(s) if x}
        return len(labs)
    df["n_labs"] = df["labs_rors"].map(n_labs) if "labs_rors" in df.columns else 0

    return df

def keep_only_labs(internal: pd.DataFrame, pubs_all: pd.DataFrame) -> pd.DataFrame:
    """Keep the 61 'lab' units; fallback to those seen in pubs if 'Unit Type' missing."""
    if "unit_type" in internal.columns:
        labs = internal[internal["unit_type"].astype(str).str.lower().eq("lab")].copy()
    else:
        seen = set(explode_labs(pubs_all)["lab_ror"].dropna().unique())
        labs = internal[internal["lab_ror"].isin(seen)].copy()
    labs = labs.drop_duplicates(subset=["lab_ror"]).sort_values("lab_name")
    return labs

def compute_per_lab_overview(
    pubs_1923_all: pd.DataFrame,
    labs_dict: pd.DataFrame,
    ul_total_works_1923: int,
) -> pd.DataFrame:
    """
    Build the per-lab overview (2019–2023), using dict_internal when available:
      - Publications       -> labs_dict['Total publications'] if present, else count from pubs
      - % UL               -> labs_dict['% of Lorraine production'] (ratio) if present, else pubs/UL_total
      - % LUE              -> labs_dict['count of IS_LUE'] / publications (abs ratio) if present, else compute from pubs
      - % top10/top1       -> from pubs (field-based flags)
      - Avg FWCI (FR)      -> mean fwci_fr per lab from pubs
      - % internal collab  -> from pubs: share of works with >=2 labs
      - % international    -> from labs_dict['International collabs (ratio)'] if present, else pubs
      - Links              -> ROR + OpenAlex (institution id if available)
    """
    # explode lab memberships
    pubs_flags = compute_pub_level_flags(pubs_1923_all)
    el = explode_labs(pubs_flags).dropna(subset=["lab_ror"])

    per_work = pubs_flags[["openalex_id","in_lue","is_pp10_field","is_pp1_field","fwci_fr","has_international","n_labs"]].drop_duplicates("openalex_id")
    x = el.merge(per_work, on="openalex_id", how="left")

    agg = x.groupby("lab_ror", as_index=False).agg(
        pubs=("openalex_id", "nunique"),
        lue_count=("in_lue", "sum"),
        top10_count=("is_pp10_field", "sum"),
        top1_count=("is_pp1_field", "sum"),
        avg_fwci=("fwci_fr", "mean"),
        intl_count=("has_international", "sum"),
        multi_lab_count=("n_labs", lambda s: (s >= 2).sum()),
    )

    # bring dict_internal values where available
    labs = labs_dict.copy()
    # Total publications (dict column can be named differently → mapped in data_io)
    pubs_from_dict = labs.get("pubs_total_internal")
    share_from_dict = labs.get("share_of_ul_internal")  # ratio (e.g., 0.00667)

    out = agg.merge(labs[["lab_ror","lab_name","lab_openalex_id","intl_ratio","lue_count"]], on="lab_ror", how="left")

    # publications
    out["pubs_final"] = np.where(
        pubs_from_dict.notna().values if pubs_from_dict is not None else False,
        labs.set_index("lab_ror").loc[out["lab_ror"], "pubs_total_internal"].to_numpy(),
        out["pubs"].to_numpy(),
    ).astype(float)

    # % UL (ratio)
    if share_from_dict is not None:
        out["share_of_ul"] = labs.set_index("lab_ror").loc[out["lab_ror"], "share_of_ul_internal"].to_numpy()
    else:
        out["share_of_ul"] = (out["pubs_final"] / float(ul_total_works_1923)).replace([np.inf, -np.inf], np.nan)

    # % LUE (abs ratio)
    if "lue_count" in out.columns and out["lue_count"].notna().any():
        out["lue_ratio"] = (pd.to_numeric(out["lue_count"], errors="coerce") / out["pubs_final"]).replace([np.inf, -np.inf], np.nan)
    else:
        out["lue_ratio"] = (out["lue_count"] / out["pubs_final"]).replace([np.inf, -np.inf], np.nan)

    # % top10 / % top1 from pubs
    out["top10_ratio"] = (out["top10_count"] / out["pubs_final"]).replace([np.inf, -np.inf], np.nan)
    out["top1_ratio"]  = (out["top1_count"]  / out["pubs_final"]).replace([np.inf, -np.inf], np.nan)

    # % international: prefer dict ratio, fallback to pubs-based
    out["intl_ratio_calc"] = (out["intl_count"] / out["pubs_final"]).replace([np.inf, -np.inf], np.nan)
    if "intl_ratio" in out.columns and out["intl_ratio"].notna().any():
        out["intl_ratio_final"] = out["intl_ratio"].astype(float)
    else:
        out["intl_ratio_final"] = out["intl_ratio_calc"]

    # % internal collab
    out["internal_collab_ratio"] = (out["multi_lab_count"] / out["pubs_final"]).replace([np.inf, -np.inf], np.nan)

    # Links
    def openalex_institution_url(openalex_id: str | None) -> str:
        if openalex_id:
            return (
                "https://openalex.org/works?"
                f"page=1&filter=authorships.institutions.id:{openalex_id},"
                "type:types/article|types/book-chapter|types/review|types/book,"
                f"publication_year:{YEAR_MIN}-{YEAR_MAX}"
            )
        return ""
    out["openalex_url"] = out["lab_openalex_id"].map(openalex_institution_url)
    out["ror_url"] = out["lab_ror"].map(lambda r: f"https://ror.org/{r}" if r else "")

    # Display columns (percent progress)
    out["share_pct_display"]   = out["share_of_ul"] * 100.0
    out["lue_pct_display"]     = out["lue_ratio"]   * 100.0
    out["top10_pct_display"]   = out["top10_ratio"] * 100.0
    out["top1_pct_display"]    = out["top1_ratio"]  * 100.0
    out["intl_pct_display"]    = out["intl_ratio_final"] * 100.0
    out["internal_collab_pct"] = out["internal_collab_ratio"] * 100.0

    # Sort A–Z
    out = out.sort_values("lab_name")

    # Final selection + renames
    final = out.rename(columns={
        "pubs_final": "pubs",
        "avg_fwci": "avg_fwci",
    })

    cols = [
        "lab_name","lab_ror","pubs",
        "share_pct_display","lue_pct_display","top10_pct_display","top1_pct_display",
        "avg_fwci","internal_collab_pct","intl_pct_display",
        "openalex_url","ror_url",
    ]
    return final[cols]

def ensure_all_fields(df_counts: pd.DataFrame, topics_h: pd.DataFrame) -> pd.DataFrame:
    all_fields = topics_h[["field_id","field_name","domain_name"]].drop_duplicates()
    out = all_fields.merge(df_counts, on="field_id", how="left").fillna({"count": 0})
    out["count"] = out["count"].astype(int)
    out["field"] = out["field_name"]
    out["domain"] = out["domain_name"]
    return out[["field_id","field","domain","count"]]

def field_distribution_for_lab(pubs_all: pd.DataFrame, lab_ror: str, years: List[int], topics_h: pd.DataFrame) -> pd.DataFrame:
    sub = filter_years(pubs_all, years)
    el = explode_labs(sub)
    sub = sub.merge(el, on=["openalex_id","year"], how="left")
    sub = sub[sub["lab_ror"] == lab_ror].copy()
    if sub.empty:
        return ensure_all_fields(pd.DataFrame({"field_id":[],"count":[]}), topics_h)
    sub["primary_field_id"] = pd.to_numeric(sub["primary_field_id"], errors="coerce")
    g = sub.groupby("primary_field_id", as_index=False)["openalex_id"].nunique().rename(columns={"openalex_id":"count","primary_field_id":"field_id"})
    return ensure_all_fields(g, topics_h)

def chart_fields_horizontal(df_fields: pd.DataFrame, field_order: List[str], domain_palette: Dict[str,str], title: str, normalize: bool = False) -> alt.Chart:
    df = df_fields.copy()
    df["field"] = df["field"].astype("category")
    # pandas >= 2.1: set_categories returns a new Categorical (no 'inplace' kwarg)
    df["field"] = df["field"].cat.set_categories(field_order)
    df = df.sort_values("field")

    total = df["count"].sum()
    if normalize and total > 0:
        df["value"] = df["count"] / total * 100.0
        x_title, x_scale, fmt = "% of lab works", alt.Scale(domain=[0,100]), ".1f"
    else:
        df["value"] = df["count"]
        x_title, x_scale, fmt = "Publications", alt.Scale(nice=True), ".0f"

    color_scale = alt.Scale(
        domain=list(domain_palette.keys()),
        range=[domain_palette[d] for d in domain_palette.keys()]
    )

    height = max(26 * 22, 300)

    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("field:N", sort=field_order, axis=alt.Axis(labelLimit=1000, title=None)),
            x=alt.X("value:Q", title=x_title, scale=x_scale),
            color=alt.Color("domain:N", legend=None, scale=color_scale),
            tooltip=[
                alt.Tooltip("field:N", title="Field"),
                alt.Tooltip("domain:N", title="Domain"),
                alt.Tooltip("count:Q", title="Count", format=".0f"),
                alt.Tooltip("value:Q", title="% (if normalized)", format=fmt),
            ],
        )
        .properties(title=title, height=height)
    )

    # count labels near y-axis
    text = (
        alt.Chart(df)
        .mark_text(align="left", baseline="middle", dx=5, size=10)
        .encode(y=alt.Y("field:N", sort=field_order, axis=None), x=alt.value(0), text=alt.Text("count:Q", format=".0f"))
    )

    return text + bars

def compute_copubs_between_labs(pubs_all: pd.DataFrame, lab_left: str, lab_right: str, years: List[int]) -> pd.DataFrame:
    sub = filter_years(pubs_all, years).copy()
    def has_both(labs_s: object) -> bool:
        labs = set(split_positions(labs_s))
        return lab_left in labs and lab_right in labs
    mask = sub["labs_rors"].map(has_both) if "labs_rors" in sub.columns else False
    return sub.loc[mask].copy()

def kpis_for_copubs(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return dict(total=0, lue=0, top10=0, top1=0, avg_fwci=0.0)
    return dict(
        total=int(df["openalex_id"].nunique()),
        lue=int(df["in_lue"].sum()),
        top10=int(df["is_pp10_field"].sum()),
        top1=int(df["is_pp1_field"].sum()),
        avg_fwci=float(pd.to_numeric(df["fwci_fr"], errors="coerce").mean() or 0.0),
    )

def stacked_evolution_by_domain(df: pd.DataFrame, topics: pd.DataFrame, domain_palette: Dict[str,str], years: List[int]) -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame({"year":[], "domain":[],"count":[]})).mark_bar()

    look = topics.drop_duplicates("domain_id")[["domain_id","domain_name"]].rename(columns={"domain_name":"domain"})
    temp = df[["openalex_id","year","primary_domain_id"]].drop_duplicates().merge(
        look, left_on="primary_domain_id", right_on="domain_id", how="left"
    )
    g = temp.groupby(["year","domain"], as_index=False)["openalex_id"].nunique().rename(columns={"openalex_id":"count"})

    domain_order = sorted(domain_palette.keys())
    color_scale = alt.Scale(domain=domain_order, range=[domain_palette[d] for d in domain_order])

    return (
        alt.Chart(g)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Year", sort=[str(y) for y in years]),
            y=alt.Y("count:Q", title="Co-publications", stack="zero"),
            color=alt.Color("domain:N", scale=color_scale),
            tooltip=[alt.Tooltip("year:O"), alt.Tooltip("domain:N"), alt.Tooltip("count:Q", format=".0f")],
        )
        .properties(height=280)
    )

def top_authors_from_copubs(copubs: pd.DataFrame, pubs_all_years: pd.DataFrame, years: List[int]) -> pd.DataFrame:
    rows = []
    if {"authors","authors_id"}.issubset(copubs.columns):
        for _, r in copubs[["openalex_id","authors","authors_id","fwci_fr","year"]].iterrows():
            names = [LEAD.sub("", x).strip() for x in str(r["authors"] or "").split("|") if x.strip()]
            ids   = [LEAD.sub("", x).strip() for x in str(r["authors_id"] or "").split("|") if x.strip()]
            if len(names) < len(ids): names += [""] * (len(ids) - len(names))
            if len(ids)   < len(names): ids   += [""] * (len(names) - len(ids))
            for nm, aid in zip(names, ids):
                if not aid and not nm:
                    continue
                rows.append({"author_id": aid, "Author": nm, "openalex_id": r["openalex_id"], "fwci_fr": r["fwci_fr"], "year": r["year"]})
        ea = pd.DataFrame(rows)
    else:
        ea = explode_authors(copubs).rename(columns={"author_name":"Author"}).merge(
            copubs[["openalex_id","fwci_fr","year"]], on="openalex_id", how="left"
        )

    if ea.empty:
        return pd.DataFrame(columns=["Author","author_id","ORCID","Copubs","FWCI_FR (copubs)","Total pubs (timeframe)","FWCI_FR (overall)","Is Lorraine","Lab(s)"])

    agg = (
        ea.groupby(["author_id","Author"], as_index=False)
          .agg(Copubs=("openalex_id","nunique"), fwci_copubs=("fwci_fr","mean"))
          .rename(columns={"fwci_copubs":"FWCI_FR (copubs)"})
    )

    ea_all = explode_authors(filter_years(pubs_all_years, years)).rename(columns={"author_name":"Author"})
    tot = ea_all.groupby(["author_id","Author"], as_index=False)["openalex_id"].nunique().rename(columns={"openalex_id":"Total pubs (timeframe)"})

    overall = author_global_metrics(pubs_all_years).rename(columns={
        "author_id":"author_id", "author_name":"Author",
        "avg_fwci_overall":"FWCI_FR (overall)", "total_pubs":"Total pubs (overall)"
    })[["author_id","Author","FWCI_FR (overall)","Total pubs (overall)"]]

    # Optional enrichment
    try:
        from lib.data_io import load_authors_lookup
        lk = load_authors_lookup()
    except Exception:
        lk = None
    if lk is not None and not lk.empty:
        lk = lk.rename(columns={"orcid":"ORCID","is_lorraine":"Is Lorraine","labs_from_dict":"Lab(s)"})[
            ["author_id","ORCID","Is Lorraine","Lab(s)"]
        ]
    else:
        lk = pd.DataFrame(columns=["author_id","ORCID","Is Lorraine","Lab(s)"])

    out = agg.merge(tot, on=["author_id","Author"], how="left") \
             .merge(overall, on=["author_id","Author"], how="left") \
             .merge(lk, on="author_id", how="left")

    return out.sort_values(["Copubs","FWCI_FR (copubs)"], ascending=[False, False])

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
with st.spinner("Loading data..."):
    pubs_all = prepare_core()            # all UL works
    topics = load_topics()
    internal_raw = load_internal()

    # Keep only labs (no other internal structures) and sort A–Z
    internal = keep_only_labs(internal_raw, pubs_all)

    topics_h, DOMAIN_ORDER, FIELD_ORDER = build_topics_hierarchy(topics)
    DOMAIN_PALETTE = make_domain_palette(DOMAIN_ORDER)

# Core subsets for the fixed topline/table window
pubs_1923_all = filter_years(pubs_all, YEARS_1923)
pubs_1923_with_labs = pubs_with_labs(pubs_1923_all)

# ------------------------------------------------------------
# 1) Topline metrics (against the whole UL dataset, 2019–2023)
# ------------------------------------------------------------
UL_TOTAL_1923 = int(pubs_1923_all["openalex_id"].nunique())
LAB_TOTAL_1923 = int(pubs_1923_with_labs["openalex_id"].nunique())
coverage_pct = (LAB_TOTAL_1923 / UL_TOTAL_1923 * 100.0) if UL_TOTAL_1923 else 0.0

# Totals across all UL works (not only lab subset)
for c in ["in_lue","is_pp10_field","is_pp1_field"]:
    if c in pubs_1923_all.columns:
        pubs_1923_all[c] = pubs_1923_all[c].astype("boolean").fillna(False).astype(bool)

UL_LUE = int(pubs_1923_all["in_lue"].sum()) if "in_lue" in pubs_1923_all.columns else 0
UL_T10 = int(pubs_1923_all["is_pp10_field"].sum()) if "is_pp10_field" in pubs_1923_all.columns else 0
UL_T01 = int(pubs_1923_all["is_pp1_field"].sum()) if "is_pp1_field" in pubs_1923_all.columns else 0

LAB_LUE = int(pubs_1923_with_labs["in_lue"].sum()) if "in_lue" in pubs_1923_with_labs.columns else 0
LAB_T10 = int(pubs_1923_with_labs["is_pp10_field"].sum()) if "is_pp10_field" in pubs_1923_with_labs.columns else 0
LAB_T01 = int(pubs_1923_with_labs["is_pp1_field"].sum()) if "is_pp1_field" in pubs_1923_with_labs.columns else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Number of labs", f"{int(internal['lab_ror'].nunique())}")
c2.metric("Publications covered by labs (2019–2023)", f"{LAB_TOTAL_1923:,}", f"{coverage_pct:.1f}% of UL")
c3.metric("LUE publications (labs)", f"{LAB_LUE:,}", f"{(LAB_LUE / UL_LUE * 100.0) if UL_LUE else 0:.1f}% of UL LUE")
c4.metric("PP top10% (labs)", f"{LAB_T10:,}", f"{(LAB_T10 / UL_T10 * 100.0) if UL_T10 else 0:.1f}% of UL top10%")
c5.metric("PP top1% (labs)", f"{LAB_T01:,}", f"{(LAB_T01 / UL_T01 * 100.0) if UL_T01 else 0:.1f}% of UL top1%")

st.divider()

# ------------------------------------------------------------
# 2) Per-lab overview table (2019–2023, versus whole UL)
# ------------------------------------------------------------
# Bring dict_internal values (publications, % UL, % LUE (abs), % intl) and compute the rest
# Map columns in data_io so we have:
#   lab_ror, lab_name, unit_type, lab_openalex_id, pubs_total_internal, share_of_ul_internal, lue_count, intl_ratio
# (see “Needed in data_io” section below)
per_lab = compute_per_lab_overview(
    pubs_1923_all=pubs_1923_with_labs,   # only pubs that have at least one lab (to compute lab KPIs)
    labs_dict=internal,
    ul_total_works_1923=UL_TOTAL_1923,   # denominator for % UL
)

max_share   = float(per_lab["share_pct_display"].max() or 1.0)
max_lue     = float(per_lab["lue_pct_display"].max() or 1.0)
max_t10     = float(per_lab["top10_pct_display"].max() or 1.0)
max_t01     = float(per_lab["top1_pct_display"].max() or 1.0)
max_intl    = float(per_lab["intl_pct_display"].max() or 1.0)
max_intcoll = float(per_lab["internal_collab_pct"].max() or 1.0)

st.subheader("Per-lab overview (2019–2023) — ratios vs whole UL")
st.dataframe(
    per_lab[
        [
            "lab_name","pubs","share_pct_display","lue_pct_display","top10_pct_display","top1_pct_display",
            "avg_fwci","internal_collab_pct","intl_pct_display",
            "lab_ror","openalex_url","ror_url",
        ]
    ],
    width="stretch",
    hide_index=True,
    column_config={
        "lab_name": "Lab Name",
        "pubs": st.column_config.NumberColumn("Publications (2019–2023)", format="%.0f"),
        "share_pct_display": st.column_config.ProgressColumn("% Université de Lorraine", min_value=0.0, max_value=max_share, format="%.1f%%"),
        "lue_pct_display": st.column_config.ProgressColumn("% of pubs LUE", min_value=0.0, max_value=max_lue, format="%.1f%%"),
        "top10_pct_display": st.column_config.ProgressColumn("% of top10%", min_value=0.0, max_value=max_t10, format="%.1f%%"),
        "top1_pct_display": st.column_config.ProgressColumn("% of top1%", min_value=0.0, max_value=max_t01, format="%.1f%%"),
        "avg_fwci": st.column_config.NumberColumn("Avg FWCI (FR)", format="%.2f"),
        "internal_collab_pct": st.column_config.ProgressColumn("% collab with another internal lab", min_value=0.0, max_value=max_intcoll, format="%.1f%%"),
        "intl_pct_display": st.column_config.ProgressColumn("% international", min_value=0.0, max_value=max_intl, format="%.1f%%"),
        # hidden by default; user can enable from column menu
        "lab_ror": st.column_config.TextColumn("ROR ID"),
        "openalex_url": st.column_config.LinkColumn("See in OpenAlex"),
        "ror_url": st.column_config.LinkColumn("ROR page"),
    },
)

st.divider()

# ------------------------------------------------------------
# 3) Year filter for charts below (applies to sections 4–5)
# ------------------------------------------------------------
st.subheader("Charts timeframe")
years_sel = st.multiselect("Years", YEARS_1923, default=YEARS_1923, key="years_for_charts", help="Applies only to sections 4 and 5.")
if not years_sel:
    st.warning("Select at least one year.")
    st.stop()

# ------------------------------------------------------------
# 4) Compare two labs (side-by-side)
# ------------------------------------------------------------
st.subheader("Compare two labs")

labels = internal["lab_name"].tolist()
name2ror = dict(zip(internal["lab_name"], internal["lab_ror"]))
cL, cR = st.columns(2)
with cL:
    left_label = st.selectbox("Left lab", options=labels, index=0 if labels else 0)
with cR:
    right_label = st.selectbox("Right lab", options=labels, index=1 if len(labels) > 1 else 0)
if not labels:
    st.info("No labs available.")
    st.stop()

left_ror, right_ror = name2ror[left_label], name2ror[right_label]

def lab_kpis_for_years(pubs_all: pd.DataFrame, lab_ror: str, years: List[int]) -> Dict[str, int | float]:
    sub = filter_years(pubs_all, years)
    el = explode_labs(sub)
    sub = sub.merge(el, on=["openalex_id","year"], how="left")
    sub = sub[sub["lab_ror"] == lab_ror].copy()

    pubs_n = int(sub["openalex_id"].nunique())
    lue_n  = int(sub["in_lue"].sum()) if "in_lue" in sub.columns else 0
    avg_fw = float(pd.to_numeric(sub["fwci_fr"], errors="coerce").mean() or 0.0)
    return {"pubs": pubs_n, "lue": lue_n, "avg_fwci": avg_fw}

kpi_left  = lab_kpis_for_years(pubs_all, left_ror, years_sel)
kpi_right = lab_kpis_for_years(pubs_all, right_ror, years_sel)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric(f"{left_label} — Publications", f"{kpi_left['pubs']:,}")
k2.metric(f"{left_label} — LUE", f"{kpi_left['lue']:,}")
k3.metric(f"{left_label} — Avg FWCI (FR)", f"{kpi_left['avg_fwci']:.2f}")
k4.metric(f"{right_label} — Publications", f"{kpi_right['pubs']:,}")
k5.metric(f"{right_label} — LUE", f"{kpi_right['lue']:,}")
k6.metric(f"{right_label} — Avg FWCI (FR)", f"{kpi_right['avg_fwci']:.2f}")

left_fields  = field_distribution_for_lab(pubs_all, left_ror, years_sel, topics_h)
right_fields = field_distribution_for_lab(pubs_all, right_ror, years_sel, topics_h)

cA, cB = st.columns(2)
with cA:
    st.altair_chart(
        chart_fields_horizontal(left_fields, FIELD_ORDER, DOMAIN_PALETTE, f"{left_label} — Field distribution (volume)"),
        width="stretch",
    )
with cB:
    st.altair_chart(
        chart_fields_horizontal(right_fields, FIELD_ORDER, DOMAIN_PALETTE, f"{right_label} — Field distribution (volume)"),
        width="stretch",
    )

cA, cB = st.columns(2)
with cA:
    st.altair_chart(
        chart_fields_horizontal(left_fields, FIELD_ORDER, DOMAIN_PALETTE, f"{left_label} — Field distribution (% of lab works)", normalize=True),
        width="stretch",
    )
with cB:
    st.altair_chart(
        chart_fields_horizontal(right_fields, FIELD_ORDER, DOMAIN_PALETTE, f"{right_label} — Field distribution (% of lab works)", normalize=True),
        width="stretch",
    )

st.divider()

# ------------------------------------------------------------
# 5) Collaboration between selected labs
# ------------------------------------------------------------
st.subheader("Collaboration between selected labs")

copubs = compute_copubs_between_labs(pubs_all, left_ror, right_ror, years_sel)
kpis = kpis_for_copubs(copubs)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Co-publications", f"{kpis['total']:,}")
c2.metric("LUE", f"{kpis['lue']:,}")
c3.metric("PP top10%", f"{kpis['top10']:,}")
c4.metric("PP top1%", f"{kpis['top1']:,}")
c5.metric("Avg FWCI (FR)", f"{kpis['avg_fwci']:.2f}")

st.markdown("#### Co-publications by year & domain")
st.altair_chart(stacked_evolution_by_domain(copubs, topics, DOMAIN_PALETTE, years_sel), width="stretch")

# Top authors (table + CSV)
st.markdown("#### Top authors in co-publications")
top_auth = top_authors_from_copubs(copubs, pubs_all, years_sel)
if top_auth.empty:
    st.info("No authors found for the current selection.")
else:
    st.dataframe(
        top_auth.head(10)[["Author","author_id","ORCID","Copubs","FWCI_FR (copubs)","Total pubs (timeframe)","FWCI_FR (overall)","Is Lorraine","Lab(s)"]],
        width="stretch",
        hide_index=True,
        column_config={
            "author_id": st.column_config.TextColumn("Author ID"),
            "Copubs": st.column_config.NumberColumn(format="%.0f"),
            "FWCI_FR (copubs)": st.column_config.NumberColumn(format="%.2f"),
            "Total pubs (timeframe)": st.column_config.NumberColumn(format="%.0f"),
            "FWCI_FR (overall)": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.download_button(
        "Download full authors CSV",
        data=top_auth.to_csv(index=False).encode("utf-8"),
        file_name=f"copubs_top_authors_{left_ror}_vs_{right_ror}.csv",
        mime="text/csv",
    )

# All co-publications — exportable
st.markdown("#### All co-publications (top 10 by citations)")
look = topics[["topic_id","subfield_id","subfield_name","field_id","field_name","domain_id","domain_name"]].drop_duplicates()
table = copubs.merge(
    look.rename(columns={"subfield_name":"Subfield","field_name":"Field","domain_name":"Domain"}),
    left_on="primary_subfield_id", right_on="subfield_id", how="left"
)

visible_cols = ["year","type","title","in_lue","is_pp10_field","is_pp1_field","Subfield","Field","citation_count"]
hidden_cols  = ["openalex_id","doi","Domain"]
cols_present = [c for c in visible_cols + hidden_cols if c in table.columns]

table = table.sort_values(["citation_count","year"], ascending=[False, False])

st.dataframe(
    table[cols_present].head(10),
    width="stretch",
    hide_index=True,
    column_config={
        "year": st.column_config.NumberColumn("Publication Year", format="%.0f"),
        "type": "Publication Type",
        "title": "Title",
        "in_lue": "In LUE",
        "is_pp10_field": "Is_PPtop10%",
        "is_pp1_field": "Is_PPtop1%",
        "citation_count": st.column_config.NumberColumn("Number of citations", format="%.0f"),
        "openalex_id": st.column_config.TextColumn("OpenAlex ID"),
        "doi": st.column_config.TextColumn("DOI"),
        "Subfield": "Subfield name",
        "Field": "Field name",
        "Domain": "Domain name",
    },
)

st.download_button(
    "Download all co-publications (CSV)",
    data=table[cols_present].to_csv(index=False).encode("utf-8"),
    file_name=f"copubs_{left_ror}_vs_{right_ror}.csv",
    mime="text/csv",
)
