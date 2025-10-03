# pages/1_🏭_Lab_View.py
from __future__ import annotations

import re
from collections import defaultdict
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

# Fixed analysis window for topline/table (per spec)
YEAR_MIN, YEAR_MAX = 2019, 2023
YEARS_DEFAULT = list(range(YEAR_MIN, YEAR_MAX + 1))

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
    """
    Returns:
      topics_clean: ['domain_id','domain_name','field_id','field_name','subfield_id','subfield_name']
      domain_order: stable list of domains (A–Z by domain_name)
      field_order:  stable list of fields (grouped by domain, then A–Z by field_name)
    """
    keep = ["domain_id","domain_name","field_id","field_name","subfield_id","subfield_name"]
    t = topics[keep].drop_duplicates().copy()
    # Domain order A–Z
    domains = t[["domain_id","domain_name"]].drop_duplicates().sort_values("domain_name")
    domain_order = domains["domain_name"].tolist()

    # Field order: domain groups, inside A–Z by field_name
    fields = t[["domain_name","field_id","field_name"]].drop_duplicates()
    fields = fields.sort_values(["domain_name","field_name"])
    field_order = fields["field_name"].tolist()

    return t, domain_order, field_order

def make_domain_palette(domain_names: List[str]) -> Dict[str, str]:
    """
    Assign a stable color per domain. Extend palette if needed.
    (Colors chosen for good contrast; keep stable once assigned.)
    """
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2",
    ]
    pal = {}
    for i, dn in enumerate(domain_names):
        pal[dn] = base[i % len(base)]
    return pal

def prepare_core():
    """Load and filter to publications with at least one lab (any year)."""
    pubs = load_core().copy()

    # numeric / boolean coercions (guard against mixed dtypes)
    for c in ["year", "citation_count"]:
        if c in pubs.columns:
            pubs[c] = pd.to_numeric(pubs[c], errors="coerce")
    for c in ["fwci_fr"]:
        if c in pubs.columns:
            pubs[c] = pd.to_numeric(pubs[c], errors="coerce")
    for c in ["in_lue", "is_pp10_field", "is_pp1_field"]:
        if c in pubs.columns:
            pubs[c] = pubs[c].fillna(False).astype(bool)

    # keep rows with at least one lab present
    has_lab = pubs["labs_rors"].fillna("").astype(str).str.strip().ne("")
    pubs = pubs.loc[has_lab].copy()

    return pubs

def filter_years(df: pd.DataFrame, years: List[int]) -> pd.DataFrame:
    if "year" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["year"].isin(years)].copy()

def compute_pub_level_flags(pubs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-publication flags we need once:
      - has_international: at least one inst country != "FR"
      - n_labs: number of unique labs (from labs_rors)
    """
    df = pubs.copy()

    # International flag
    if "inst_countries" in df.columns:
        def has_foreign(countries: object) -> bool:
            cs = {x.upper() for x in split_positions(countries)}
            cs.discard("")  # remove empties
            # If any non-FR exists -> international
            return any(c and c != "FR" for c in cs)
        df["has_international"] = df["inst_countries"].map(has_foreign)
    else:
        df["has_international"] = False

    # Number of labs per pub
    def n_labs(s: object) -> int:
        labs = {x for x in split_positions(s) if x}
        return len(labs)
    df["n_labs"] = df["labs_rors"].map(n_labs)

    return df

def lab_only(internal: pd.DataFrame, pubs: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the 61 labs (exclude other internal structures).
    Prefer 'unit_type' == 'lab'; fallback: labs observed in pubs.
    """
    if "unit_type" in internal.columns:
        labs = internal[internal["unit_type"].astype(str).str.lower().eq("lab")].copy()
    else:
        seen = set(explode_labs(pubs)["lab_ror"].dropna().unique())
        labs = internal[internal["lab_ror"].isin(seen)].copy()
    # de-dup and sort A–Z
    labs = labs.drop_duplicates(subset=["lab_ror"]).sort_values("lab_name")
    return labs

def compute_per_lab_summary(pubs_1923: pd.DataFrame, internal_labs: pd.DataFrame) -> pd.DataFrame:
    """
    Build the per-lab table for 2019–2023:
      - publications, share of UL, %LUE, %top10, %top1, Avg FWCI_FR,
        %collab internal, %international, ROR, OpenAlex link
    """
    # Pub-level flags used by grouping
    plf = compute_pub_level_flags(pubs_1923)

    # explode works to (work, lab_ror)
    el = explode_labs(plf)
    el = el.dropna(subset=["lab_ror"])

    # For metrics that are publication-level, we need one row per work
    # to avoid double-counting per lab.
    work_flags = plf[["openalex_id", "in_lue", "is_pp10_field", "is_pp1_field", "fwci_fr", "has_international", "n_labs"]].drop_duplicates("openalex_id")

    # Total UL works in 2019–2023 (denominator for share of UL)
    ul_total = int(plf["openalex_id"].nunique())

    # join flags then aggregate per lab
    x = el.merge(work_flags, on="openalex_id", how="left")

    g = x.groupby("lab_ror", as_index=False).agg(
        pubs=("openalex_id", "nunique"),
        lue_count=("in_lue", "sum"),
        top10_count=("is_pp10_field", "sum"),
        top1_count=("is_pp1_field", "sum"),
        avg_fwci=("fwci_fr", "mean"),
        intl_count=("has_international", "sum"),
        multi_lab_count=("n_labs", lambda s: (s >= 2).sum()),
    )

    # ratios
    g["share_of_ul"] = (g["pubs"] / ul_total).replace([np.inf, -np.inf], np.nan)
    g["lue_ratio"] = (g["lue_count"] / g["pubs"]).replace([np.inf, -np.inf], np.nan)
    g["top10_ratio"] = (g["top10_count"] / g["pubs"]).replace([np.inf, -np.inf], np.nan)
    g["top1_ratio"] = (g["top1_count"] / g["pubs"]).replace([np.inf, -np.inf], np.nan)
    g["intl_ratio"] = (g["intl_count"] / g["pubs"]).replace([np.inf, -np.inf], np.nan)
    g["internal_collab_ratio"] = (g["multi_lab_count"] / g["pubs"]).replace([np.inf, -np.inf], np.nan)

    # Add lab names and OpenAlex/ROR links
    labs = internal_labs[["lab_ror", "lab_name"]].drop_duplicates()
    # Optional: if your dict_internal has openalex id for the lab
    if "lab_openalex_id" in internal_labs.columns:
        labs = labs.merge(internal_labs[["lab_ror","lab_openalex_id"]], on="lab_ror", how="left")
    else:
        labs["lab_openalex_id"] = None

    g = g.merge(labs, on="lab_ror", how="left")

    def openalex_institution_url(openalex_id: str | None) -> str:
        # Per spec, institution ID filter; fallback to ROR-filter if missing.
        if openalex_id:
            return (
                "https://openalex.org/works?"
                f"page=1&filter=authorships.institutions.id:{openalex_id},"
                "type:types/article|types/book-chapter|types/review|types/book,"
                f"publication_year:{YEAR_MIN}-{YEAR_MAX}"
            )
        return ""

    g["openalex_url"] = g["lab_openalex_id"].apply(openalex_institution_url)
    g["ror_url"] = g["lab_ror"].apply(lambda r: f"https://ror.org/{r}" if r else "")

    # ordering
    g = g.merge(internal_labs[["lab_ror","lab_name"]], on=["lab_ror","lab_name"], how="left")
    g = g.sort_values("lab_name")

    # progress reference maxima (max = best lab)
    g["share_pct_display"]   = g["share_of_ul"] * 100.0
    g["lue_pct_display"]     = g["lue_ratio"]   * 100.0
    g["top10_pct_display"]   = g["top10_ratio"] * 100.0
    g["top1_pct_display"]    = g["top1_ratio"]  * 100.0
    g["intl_pct_display"]    = g["intl_ratio"]  * 100.0
    g["internal_collab_pct"] = g["internal_collab_ratio"] * 100.0

    return g, ul_total

def _ensure_all_fields(df_counts: pd.DataFrame, topics_h: pd.DataFrame) -> pd.DataFrame:
    # df_counts columns: ['field_id','count']; we must bring all fields (26)
    all_fields = topics_h[["field_id","field_name","domain_name"]].drop_duplicates()
    out = all_fields.merge(df_counts, on="field_id", how="left").fillna({"count": 0})
    out["count"] = out["count"].astype(int)
    out["field"] = out["field_name"]
    out["domain"] = out["domain_name"]
    return out[["field_id","field","domain","count"]]

def field_distribution_for_lab(pubs: pd.DataFrame, lab_ror: str, years: List[int], topics_h: pd.DataFrame) -> pd.DataFrame:
    """Counts by field for a lab; guarantees 26 rows (all fields) with 0s."""
    sub = filter_years(pubs, years)
    el = explode_labs(sub)
    sub = sub.merge(el, on=["openalex_id","year"], how="left")
    sub = sub[sub["lab_ror"] == lab_ror].copy()
    if sub.empty:
        return _ensure_all_fields(pd.DataFrame({"field_id":[],"count":[]}), topics_h)

    # count by primary_field_id
    sub["primary_field_id"] = pd.to_numeric(sub["primary_field_id"], errors="coerce")
    g = sub.groupby("primary_field_id", as_index=False)["openalex_id"].nunique().rename(columns={"openalex_id":"count", "primary_field_id":"field_id"})
    return _ensure_all_fields(g, topics_h)

def chart_fields_horizontal(df_fields: pd.DataFrame, field_order: List[str], domain_palette: Dict[str,str], title: str, normalize: bool = False) -> alt.Chart:
    """
    df_fields: ['field','domain','count']
    normalize: if True, use % of total for the lab, but still print counts next to y-axis.
    Always show ALL field labels; fixed order; domain colors.
    """
    df = df_fields.copy()
    df["field"] = df["field"].astype("category")
    df["field"].cat.set_categories(field_order, inplace=True)
    df = df.sort_values("field")

    total = df["count"].sum()
    if normalize and total > 0:
        df["value"] = df["count"] / total * 100.0
        x_title = "% of lab works"
        x_scale = alt.Scale(domain=[0, 100])
        fmt = ".1f"
    else:
        df["value"] = df["count"]
        # set a modest headroom for nicer visuals
        x_title = "Publications"
        x_scale = alt.Scale(nice=True)
        fmt = ".0f"

    # color by domain
    color_scale = alt.Scale(domain=list(domain_palette.keys()), range=[domain_palette[d] for d in domain_palette.keys()])

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
                alt.Tooltip("value:Q", title="% (if normalized)", format=".1f"),
            ],
        )
        .properties(title=title, height=height)
    )

    # text with absolute count placed near axis (small)
    text = (
        alt.Chart(df)
        .mark_text(align="left", baseline="middle", dx=5, size=10)
        .encode(
            y=alt.Y("field:N", sort=field_order, axis=None),
            x=alt.value(0),
            text=alt.Text("count:Q", format=".0f"),
        )
    )

    return (text + bars)

def compute_copubs_between_labs(pubs: pd.DataFrame, lab_left: str, lab_right: str, years: List[int]) -> pd.DataFrame:
    """Subset of pubs (selected years) that include both labs in labs_rors."""
    sub = filter_years(pubs, years).copy()

    def has_both(labs_s: object) -> bool:
        labs = set(split_positions(labs_s))
        return lab_left in labs and lab_right in labs

    mask = sub["labs_rors"].map(has_both)
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
    """
    Stacked bar of co-publications per year by domain (primary_domain_id/name).
    """
    if df.empty:
        return alt.Chart(pd.DataFrame({"year":[], "domain":[],"count":[]})).mark_bar()

    # join domain
    look = topics.drop_duplicates("domain_id")[["domain_id","domain_name"]].rename(columns={"domain_name":"domain"})
    temp = df[["openalex_id","year","primary_domain_id"]].drop_duplicates()
    temp = temp.merge(look, left_on="primary_domain_id", right_on="domain_id", how="left")
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
    """
    Build top authors table:
      - Copubs (unique works in the copubs set)
      - ORCID, Is Lorraine, Lab(s) from dict_authors (if available)
      - FWCI_FR of the copubs for this author
      - Total pubs in timeframe (within whole dataset, not just copubs)
      - FWCI_FR overall (whole dataset)
    Only 10 top rows are shown on the page; CSV download contains all.
    """
    rows = []
    cols_present = {"authors","authors_id"}.issubset(copubs.columns)
    if cols_present:
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
        # fallback: explode from core and filter
        ea = explode_authors(copubs).rename(columns={"author_name":"Author"})
        ea = ea.merge(copubs[["openalex_id","fwci_fr","year"]], on=["openalex_id"], how="left")

    if ea.empty:
        return pd.DataFrame(columns=["Author","author_id","ORCID","Copubs","FWCI_FR (copubs)","Total pubs (timeframe)","FWCI_FR (overall)","Is Lorraine","Lab(s)"])

    # Copub counts and FWCI in copubs
    agg = (
        ea.groupby(["author_id","Author"], as_index=False)
          .agg(Copubs=("openalex_id","nunique"), fwci_copubs=("fwci_fr","mean"))
    ).rename(columns={"fwci_copubs":"FWCI_FR (copubs)"})

    # Total pubs in timeframe (within whole dataset)
    ea_all = explode_authors(filter_years(pubs_all_years, years)).rename(columns={"author_name":"Author"})
    tot = ea_all.groupby(["author_id","Author"], as_index=False)["openalex_id"].nunique().rename(columns={"openalex_id":"Total pubs (timeframe)"})

    # Overall stats (whole dataset) from helper
    overall = author_global_metrics(pubs_all_years).rename(columns={
        "author_id":"author_id",
        "author_name":"Author",
        "avg_fwci_overall":"FWCI_FR (overall)",
        "total_pubs":"Total pubs (overall)"
    })[["author_id","Author","FWCI_FR (overall)","Total pubs (overall)"]]

    # Optional enrichment from dict_authors
    lookup = None
    try:
        from lib.data_io import load_authors_lookup
        lookup = load_authors_lookup()
    except Exception:
        lookup = None

    if lookup is not None and not lookup.empty:
        lk = lookup.rename(columns={"orcid":"ORCID","is_lorraine":"Is Lorraine","labs_from_dict":"Lab(s)"}).copy()
        lk = lk[["author_id","ORCID","Is Lorraine","Lab(s)"]]
    else:
        lk = pd.DataFrame(columns=["author_id","ORCID","Is Lorraine","Lab(s)"])

    out = (
        agg.merge(tot, on=["author_id","Author"], how="left")
           .merge(overall, on=["author_id","Author"], how="left")
           .merge(lk, on="author_id", how="left")
    )

    # Sort by Copubs desc, then by FWCI_FR(copubs)
    out = out.sort_values(["Copubs","FWCI_FR (copubs)"], ascending=[False, False])
    return out

# ------------------------------------------------------------
# Data load
# ------------------------------------------------------------
with st.spinner("Loading data..."):
    pubs_raw = prepare_core()                  # subset with at least one lab
    topics = load_topics()                     # topic/subfield/field/domain dictionary
    internal = load_internal()                 # lab roster + KPIs
    internal = lab_only(internal, pubs_raw)    # keep only the 61 labs

    topics_h, DOMAIN_ORDER, FIELD_ORDER = build_topics_hierarchy(topics)
    DOMAIN_PALETTE = make_domain_palette(DOMAIN_ORDER)

# ------------------------------------------------------------
# 1) Topline metrics (fixed: 2019–2023)
# ------------------------------------------------------------
pubs_1923 = pubs_raw[(pubs_raw["year"] >= YEAR_MIN) & (pubs_raw["year"] <= YEAR_MAX)].copy()

per_lab, ul_total_1923 = compute_per_lab_summary(pubs_1923, internal)

n_labs = int(internal["lab_ror"].nunique())
n_lab_pubs = int(pubs_1923["openalex_id"].nunique())

# Coverage (% of the Université they cover) relative to ALL UL pubs 2019–2023 (with at least one lab)
# NOTE: If you want coverage vs ALL university pubs (including those with no labs), provide that total here.
coverage_pct = 100.0  # by definition, pubs_1923 is already “subset with at least one lab”

# LUE / top10 / top1 totals (lab subset) and proportion vs UL totals (all pubs_raw)
lab_lue = int(pubs_1923["in_lue"].sum())
lab_top10 = int(pubs_1923["is_pp10_field"].sum())
lab_top1 = int(pubs_1923["is_pp1_field"].sum())

ul_all_1923 = load_core()
ul_all_1923 = ul_all_1923[(ul_all_1923["year"] >= YEAR_MIN) & (ul_all_1923["year"] <= YEAR_MAX)].copy()
for c in ["in_lue","is_pp10_field","is_pp1_field"]:
    if c in ul_all_1923.columns:
        ul_all_1923[c] = ul_all_1923[c].fillna(False).astype(bool)
UL_LUE = int(ul_all_1923["in_lue"].sum()) if "in_lue" in ul_all_1923.columns else lab_lue
UL_T10 = int(ul_all_1923["is_pp10_field"].sum()) if "is_pp10_field" in ul_all_1923.columns else lab_top10
UL_T01 = int(ul_all_1923["is_pp1_field"].sum()) if "is_pp1_field" in ul_all_1923.columns else lab_top1

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Number of labs", f"{n_labs}")
c2.metric("Publications covered by labs (2019–2023)", f"{n_lab_pubs:,}", f"{coverage_pct:.0f}% of UL (subset)")
c3.metric("LUE publications", f"{lab_lue:,}", f"{(lab_lue / UL_LUE * 100.0) if UL_LUE else 0:.1f}% of UL LUE")
c4.metric("PP top10% publications", f"{lab_top10:,}", f"{(lab_top10 / UL_T10 * 100.0) if UL_T10 else 0:.1f}% of UL top10%")
c5.metric("PP top1% publications", f"{lab_top1:,}", f"{(lab_top1 / UL_T01 * 100.0) if UL_T01 else 0:.1f}% of UL top1%")

st.divider()

# ------------------------------------------------------------
# 2) Per-lab overview table (fixed: 2019–2023)
# ------------------------------------------------------------
summary = per_lab.copy()

max_share   = float(summary["share_pct_display"].max() or 1.0)
max_lue     = float(summary["lue_pct_display"].max() or 1.0)
max_t10     = float(summary["top10_pct_display"].max() or 1.0)
max_t01     = float(summary["top1_pct_display"].max() or 1.0)
max_intl    = float(summary["intl_pct_display"].max() or 1.0)
max_intcoll = float(summary["internal_collab_pct"].max() or 1.0)

visible_cols = [
    "lab_name",
    "pubs",
    "share_pct_display",
    "lue_pct_display",
    "top10_pct_display",
    "top1_pct_display",
    "avg_fwci",
    "internal_collab_pct",
    "intl_pct_display",
]
hidden_cols = ["lab_ror","openalex_url","ror_url"]
to_show = visible_cols + hidden_cols

st.subheader("Per-lab overview (2019–2023)")
st.dataframe(
    summary[to_show],
    use_container_width=True,
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
# 3) Year filter (applies to charts below)
# ------------------------------------------------------------
st.subheader("Charts timeframe")
years_sel = st.multiselect("Years", YEARS_DEFAULT, default=YEARS_DEFAULT, key="years_for_charts", help="Applies to the charts in sections 4 and 5 below.")
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

# KPIs for each lab in selected years
def lab_kpis_for_years(pubs: pd.DataFrame, lab_ror: str, years: List[int]) -> Dict[str, int | float]:
    sub = filter_years(pubs, years)
    el = explode_labs(sub)
    sub = sub.merge(el, on=["openalex_id","year"], how="left")
    sub = sub[sub["lab_ror"] == lab_ror].copy()

    pubs_n = int(sub["openalex_id"].nunique())
    lue_n  = int(sub["in_lue"].sum())
    avg_fw = float(pd.to_numeric(sub["fwci_fr"], errors="coerce").mean() or 0.0)
    return {"pubs": pubs_n, "lue": lue_n, "avg_fwci": avg_fw}

kpi_left  = lab_kpis_for_years(pubs_raw, left_ror, years_sel)
kpi_right = lab_kpis_for_years(pubs_raw, right_ror, years_sel)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric(f"{left_label} — Publications", f"{kpi_left['pubs']:,}")
k2.metric(f"{left_label} — LUE", f"{kpi_left['lue']:,}")
k3.metric(f"{left_label} — Avg FWCI (FR)", f"{kpi_left['avg_fwci']:.2f}")
k4.metric(f"{right_label} — Publications", f"{kpi_right['pubs']:,}")
k5.metric(f"{right_label} — LUE", f"{kpi_right['lue']:,}")
k6.metric(f"{right_label} — Avg FWCI (FR)", f"{kpi_right['avg_fwci']:.2f}")

# Field distributions (volume and %)
left_fields  = field_distribution_for_lab(pubs_raw, left_ror, years_sel, topics_h)
right_fields = field_distribution_for_lab(pubs_raw, right_ror, years_sel, topics_h)

cA, cB = st.columns(2)
with cA:
    st.altair_chart(chart_fields_horizontal(left_fields, FIELD_ORDER, DOMAIN_PALETTE, f"{left_label} — Field distribution (volume)"), use_container_width=True)
with cB:
    st.altair_chart(chart_fields_horizontal(right_fields, FIELD_ORDER, DOMAIN_PALETTE, f"{right_label} — Field distribution (volume)"), use_container_width=True)

cA, cB = st.columns(2)
with cA:
    st.altair_chart(chart_fields_horizontal(left_fields, FIELD_ORDER, DOMAIN_PALETTE, f"{left_label} — Field distribution (% of lab works)", normalize=True), use_container_width=True)
with cB:
    st.altair_chart(chart_fields_horizontal(right_fields, FIELD_ORDER, DOMAIN_PALETTE, f"{right_label} — Field distribution (% of lab works)", normalize=True), use_container_width=True)

st.divider()

# ------------------------------------------------------------
# 5) Collaboration between selected labs
# ------------------------------------------------------------
st.subheader("Collaboration between selected labs")

copubs = compute_copubs_between_labs(pubs_raw, left_ror, right_ror, years_sel)

kpis = kpis_for_copubs(copubs)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Co-publications", f"{kpis['total']:,}")
c2.metric("LUE", f"{kpis['lue']:,}")
c3.metric("PP top10%", f"{kpis['top10']:,}")
c4.metric("PP top1%", f"{kpis['top1']:,}")
c5.metric("Avg FWCI (FR)", f"{kpis['avg_fwci']:.2f}")

# Evolution stacked by domain
st.markdown("#### Co-publications by year & domain")
evo = stacked_evolution_by_domain(copubs, topics, DOMAIN_PALETTE, years_sel)
st.altair_chart(evo, use_container_width=True)

# Top authors among the copubs
st.markdown("#### Top authors in co-publications")
top_auth = top_authors_from_copubs(copubs, pubs_raw, years_sel)
if top_auth.empty:
    st.info("No authors found for the current selection.")
else:
    st.dataframe(
        top_auth.head(10)[["Author","author_id","ORCID","Copubs","FWCI_FR (copubs)","Total pubs (timeframe)","FWCI_FR (overall)","Is Lorraine","Lab(s)"]],
        use_container_width=True, hide_index=True,
        column_config={
            "author_id": st.column_config.TextColumn("Author ID"),
            "Copubs": st.column_config.NumberColumn(format="%.0f"),
            "FWCI_FR (copubs)": st.column_config.NumberColumn(format="%.2f"),
            "Total pubs (timeframe)": st.column_config.NumberColumn(format="%.0f"),
            "FWCI_FR (overall)": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    csv_all_auth = top_auth.to_csv(index=False).encode("utf-8")
    st.download_button("Download full authors CSV", data=csv_all_auth, file_name=f"copubs_top_authors_{left_ror}_vs_{right_ror}.csv", mime="text/csv")

# All co-publications — exportable table
st.markdown("#### All co-publications (top 10 by citations)")

# Join subfield/field/domain from topics
look = topics[["topic_id","subfield_id","subfield_name","field_id","field_name","domain_id","domain_name"]].drop_duplicates()
# primary topic/subfield/field/domain:
table = copubs.merge(
    look.rename(columns={
        "subfield_name":"Subfield",
        "field_name":"Field",
        "domain_name":"Domain"
    }),
    left_on="primary_subfield_id",
    right_on="subfield_id",
    how="left"
)

# Visible by default
visible_cols = ["year","type","title","in_lue","is_pp10_field","is_pp1_field","Subfield","Field","citation_count"]
hidden_cols  = ["openalex_id","doi","Domain"]
cols_present = [c for c in visible_cols + hidden_cols if c in table.columns]

# order & top 10 most cited
table = table.sort_values(["citation_count","year"], ascending=[False, False])

st.dataframe(
    table[cols_present].head(10),
    use_container_width=True, hide_index=True,
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
    }
)
csv_copubs = table[cols_present].to_csv(index=False).encode("utf-8")
st.download_button("Download all co-publications (CSV)", data=csv_copubs, file_name=f"copubs_{left_ror}_vs_{right_ror}.csv", mime="text/csv")
