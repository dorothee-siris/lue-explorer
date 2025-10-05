# pages/2_📚_Topic_View.py
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from lib.data_io import load_parquet  # uses your existing helper

st.set_page_config(page_title="Topic View", page_icon="📚", layout="wide")
st.title("📚 Topic View")

YEAR_MIN, YEAR_MAX = 2019, 2023
UL_OPENALEX_ID = "i90183372"  # Université de Lorraine (for OpenAlex links)

# -------------------------------
# Helpers: parsing, ordering, viz
# -------------------------------
PAIR_RE = re.compile(r"^\s*([^\s(]+)\s*\(([^)]*)\)\s*$")  # "<id> (<value>)"

def split_pipe(s: object) -> List[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    return [x.strip() for x in str(s).split("|") if str(x).strip()]

def parse_id_value_list(cell: object, cast=float) -> pd.DataFrame:
    """
    Parse strings like:  "11 (0.3341) | 13 (0.4709)"
    → DataFrame: id (str), value (cast)
    """
    rows = []
    for tok in split_pipe(cell):
        m = PAIR_RE.match(tok)
        if not m:
            continue
        key, val = m.group(1).strip(), m.group(2).strip()
        try:
            v = cast(str(val).replace(",", "."))
        except Exception:
            try:
                v = float(str(val).replace(",", "."))
            except Exception:
                v = np.nan
        rows.append({"id": key, "value": v})
    return pd.DataFrame(rows)

def parse_parallel_lists(**cols) -> pd.DataFrame:
    """
    Given parallel pipe-separated columns (same length), e.g.:
      names="A | B", types="x | y", counts="1 | 2"
    → DataFrame aligned across columns.
    """
    keys = list(cols.keys())
    lists = [split_pipe(cols[k]) for k in keys]
    L = max((len(x) for x in lists), default=0)
    aligned = {}
    for k, arr in zip(keys, lists):
        aligned[k] = arr + [""] * (L - len(arr))
    return pd.DataFrame(aligned)

def build_orders(topics: pd.DataFrame) -> Tuple[List[int], Dict[int, List[int]], Dict[int, List[int]]]:
    """Return domain order; field order per domain; subfield order per field."""
    t = topics.copy()
    for c in ["domain_id","field_id","subfield_id"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t = t.dropna(subset=["domain_id","field_id","subfield_id"]).astype({"domain_id":int, "field_id":int, "subfield_id":int})
    domain_order = sorted(t["domain_id"].unique().tolist())
    field_order_by_domain: Dict[int, List[int]] = {}
    subfield_order_by_field: Dict[int, List[int]] = {}
    for d in domain_order:
        field_order_by_domain[d] = sorted(t.loc[t["domain_id"] == d, "field_id"].unique().tolist())
    for f in sorted(t["field_id"].unique()):
        subfield_order_by_field[f] = sorted(t.loc[t["field_id"] == f, "subfield_id"].unique().tolist())
    return domain_order, field_order_by_domain, subfield_order_by_field

def make_domain_palette(domain_ids: List[int], topics: pd.DataFrame) -> Dict[str, str]:
    """Stable color per domain name; order by domain_id."""
    base = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
        "#4c78a8","#f58518","#54a24b","#e45756","#b279a2",
    ]
    names = (topics[["domain_id","domain_name"]].drop_duplicates().sort_values("domain_id"))
    palette = {}
    for i, (_, row) in enumerate(names.iterrows()):
        palette[str(row["domain_name"])] = base[i % len(base)]
    return palette

def whisker_chart(df: pd.DataFrame, y_field: str, color_field: str, title: str, log=True, palette=None, fixed_order=None, height=None):
    """
    df columns required: [y_field, 'min','q1','median','q3','max', color_field]
    Horizontal box-with-whiskers; log scale optional.
    """
    data = df.copy()
    eps = 1e-3
    for c in ["min","q1","median","q3","max"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
        if log:
            data[c] = data[c].clip(lower=eps)

    cat_order = fixed_order if fixed_order is not None else data[y_field].tolist()

    color_scale = None
    if palette:
        dom = list(palette.keys())
        rng = [palette[k] for k in dom]
        color_scale = alt.Scale(domain=dom, range=rng)

    base = alt.Chart(data).transform_calculate(
        min_x="datum.min", max_x="datum.max", q1_x="datum.q1", q3_x="datum.q3", med_x="datum.median"
    )

    rule = base.mark_rule().encode(
        y=alt.Y(f"{y_field}:N", sort=cat_order, axis=alt.Axis(title=None, labelLimit=1000)),
        x=alt.X("min:Q", title="FWCI_FR (log)" if log else "FWCI_FR", scale=alt.Scale(type="log") if log else alt.Scale(nice=True)),
        x2="max:Q",
        color=alt.Color(f"{color_field}:N", legend=None, scale=color_scale) if color_field else alt.value("#888"),
        tooltip=[y_field, alt.Tooltip("min:Q", title="Min", format=".2f"), alt.Tooltip("q1:Q", title="Q1", format=".2f"),
                 alt.Tooltip("median:Q", title="Median", format=".2f"), alt.Tooltip("q3:Q", title="Q3", format=".2f"),
                 alt.Tooltip("max:Q", title="Max", format=".2f")],
    )

    box = base.mark_bar(size=10, opacity=0.6).encode(
        y=alt.Y(f"{y_field}:N", sort=cat_order, axis=None),
        x=alt.X("q1:Q"),
        x2="q3:Q",
        color=alt.Color(f"{color_field}:N", legend=None, scale=color_scale) if color_field else alt.value("#888"),
    )

    tick = base.mark_tick(thickness=2, size=20).encode(
        y=alt.Y(f"{y_field}:N", sort=cat_order, axis=None),
        x=alt.X("median:Q"),
        color=alt.value("#111"),
    )

    return (rule + box + tick).properties(
        title=title,
        height=height or max(26 * len(cat_order), 240),
        width="container",  # <- responsive width inside the element
    )


def percent_bar_with_counts(df, y, pct_col, count_col, title, color_value, order=None, height=None):
    """
    Horizontal bars for percentages, with the absolute count printed near the y-axis.
    df[pct_col] expected in 0..1
    """
    data = df.copy()
    data["pct_display"] = pd.to_numeric(data[pct_col], errors="coerce").fillna(0) * 100.0
    data[count_col] = pd.to_numeric(data[count_col], errors="coerce").fillna(0)

    bars = alt.Chart(data).mark_bar().encode(
        y=alt.Y(f"{y}:N", sort=order, axis=alt.Axis(title=None, labelLimit=1000)),
        x=alt.X("pct_display:Q", title="%", scale=alt.Scale(domain=[0, float(max(1.0, data['pct_display'].max()))])),
        color=alt.value(color_value),
        tooltip=[y, alt.Tooltip("pct_display:Q", title="%", format=".1f"), alt.Tooltip(f"{count_col}:Q", title="Count", format=".0f")],
    )

    text = alt.Chart(data).mark_text(align="left", baseline="middle", dx=5, size=10).encode(
        y=alt.Y(f"{y}:N", sort=order, axis=None),
        x=alt.value(0),
        text=alt.Text(f"{count_col}:Q", format=".0f"),
    )

    return (text + bars).properties(
        title=title,
        height=height or max(26*len(data), 240),
        width="container",  # <- responsive width inside the element
    )

def openalex_for_domain(domain_id: int) -> str:
    return (
        "https://openalex.org/works?"
        f"page=1&filter=authorships.institutions.lineage:{UL_OPENALEX_ID},"
        "type:types/article|types/book-chapter|types/review|types/book,"
        f"publication_year:{YEAR_MIN}-{YEAR_MAX},"
        f"primary_topic.domain.id:domains/{int(domain_id)}"
    )

def openalex_for_field(field_id: int) -> str:
    return (
        "https://openalex.org/works?"
        f"page=1&filter=authorships.institutions.lineage:{UL_OPENALEX_ID},"
        "type:types/article|types/book-chapter|types/review|types/book,"
        f"publication_year:{YEAR_MIN}-{YEAR_MAX},"
        f"primary_topic.field.id:fields/{int(field_id)}"
    )

# -------------------------------
# Load data (only your files)
# -------------------------------
with st.spinner("Loading indicators..."):
    DI = load_parquet("ul_domains_indicators.parquet").copy()
    FI = load_parquet("ul_fields_indicators.parquet").copy()
    TOP = load_parquet("all_topics.parquet").copy()
    UNITS = load_parquet("ul_units_indicators.parquet").copy()

# Lab name lookups from ul_units_indicators
LAB_NAMES = (UNITS[["ROR","Unit Name"]]
             .dropna()
             .drop_duplicates()
             .rename(columns={"ROR":"lab_ror","Unit Name":"lab_name"}))

# Canonical orders + palette
DOMAIN_ORDER, FIELD_ORDER_BY_DOMAIN, SUBFIELD_ORDER_BY_FIELD = build_orders(TOP)
DOMAIN_NAME_MAP = (TOP[["domain_id","domain_name"]].drop_duplicates()
                   .sort_values("domain_id").set_index("domain_id")["domain_name"].to_dict())
FIELD_NAME_MAP = (TOP[["field_id","field_name"]].drop_duplicates()
                  .sort_values("field_id").set_index("field_id")["field_name"].to_dict())
FIELD_TO_DOMAIN = (TOP[["field_id","domain_id","domain_name"]]
                   .drop_duplicates()
                   .set_index("field_id")[["domain_id","domain_name"]])
DOMAIN_PALETTE = make_domain_palette(DOMAIN_ORDER, TOP)

# Ensure name columns exist/consistent
if "Domain name" not in DI.columns:
    DI["Domain name"] = DI["Domain ID"].map(DOMAIN_NAME_MAP)
if "See in OpenAlex" not in DI.columns:
    DI["See in OpenAlex"] = DI["Domain ID"].map(openalex_for_domain)

if "Field name" not in FI.columns:
    FI["Field name"] = FI["Field ID"].map(FIELD_NAME_MAP)
if "See in OpenAlex" not in FI.columns:
    FI["See in OpenAlex"] = FI["Field ID"].map(openalex_for_field)

# Add domain info to fields for colors/order
FI["__field_id_int"] = pd.to_numeric(FI["Field ID"], errors="coerce").astype("Int64")
FI = FI.merge(FIELD_TO_DOMAIN, left_on="__field_id_int", right_index=True, how="left")
FI = FI.drop(columns=["__field_id_int"])

# Sort DI and FI by canonical order
DI["__d_sort"] = pd.to_numeric(DI["Domain ID"], errors="coerce")
DI = DI.sort_values("__d_sort").drop(columns=["__d_sort"])
FI["__f_sort"] = pd.to_numeric(FI["Field ID"], errors="coerce")
FI = FI.sort_values("__f_sort").drop(columns=["__f_sort"])

# ----------------------------------------
# Tabs: Domains overview / Fields overview
# ----------------------------------------
tab_domains, tab_fields = st.tabs(["Domains", "Fields"])

# ====== DOMAINS TAB ======
with tab_domains:
    st.subheader("Domain overview (2019–2023)")

    # Maxima for progress columns
    def _max(series):
        try:
            s = pd.to_numeric(series, errors="coerce") * 100.0
            m = float(s.max())
            return m if m > 0 else 1.0
        except Exception:
            return 1.0

    max_uni   = _max(DI["% Pubs (uni level)"])
    max_lue_d = _max(DI["% Pubs LUE (domain level)"])
    max_t10_d = _max(DI["% PPtop10% (domain level)"])
    max_t01_d = _max(DI["% PPtop1% (domain level)"])
    max_coll  = _max(DI["% internal collaboration"])
    max_intl  = _max(DI["% international"])
    max_ind   = _max(DI["% industrial"])

    max_lue_ul = _max(DI["% Pubs LUE (uni level)"])
    max_t10_ul = _max(DI["% PPtop10% (uni level)"])
    max_t01_ul = _max(DI["% PPtop1% (uni level)"])

    dom_cols = [
        "Domain ID","Domain name","Pubs",
        "% Pubs (uni level)","% Pubs LUE (domain level)","% PPtop10% (domain level)","% PPtop1% (domain level)",
        "% internal collaboration","% international","% industrial",
        "Avg FWCI (France)","% Pubs LUE (uni level)","% PPtop10% (uni level)","% PPtop1% (uni level)","See in OpenAlex"
    ]
    dom_tbl = DI[dom_cols].copy()

    # convert to display percent
    for c in ["% Pubs (uni level)","% Pubs LUE (domain level)","% PPtop10% (domain level)","% PPtop1% (domain level)",
              "% internal collaboration","% international","% industrial",
              "% Pubs LUE (uni level)","% PPtop10% (uni level)","% PPtop1% (uni level)"]:
        if c in dom_tbl.columns:
            dom_tbl[c] = pd.to_numeric(dom_tbl[c], errors="coerce") * 100.0

    st.dataframe(
        dom_tbl,
        width="stretch",
        hide_index=True,
        column_config={
            "Domain ID": st.column_config.NumberColumn("Domain ID", format="%.0f"),
            "Domain name": "Domain name",
            "Pubs": st.column_config.NumberColumn("Publication counts", format="%.0f"),
            "% Pubs (uni level)": st.column_config.ProgressColumn("% Pubs (uni level)", min_value=0.0, max_value=max_uni, format="%.1f%%"),
            "% Pubs LUE (domain level)": st.column_config.ProgressColumn("% Pubs LUE (domain level)", min_value=0.0, max_value=max_lue_d, format="%.1f%%"),
            "% PPtop10% (domain level)": st.column_config.ProgressColumn("% PPtop10% (domain level)", min_value=0.0, max_value=max_t10_d, format="%.1f%%"),
            "% PPtop1% (domain level)": st.column_config.ProgressColumn("% PPtop1% (domain level)", min_value=0.0, max_value=max_t01_d, format="%.1f%%"),
            "% internal collaboration": st.column_config.ProgressColumn("% internal collaboration", min_value=0.0, max_value=max_coll, format="%.1f%%"),
            "% international": st.column_config.ProgressColumn("% international", min_value=0.0, max_value=max_intl, format="%.1f%%"),
            "% industrial": st.column_config.ProgressColumn("% industrial", min_value=0.0, max_value=max_ind, format="%.1f%%"),
            "Avg FWCI (France)": st.column_config.NumberColumn("Avg FWCI (FR)", format="%.2f"),
            "% Pubs LUE (uni level)": st.column_config.ProgressColumn("% Pubs LUE (UL level)", min_value=0.0, max_value=max_lue_ul, format="%.1f%%"),
            "% PPtop10% (uni level)": st.column_config.ProgressColumn("% PPtop10% (UL level)", min_value=0.0, max_value=max_t10_ul, format="%.1f%%"),
            "% PPtop1% (uni level)": st.column_config.ProgressColumn("% PPtop1% (UL level)", min_value=0.0, max_value=max_t01_ul, format="%.1f%%"),
            "See in OpenAlex": st.column_config.LinkColumn("See in OpenAlex"),
        },
    )

    st.markdown("#### FWCI_FR spread by domain (log scale)")
    wh = (DI[["Domain ID","Domain name","FWCI_FR min","FWCI_FR Q1","FWCI_FR Q2","FWCI_FR Q3","FWCI_FR max"]]
          .rename(columns={"FWCI_FR Q2":"FWCI_FR median"}))
    wh = wh.dropna(subset=["FWCI_FR min","FWCI_FR Q1","FWCI_FR median","FWCI_FR Q3","FWCI_FR max"]).copy()
    wh["color_key"] = wh["Domain name"]
    wh = wh.rename(columns={
        "Domain name":"Domain",
        "FWCI_FR min":"min", "FWCI_FR Q1":"q1", "FWCI_FR median":"median", "FWCI_FR Q3":"q3", "FWCI_FR max":"max"
    })
    st.altair_chart(
        whisker_chart(
            wh, y_field="Domain", color_field="color_key", title="FWCI_FR (log) — domains",
            log=True, palette=DOMAIN_PALETTE, fixed_order=[(DOMAIN_NAME_MAP.get(d) or "") for d in DOMAIN_ORDER]
        ),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Explore a domain")

    # domain selector
    domain_labels = [(d, DOMAIN_NAME_MAP.get(d, f"Domain {d}")) for d in DOMAIN_ORDER]
    sel_label = st.selectbox("Choose domain", options=[name for _, name in domain_labels], index=0, key="pick_domain")
    sel_id = domain_labels[[name for _, name in domain_labels].index(sel_label)][0]
    sel_color = DOMAIN_PALETTE.get(sel_label, "#4c78a8")

    row = DI.loc[DI["Domain ID"] == sel_id].head(1)
    if row.empty:
        st.info("No indicators available for this domain.")
        st.stop()
    row = row.iloc[0]

    # ----- Labs contribution (>= 2%) -----
    st.markdown("#### Labs contribution")

    labs_pct_df = parse_id_value_list(row.get("By lab: % of domain pubs"), cast=float)   # id: lab_ror, value: ratio
    labs_cnt_df = parse_id_value_list(row.get("By lab: count"), cast=float)
    labs = labs_pct_df.merge(labs_cnt_df, on="id", how="left", suffixes=("_pct","_count")).rename(columns={"id":"lab_ror","value_pct":"ratio","value_count":"count"})
    labs = labs.merge(LAB_NAMES, on="lab_ror", how="left")
    labs["lab_label"] = labs["lab_name"].fillna(labs["lab_ror"])
    labs = labs[pd.to_numeric(labs["ratio"], errors="coerce") >= 0.02].copy()
    labs = labs.sort_values("ratio", ascending=False)

    cA, cB = st.columns(2)
    with cA:
        if labs.empty:
            st.info("No labs above the 2% threshold for this domain.")
        else:
            order = labs["lab_label"].tolist()
            st.altair_chart(
                percent_bar_with_counts(
                    labs, y="lab_label", pct_col="ratio", count_col="count",
                    title=f"{sel_label} — labs’ share of domain (≥2%)",
                    color_value=sel_color, order=order, height=max(26*len(order), 240)
                ),
                use_container_width=True,
            )

    # Whiskers by lab for selected domain (only those shown at left)
    def _lab_whisk(colname):
        return parse_id_value_list(row.get(colname), cast=float).rename(columns={"id":"lab_ror"})

    w = (_lab_whisk("By lab: FWCI_FR min")
         .merge(_lab_whisk("By lab: FWCI_FR Q1"), on="lab_ror", how="outer", suffixes=("","_q1"))
         .merge(_lab_whisk("By lab: FWCI_FR Q2"), on="lab_ror", how="outer", suffixes=("","_q2"))
         .merge(_lab_whisk("By lab: FWCI_FR Q3"), on="lab_ror", how="outer", suffixes=("","_q3"))
         .merge(_lab_whisk("By lab: FWCI_FR max"), on="lab_ror", how="outer", suffixes=("","_max"))
    )
    w = w.rename(columns={"value":"min","value_q1":"q1","value_q2":"median","value_q3":"q3","value_max":"max"})
    w = w.merge(labs[["lab_ror","lab_label"]], on="lab_ror", how="inner")

    with cB:
        if w.empty:
            st.info("No FWCI_FR distribution available for selected labs.")
        else:
            st.altair_chart(
                whisker_chart(
                    w.assign(Domain=sel_label)[["lab_label","min","q1","median","q3","max","Domain"]],
                    y_field="lab_label", color_field="Domain", title=f"{sel_label} — FWCI_FR spread for labs (log)",
                    log=True, palette={sel_label: sel_color}, fixed_order=labs["lab_label"].tolist(), height=max(26*len(labs), 240)
                ),
                use_container_width=True,
            )

    st.divider()

    # ----- Top partners -----
    st.markdown("#### Top partners")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 20 French partners**")
        fr = parse_parallel_lists(
            name=row.get("Top 20 FR partners (name)"),
            type=row.get("Top 20 FR partners (type)"),
            copubs=row.get("Top 20 FR partners (totals copubs in this domain)"),
            share=row.get("Top 20 FR partners (% of UL total copubs)"),
        )
        for col in ["copubs","share"]:
            if col in fr: fr[col] = pd.to_numeric(fr[col].str.replace(",", "."), errors="coerce")
        st.dataframe(
            fr.head(20),
            width="stretch",
            hide_index=True,
            column_config={
                "name":"Partner",
                "type":"Type",
                "copubs": st.column_config.NumberColumn("Co-pubs in domain", format="%.0f"),
                "share": st.column_config.NumberColumn("% of UL co-pubs with partner", format="%.1f%%"),
            },
        )

    with c2:
        st.markdown("**Top 20 international partners**")
        intl = parse_parallel_lists(
            name=row.get("Top 20 int partners (name)"),
            type=row.get("Top 20 int partners (type)"),
            country=row.get("Top 20 int partners (country)"),
            copubs=row.get("Top 20 int partners (totals copubs in this domain)"),
            share=row.get("Top 20 int partners (% of UL total copubs)"),
        )
        for col in ["copubs","share"]:
            if col in intl: intl[col] = pd.to_numeric(intl[col].str.replace(",", "."), errors="coerce")
        st.dataframe(
            intl.head(20),
            width="stretch",
            hide_index=True,
            column_config={
                "name":"Partner",
                "type":"Type",
                "country":"Country",
                "copubs": st.column_config.NumberColumn("Co-pubs in domain", format="%.0f"),
                "share": st.column_config.NumberColumn("% of UL co-pubs with partner", format="%.1f%%"),
            },
        )

    st.divider()

    # ----- Top authors -----
    st.markdown("#### Top 20 authors")
    authors = parse_parallel_lists(
        name=row.get("Top 20 authors (name)"),
        pubs=row.get("Top 20 authors (pubs)"),
        orcid=row.get("Top 20 authors (Orcid)"),
        author_id=row.get("Top 20 authors (ID)"),
        fwci=row.get("Top 20 authors (Average FWCI_FR)"),
        t10=row.get("Top 20 authors (PPtop10% Count)"),
        t01=row.get("Top 20 authors (PPtop1% Count)"),
        lorraine=row.get("Top 20 authors (Is Lorraine)"),
        labs=row.get("Top 20 authors (Labs)"),
    )
    for col in ["pubs","fwci","t10","t01"]:
        if col in authors: authors[col] = pd.to_numeric(authors[col].str.replace(",", "."), errors="coerce")
    if "lorraine" in authors:
        authors["lorraine"] = authors["lorraine"].str.strip().str.lower().map({"true": True, "false": False}).fillna(authors["lorraine"])
    authors = authors.sort_values(["pubs","fwci"], ascending=[False, False])

    st.dataframe(
        authors.head(20)[["name","pubs","fwci","t10","t01","lorraine","orcid","author_id","labs"]],
        width="stretch",
        hide_index=True,
        column_config={
            "name":"Author",
            "pubs": st.column_config.NumberColumn("Pubs", format="%.0f"),
            "fwci": st.column_config.NumberColumn("Average FWCI_FR", format="%.2f"),
            "t10": st.column_config.NumberColumn("PPtop10% Count", format="%.0f"),
            "t01": st.column_config.NumberColumn("PPtop1% Count", format="%.0f"),
            "lorraine":"Is Lorraine",
            "orcid": st.column_config.TextColumn("ORCID"),
            "author_id": st.column_config.TextColumn("Author ID"),
            "labs": "Lab(s)",
        },
    )

    st.divider()

    # ----- Thematic shape: field distribution within selected domain -----
    st.markdown("#### Thematic shape — field distribution within domain")
    fld_pct = parse_id_value_list(row.get("By field: % of domain pubs"), cast=float).rename(columns={"id":"field_id","value":"ratio"})
    fld_pct["field_id"] = fld_pct["field_id"].astype(str)
    fld_cnt = parse_id_value_list(row.get("By field: count"), cast=float).rename(columns={"id":"field_id","value":"count"})
    fld_cnt["field_id"] = fld_cnt["field_id"].astype(str)

    fields_canon = (TOP[TOP["domain_id"] == sel_id][["field_id","field_name","domain_name"]]
                    .drop_duplicates()
                    .sort_values("field_id"))
    fields_canon["field_id"] = fields_canon["field_id"].astype(int).astype(str)

    field_mix = fields_canon.merge(fld_pct, on="field_id", how="left").merge(fld_cnt, on="field_id", how="left")
    field_mix["ratio"] = pd.to_numeric(field_mix["ratio"], errors="coerce").fillna(0.0)
    field_mix["count"] = pd.to_numeric(field_mix["count"], errors="coerce").fillna(0.0)
    order_fields = field_mix["field_name"].tolist()

    st.altair_chart(
        percent_bar_with_counts(
            field_mix.rename(columns={"field_name":"Field"}),
            y="Field", pct_col="ratio", count_col="count",
            title=f"{sel_label} — field mix (% of domain)",
            color_value=sel_color, order=order_fields, height=max(26*len(order_fields), 260)
        ),
        use_container_width=True,
    )

# ====== FIELDS TAB ======
with tab_fields:
    st.subheader("Field overview (2019–2023)")

    def _max(series):
        try:
            s = pd.to_numeric(series, errors="coerce") * 100.0
            m = float(s.max())
            return m if m > 0 else 1.0
        except Exception:
            return 1.0

    max_uni   = _max(FI["% Pubs (uni level)"])
    max_lue_f = _max(FI["% Pubs LUE (field level)"])
    max_t10_f = _max(FI["% PPtop10% (field level)"])
    max_t01_f = _max(FI["% PPtop1% (field level)"])
    max_coll  = _max(FI["% internal collaboration"])
    max_intl  = _max(FI["% international"])
    max_ind   = _max(FI["% industrial"])
    max_lue_ul = _max(FI["% Pubs LUE (uni level)"])
    max_t10_ul = _max(FI["% PPtop10% (uni level)"])
    max_t01_ul = _max(FI["% PPtop1% (uni level)"])

    field_cols = [
        "Field ID","Field name","Pubs",
        "% Pubs (uni level)","% Pubs LUE (field level)","% PPtop10% (field level)","% PPtop1% (field level)",
        "% internal collaboration","% international","% industrial",
        "Avg FWCI (France)","% Pubs LUE (uni level)","% PPtop10% (uni level)","% PPtop1% (uni level)","See in OpenAlex"
    ]
    fld_tbl = FI[field_cols].copy()

    for c in ["% Pubs (uni level)","% Pubs LUE (field level)","% PPtop10% (field level)","% PPtop1% (field level)",
              "% internal collaboration","% international","% industrial",
              "% Pubs LUE (uni level)","% PPtop10% (uni level)","% PPtop1% (uni level)"]:
        if c in fld_tbl.columns:
            fld_tbl[c] = pd.to_numeric(fld_tbl[c], errors="coerce") * 100.0

    st.dataframe(
        fld_tbl,
        width="stretch",
        hide_index=True,
        column_config={
            "Field ID": st.column_config.NumberColumn("Field ID", format="%.0f"),
            "Field name": "Field name",
            "Pubs": st.column_config.NumberColumn("Publication counts", format="%.0f"),
            "% Pubs (uni level)": st.column_config.ProgressColumn("% Pubs (uni level)", min_value=0.0, max_value=max_uni, format="%.1f%%"),
            "% Pubs LUE (field level)": st.column_config.ProgressColumn("% Pubs LUE (field level)", min_value=0.0, max_value=max_lue_f, format="%.1f%%"),
            "% PPtop10% (field level)": st.column_config.ProgressColumn("% PPtop10% (field level)", min_value=0.0, max_value=max_t10_f, format="%.1f%%"),
            "% PPtop1% (field level)": st.column_config.ProgressColumn("% PPtop1% (field level)", min_value=0.0, max_value=max_t01_f, format="%.1f%%"),
            "% internal collaboration": st.column_config.ProgressColumn("% internal collaboration", min_value=0.0, max_value=max_coll, format="%.1f%%"),
            "% international": st.column_config.ProgressColumn("% international", min_value=0.0, max_value=max_intl, format="%.1f%%"),
            "% industrial": st.column_config.ProgressColumn("% industrial", min_value=0.0, max_value=max_ind, format="%.1f%%"),
            "Avg FWCI (France)": st.column_config.NumberColumn("Avg FWCI (FR)", format="%.2f"),
            "% Pubs LUE (uni level)": st.column_config.ProgressColumn("% Pubs LUE (UL level)", min_value=0.0, max_value=max_lue_ul, format="%.1f%%"),
            "% PPtop10% (uni level)": st.column_config.ProgressColumn("% PPtop10% (UL level)", min_value=0.0, max_value=max_t10_ul, format="%.1f%%"),
            "% PPtop1% (uni level)": st.column_config.ProgressColumn("% PPtop1% (UL level)", min_value=0.0, max_value=max_t01_ul, format="%.1f%%"),
            "See in OpenAlex": st.column_config.LinkColumn("See in OpenAlex"),
        },
    )

    st.divider()
    st.subheader("Explore a field")

    # field selector (ordered by field_id)
    field_labels = FI[["Field ID","Field name","domain_name"]].copy()
    field_labels["Field ID"] = pd.to_numeric(field_labels["Field ID"], errors="coerce")
    field_labels = field_labels.dropna(subset=["Field ID"]).sort_values("Field ID")
    # include domain for disambiguation
    field_labels["label"] = field_labels.apply(lambda r: f"{int(r['Field ID'])} — {r['Field name']} ({r['domain_name']})", axis=1)
    sel_field_label = st.selectbox("Choose field", options=field_labels["label"].tolist(), index=0, key="pick_field")
    sel_field_id = int(field_labels.loc[field_labels["label"] == sel_field_label, "Field ID"].iloc[0])
    sel_field_domain = field_labels.loc[field_labels["Field ID"] == sel_field_id, "domain_name"].iloc[0]
    sel_field_color = DOMAIN_PALETTE.get(str(sel_field_domain), "#4c78a8")

    frow = FI.loc[pd.to_numeric(FI["Field ID"], errors="coerce").eq(sel_field_id)].head(1)
    if frow.empty:
        st.info("No indicators available for this field.")
        st.stop()
    frow = frow.iloc[0]

    # ----- Labs contribution (>= 2%) -----
    st.markdown("#### Labs contribution")

    labs_pct_df = parse_id_value_list(frow.get("By lab: % of field pubs"), cast=float).rename(columns={"id":"lab_ror","value":"ratio"})
    labs_cnt_df = parse_id_value_list(frow.get("By lab: count"), cast=float).rename(columns={"id":"lab_ror","value":"count"})
    labs_f = labs_pct_df.merge(labs_cnt_df, on="lab_ror", how="left")
    labs_f = labs_f.merge(LAB_NAMES, on="lab_ror", how="left")
    labs_f["lab_label"] = labs_f["lab_name"].fillna(labs_f["lab_ror"])
    labs_f = labs_f[pd.to_numeric(labs_f["ratio"], errors="coerce") >= 0.02].copy()
    labs_f = labs_f.sort_values("ratio", ascending=False)

    cA, cB = st.columns(2)
    with cA:
        if labs_f.empty:
            st.info("No labs above the 2% threshold for this field.")
        else:
            order = labs_f["lab_label"].tolist()
            st.altair_chart(
                percent_bar_with_counts(
                    labs_f, y="lab_label", pct_col="ratio", count_col="count",
                    title=f"{sel_field_label} — labs’ share of field (≥2%)",
                    color_value=sel_field_color, order=order, height=max(26*len(order), 240)
                ),
                use_container_width=True,
            )

    # Whiskers by lab for selected field (only those shown at left)
    def _lab_whisk_f(colname):
        return parse_id_value_list(frow.get(colname), cast=float).rename(columns={"id":"lab_ror"})
    wf = (_lab_whisk_f("By lab: FWCI_FR min")
          .merge(_lab_whisk_f("By lab: FWCI_FR Q1"), on="lab_ror", how="outer", suffixes=("","_q1"))
          .merge(_lab_whisk_f("By lab: FWCI_FR Q2"), on="lab_ror", how="outer", suffixes=("","_q2"))
          .merge(_lab_whisk_f("By lab: FWCI_FR Q3"), on="lab_ror", how="outer", suffixes=("","_q3"))
          .merge(_lab_whisk_f("By lab: FWCI_FR max"), on="lab_ror", how="outer", suffixes=("","_max"))
    )
    wf = wf.rename(columns={"value":"min","value_q1":"q1","value_q2":"median","value_q3":"q3","value_max":"max"})
    wf = wf.merge(labs_f[["lab_ror","lab_label"]], on="lab_ror", how="inner")

    with cB:
        if wf.empty:
            st.info("No FWCI_FR distribution available for selected labs.")
        else:
            st.altair_chart(
                whisker_chart(
                    wf.assign(Domain=sel_field_domain)[["lab_label","min","q1","median","q3","max","Domain"]],
                    y_field="lab_label", color_field="Domain", title=f"{sel_field_label} — FWCI_FR spread for labs (log)",
                    log=True, palette={str(sel_field_domain): sel_field_color}, fixed_order=labs_f["lab_label"].tolist(), height=max(26*len(labs_f), 240)
                ),
                use_container_width=True,
            )

    st.divider()

    # ----- Top partners (field) -----
    st.markdown("#### Top partners")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top FR partners (up to 10)**")
        fr = parse_parallel_lists(
            name=frow.get("Top 10 FR partners (name)"),
            type=frow.get("Top 10 FR partners (type)"),
            copubs=frow.get("Top 10 FR partners (totals copubs in this field)"),
            share=frow.get("Top 10 FR partners (% of UL total copubs)"),
        )
        for col in ["copubs","share"]:
            if col in fr: fr[col] = pd.to_numeric(fr[col].str.replace(",", "."), errors="coerce")
        st.dataframe(
            fr.head(20),
            width="stretch",
            hide_index=True,
            column_config={
                "name":"Partner",
                "type":"Type",
                "copubs": st.column_config.NumberColumn("Co-pubs in field", format="%.0f"),
                "share": st.column_config.NumberColumn("% of UL co-pubs with partner", format="%.1f%%"),
            },
        )

    with c2:
        st.markdown("**Top international partners (up to 10)**")
        intl = parse_parallel_lists(
            name=frow.get("Top 10 int partners (name)"),
            type=frow.get("Top 10 int partners (type)"),
            country=frow.get("Top 10 int partners (country)"),
            copubs=frow.get("Top 10 int partners (totals copubs in this field)"),
            share=frow.get("Top 10 int partners (% of UL total copubs)"),
        )
        for col in ["copubs","share"]:
            if col in intl: intl[col] = pd.to_numeric(intl[col].str.replace(",", "."), errors="coerce")
        st.dataframe(
            intl.head(20),
            width="stretch",
            hide_index=True,
            column_config={
                "name":"Partner",
                "type":"Type",
                "country":"Country",
                "copubs": st.column_config.NumberColumn("Co-pubs in field", format="%.0f"),
                "share": st.column_config.NumberColumn("% of UL co-pubs with partner", format="%.1f%%"),
            },
        )

    st.divider()

    # ----- Top authors (field) -----
    st.markdown("#### Top authors")
    authors = parse_parallel_lists(
        name=frow.get("Top 10 authors (name)"),
        pubs=frow.get("Top 10 authors (pubs)"),
        orcid=frow.get("Top 10 authors (Orcid)"),
        author_id=frow.get("Top 10 authors (ID)"),
        fwci=frow.get("Top 10 authors (Average FWCI_FR)"),
        t10=frow.get("Top 10 authors (PPtop10% Count)"),
        t01=frow.get("Top 10 authors (PPtop1% Count)"),
        lorraine=frow.get("Top 10 authors (Is Lorraine)"),
        labs=frow.get("Top 10 authors (Labs)"),
    )
    for col in ["pubs","fwci","t10","t01"]:
        if col in authors: authors[col] = pd.to_numeric(authors[col].str.replace(",", "."), errors="coerce")
    if "lorraine" in authors:
        authors["lorraine"] = authors["lorraine"].str.strip().str.lower().map({"true": True, "false": False}).fillna(authors["lorraine"])
    authors = authors.sort_values(["pubs","fwci"], ascending=[False, False])

    st.dataframe(
        authors.head(20)[["name","pubs","fwci","t10","t01","lorraine","orcid","author_id","labs"]],
        width="stretch",
        hide_index=True,
        column_config={
            "name":"Author",
            "pubs": st.column_config.NumberColumn("Pubs", format="%.0f"),
            "fwci": st.column_config.NumberColumn("Average FWCI_FR", format="%.2f"),
            "t10": st.column_config.NumberColumn("PPtop10% Count", format="%.0f"),
            "t01": st.column_config.NumberColumn("PPtop1% Count", format="%.0f"),
            "lorraine":"Is Lorraine",
            "orcid": st.column_config.TextColumn("ORCID"),
            "author_id": st.column_config.TextColumn("Author ID"),
            "labs": "Lab(s)",
        },
    )

    st.divider()

    # ----- Thematic shape: subfield distribution within selected field -----
    st.markdown("#### Thematic shape — subfield distribution within field")
    sf_pct = parse_id_value_list(frow.get("By subfield: % of field pubs"), cast=float).rename(columns={"id":"subfield_id","value":"ratio"})
    sf_pct["subfield_id"] = sf_pct["subfield_id"].astype(str)
    sf_cnt = parse_id_value_list(frow.get("By subfield: count"), cast=float).rename(columns={"id":"subfield_id","value":"count"})
    sf_cnt["subfield_id"] = sf_cnt["subfield_id"].astype(str)

    # canonical subfields for this field
    sub_canon = (TOP[pd.to_numeric(TOP["field_id"], errors="coerce").eq(sel_field_id)][["subfield_id","subfield_name","domain_name"]]
                 .drop_duplicates()
                 .sort_values("subfield_id"))
    sub_canon["subfield_id"] = sub_canon["subfield_id"].astype(int).astype(str)

    sub_mix = sub_canon.merge(sf_pct, on="subfield_id", how="left").merge(sf_cnt, on="subfield_id", how="left")
    sub_mix["ratio"] = pd.to_numeric(sub_mix["ratio"], errors="coerce").fillna(0.0)
    sub_mix["count"] = pd.to_numeric(sub_mix["count"], errors="coerce").fillna(0.0)
    order_sub = sub_mix["subfield_name"].tolist()

    st.altair_chart(
        percent_bar_with_counts(
            sub_mix.rename(columns={"subfield_name":"Subfield"}),
            y="Subfield", pct_col="ratio", count_col="count",
            title=f"{sel_field_label} — subfield mix (% of field)",
            color_value=sel_field_color, order=order_sub, height=max(26*len(order_sub), 260)
        ),
        use_container_width=True,
    )
