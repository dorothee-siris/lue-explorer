# lib/data_io.py
from __future__ import annotations

import os
import re
from typing import List, Optional

import pandas as pd
import streamlit as st

from lib.constants import YEAR_START, YEAR_END

# -------------------------------------------------------------------
# Basics
# -------------------------------------------------------------------
DATA_DIR = os.environ.get("DATA_PATH") or os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data"
)
LEAD_IDX_RE = re.compile(r"^\[\d+\]\s*")  # strip leading [1] markers in pipe-lists


@st.cache_data(show_spinner=False)
def load_parquet(name: str, data_path: Optional[str] = None) -> pd.DataFrame:
    base = data_path or DATA_DIR
    path = name if os.path.isabs(name) else os.path.join(base, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected file not found: {path}")
    return pd.read_parquet(path)


def _split_positions(s: object) -> list[str]:
    """Split pipe/semicolon lists like '[1] Foo | [2] Bar' → ['Foo','Bar']."""
    if s is None or pd.isna(s):
        return []
    raw = str(s).replace(" ;", "|").replace("; ", "|").replace(";", "|")
    raw = LEAD_IDX_RE.sub("", raw)
    toks = [t.strip() for t in raw.split("|")]
    return [t for t in toks if t]


# -------------------------------------------------------------------
# Loaders: core + dictionaries (NEW schema)
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_core(data_path: Optional[str] = None) -> pd.DataFrame:
    """pubs_final.parquet → normalized snake_case columns."""
    df = load_parquet("pubs_final.parquet", data_path).copy()
    ren = {
        "OpenAlex ID": "openalex_id",
        "DOI": "doi",
        "Title": "title",
        "Publication Year": "year",
        "Publication Type": "type",
        "Authors": "authors",
        "Authors ID": "authors_id",
        "Authors ORCID": "authors_orcid",
        "Institutions": "institutions",
        "Institution Types": "inst_types",
        "Institution Countries": "inst_countries",
        "Institutions ID": "inst_ids",
        "Institutions ROR": "inst_rors",
        "FWCI_all": "fwci_all",
        "FWCI_FR": "fwci_fr",
        "Citation Count": "citation_count",
        "Citations per Year": "cites_per_year",
        "Primary Topic": "primary_topic_id",
        "Primary Subfield ID": "primary_subfield_id",
        "Primary Field ID": "primary_field_id",
        "Primary Domain ID": "primary_domain_id",
        "In_LUE": "in_lue",
        "Labs_RORs": "labs_rors",
        "Is_PPtop10%_(field)": "is_pp10_field",
        "Is_PPtop1%_(field)": "is_pp1_field",
        "Is_PPtop10%_(subfield)": "is_pp10_subfield",
        "Is_PPtop1%_(subfield)": "is_pp1_subfield",
    }
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})

    for c in ["year", "citation_count", "primary_field_id", "primary_subfield_id", "primary_domain_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "in_lue" in df.columns:
        df["in_lue"] = df["in_lue"].fillna(False).astype(bool)
    return df


@st.cache_data(show_spinner=False)
def load_internal(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    dict_internal.parquet (new): lab roster + precomputed KPIs.
    Normalizes key columns used by the Lab View table.
    """
    df = load_parquet("dict_internal.parquet", data_path).copy()
    low = {c.lower(): c for c in df.columns}

    # required / commonly used columns
    map_cols = {
        low.get("unit ror", "Unit ROR"): "lab_ror",
        low.get("unit name", "Unit Name"): "lab_name",
        low.get("total publications", "Total publications"): "pubs_19_23",
        low.get("% of lorraine production", "% of Lorraine production"): "share_of_dataset_works",
        # optional extras (may not exist depending on your build)
        low.get("international collabs (ratio)", "International collabs (ratio)"): "intl_ratio",
        low.get("company collabs (abs ratio)", "Company collabs (abs ratio)"): "company_ratio",
    }
    df = df.rename(columns={k: v for k, v in map_cols.items() if k in df.columns})

    for c in ["pubs_19_23", "share_of_dataset_works", "intl_ratio", "company_ratio"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def load_all_partners(data_path: Optional[str] = None) -> pd.DataFrame:
    """dict_all_partners.parquet → partner_name/type/country + IDs/ROR."""
    df = load_parquet("dict_all_partners.parquet", data_path).copy()
    ren = {
        "Institution Name": "partner_name",
        "Institution Type": "partner_type",
        "Country": "country",
        "Institution ID": "inst_id",
        "Institution ROR": "inst_ror",
        "Copublications": "copubs",
    }
    return df.rename(columns=ren)


@st.cache_data(show_spinner=False)
def load_top_partners(data_path: Optional[str] = None) -> pd.DataFrame:
    """dict_top_partners.parquet → enriched KPIs for ~3k partners (≥5 co-pubs)."""
    df = load_parquet("dict_top_partners.parquet", data_path).copy()
    ren = {
        "Institutions ID": "inst_id",
        "Institutions ROR": "inst_ror",
        "Copublications": "copubs",
        "dont LUE": "lue_count",
        "Average FWCI_all": "avg_fwci_all",
        "Average FWCI_FR": "avg_fwci_fr",
        "ratio_copubs_vs_partner_total": "share_partner_output",  # UL share of partner output
        "ratio_copubs_vs_UL_total": "share_of_ul_output",        # partner share of UL output
        "Fields distribution (count ; FWCI_FR ; top10 ; top1)": "fields_details",
        "Fields_UL_ratio": "fields_ul_ratio",
        "Subfields distribution (count ; FWCI_FR ; top10 ; top1)": "subfields_details",
        "Subfields_UL_ratio": "subfields_ul_ratio",
        "total_works_2019_23_articles_books_chapters_reviews": "partner_total_works",
    }
    df = df.rename(columns=ren)

    for c in ["copubs", "avg_fwci_fr", "share_partner_output", "share_of_ul_output", "partner_total_works"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_topics(data_path: Optional[str] = None) -> pd.DataFrame:
    """all_topics.parquet → topic/subfield/field/domain IDs + names."""
    df = load_parquet("all_topics.parquet", data_path).copy()
    keep = [
        "topic_id",
        "topic_name",
        "subfield_id",
        "subfield_name",
        "field_id",
        "field_name",
        "domain_id",
        "domain_name",
    ]
    return df[keep].drop_duplicates()


@st.cache_data(show_spinner=False)
def load_fields_table(data_path: Optional[str] = None) -> pd.DataFrame:
    """dict_fields.parquet (UL global distribution by field)."""
    return load_parquet("dict_fields.parquet", data_path).copy()


@st.cache_data(show_spinner=False)
def load_domains_table(data_path: Optional[str] = None) -> pd.DataFrame:
    """dict_domains.parquet (UL global distribution by domain)."""
    return load_parquet("dict_domains.parquet", data_path).copy()


@st.cache_data(show_spinner=False)
def load_authors_lookup(data_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    dict_authors.parquet (optional enrichment table).
    Expected columns (case-insensitive):
      Author Name, Author ID, ORCID, Is Lorraine, Lab(s), Publications (unique),
      Average FWCI_all / Average FWCI_FR
    """
    try:
        df = load_parquet("dict_authors.parquet", data_path).copy()
    except Exception:
        return None

    low = {c.lower(): c for c in df.columns}
    ren = {
        low.get("author name", "Author Name"): "author_name",
        low.get("author id", "Author ID"): "author_id",
        low.get("orcid", "ORCID"): "orcid",
        low.get("is lorraine", "Is Lorraine"): "is_lorraine",
        low.get("lab(s)", "Lab(s)"): "labs_from_dict",
        low.get("publications (unique)", "Publications (unique)"): "total_pubs",
        low.get("average fwci_all", "Average FWCI_all"): "avg_fwci_all",
        low.get("average fwci_fr", "Average FWCI_FR"): "avg_fwci_fr",
    }
    df = df.rename(columns=ren)
    return df[
        ["author_id", "author_name", "orcid", "is_lorraine", "labs_from_dict", "total_pubs", "avg_fwci_all", "avg_fwci_fr"]
    ]


# -------------------------------------------------------------------
# Exploders / aggregates
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def explode_labs(pubs: pd.DataFrame) -> pd.DataFrame:
    """One row per (work, lab_ror)."""
    if "labs_rors" not in pubs.columns:
        return pd.DataFrame(columns=["openalex_id", "lab_ror", "year"])
    rows = []
    for _, r in pubs[["openalex_id", "labs_rors", "year"]].iterrows():
        for rr in _split_positions(r["labs_rors"]):
            rows.append({"openalex_id": r["openalex_id"], "lab_ror": rr, "year": r["year"]})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def explode_institutions(pubs: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (work, institution): openalex_id, inst_id, inst_ror, inst_type, inst_country, year, fwci_fr, in_lue
    """
    df = pubs.copy()
    for c in ["openalex_id", "year", "fwci_fr", "in_lue", "inst_ids", "inst_rors", "inst_types", "inst_countries"]:
        if c not in df.columns:
            df[c] = None

    rows = []
    for _, r in df.iterrows():
        ids = _split_positions(r["inst_ids"])
        rors = _split_positions(r["inst_rors"])
        tys = _split_positions(r["inst_types"])
        cts = _split_positions(r["inst_countries"])
        L = max(len(ids), len(rors), len(tys), len(cts), 1)
        ids += [""] * (L - len(ids))
        rors += [""] * (L - len(rors))
        tys += [""] * (L - len(tys))
        cts += [""] * (L - len(cts))
        for i in range(L):
            rows.append(
                {
                    "openalex_id": r["openalex_id"],
                    "inst_id": ids[i] or None,
                    "inst_ror": rors[i] or None,
                    "inst_type": tys[i] or None,
                    "inst_country": cts[i] or None,
                    "year": r["year"],
                    "fwci_fr": r["fwci_fr"],
                    "in_lue": bool(r["in_lue"]) if r["in_lue"] is not None else False,
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def explode_authors(pubs: pd.DataFrame) -> pd.DataFrame:
    """
    Turn (Authors, Authors ID) into rows: openalex_id, author_id, author_name, fwci_fr, year.
    """
    need = ["openalex_id", "authors", "authors_id", "fwci_fr", "year"]
    for c in need:
        if c not in pubs.columns:
            pubs[c] = None

    rows = []
    for _, r in pubs[need].iterrows():
        names = [LEAD_IDX_RE.sub("", x).strip() for x in str(r["authors"] or "").split("|") if x.strip()]
        ids = [LEAD_IDX_RE.sub("", x).strip() for x in str(r["authors_id"] or "").split("|") if x.strip()]
        if len(names) < len(ids):
            names += [""] * (len(ids) - len(names))
        if len(ids) < len(names):
            ids += [""] * (len(names) - len(ids))
        for nm, aid in zip(names, ids):
            if not aid and not nm:
                continue
            rows.append({"openalex_id": r["openalex_id"], "author_id": aid, "author_name": nm, "fwci_fr": r["fwci_fr"], "year": r["year"]})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def author_global_metrics(pubs: pd.DataFrame) -> pd.DataFrame:
    """
    Global metrics across the whole dataset per author_id:
      total_pubs, avg_fwci_overall (FWCI_FR), labs_concat (if labs_rors exist)
    """
    ea = explode_authors(pubs)
    if ea.empty:
        return pd.DataFrame(columns=["author_id", "author_name", "total_pubs", "avg_fwci_overall", "labs_concat"])

    # best-effort labs concat (from core authorship) — optional
    if "labs_rors" in pubs.columns:
        labmap = explode_labs(pubs).groupby("openalex_id")["lab_ror"].apply(lambda s: " | ".join(sorted(set(s)))).to_dict()
        ea["labs_concat"] = ea["openalex_id"].map(labmap)
    else:
        ea["labs_concat"] = ""

    g = ea.groupby(["author_id", "author_name"], as_index=False).agg(
        total_pubs=("openalex_id", "nunique"),
        avg_fwci_overall=("fwci_fr", "mean"),
        labs_concat=("labs_concat", lambda s: " | ".join(sorted({x for x in "|".join([str(v) for v in s]).split("|") if x}))),
    )
    return g


# -------------------------------------------------------------------
# Lab tables & field distributions
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def lab_summary_table_from_internal(pubs: pd.DataFrame, internal: pd.DataFrame) -> pd.DataFrame:
    """
    Build the per-lab summary used in the Lab View table.
    Columns: lab_name, lab_ror, pubs_19_23, share_of_dataset_works, avg_fwci, openalex_ui_url, ror_url
    """
    df = internal.copy()

    # Compute avg FWCI per lab from core (FWCI_FR preferred)
    el = explode_labs(pubs)
    core = pubs[["openalex_id", "fwci_fr"]].merge(el, on="openalex_id", how="left")
    avg_fwci = core.groupby("lab_ror", as_index=False)["fwci_fr"].mean().rename(columns={"fwci_fr": "avg_fwci"})

    out = df.merge(avg_fwci, on="lab_ror", how="left")

    # Links (ROR + OpenAlex filter by ROR)
    def openalex_ui_for_ror(ror: str, y0=YEAR_START, y1=YEAR_END):
        return (
            "https://openalex.org/works?"
            f"page=1&filter=authorships.institutions.ror:{ror},"
            f"type:types/article|types/book-chapter|types/review|types/book,"
            f"publication_year:{y0}-{y1}"
        )

    out["openalex_ui_url"] = out["lab_ror"].fillna("").map(lambda r: openalex_ui_for_ror(r))
    out["ror_url"] = out["lab_ror"].fillna("").map(lambda r: f"https://ror.org/{r}" if r else "")

    return out[
        ["lab_name", "lab_ror", "pubs_19_23", "share_of_dataset_works", "avg_fwci", "openalex_ui_url", "ror_url"]
    ].sort_values("pubs_19_23", ascending=False)


@st.cache_data(show_spinner=False)
def lab_field_counts(pubs: pd.DataFrame, years: Optional[List[int]] = None) -> pd.DataFrame:
    """
    (Compat helper used by pages) Compute counts by (lab_ror, field) for selected years.
    Uses core 'primary_field_id'. Also returns in_lue_count.
    Output columns:
      lab_ror, field_id, field_name, domain_name, field, domain, count, in_lue_count
    """
    if years is None:
        years = list(range(YEAR_START, YEAR_END + 1))

    el = explode_labs(pubs)
    subset = pubs[pubs["year"].isin(years)][["openalex_id", "primary_field_id", "in_lue"]].merge(
        el, on="openalex_id", how="left"
    )
    subset["in_lue"] = subset["in_lue"].fillna(False).astype(bool)

    g = subset.groupby(["lab_ror", "primary_field_id"], as_index=False).agg(
        count=("openalex_id", "nunique"),
        in_lue_count=("in_lue", "sum"),
    )

    topics = load_topics()
    look = topics.drop_duplicates("field_id")[["field_id", "field_name", "domain_name"]]
    out = g.merge(look, left_on="primary_field_id", right_on="field_id", how="left")
    out["field_name"] = out["field_name"].fillna("Unknown")
    out["domain_name"] = out["domain_name"].fillna("Other")

    # aliases expected by some chart builders
    out["field"] = out["field_name"]
    out["domain"] = out["domain_name"]

    return out[
        ["lab_ror", "field_id", "field_name", "domain_name", "field", "domain", "count", "in_lue_count"]
    ]


@st.cache_data(show_spinner=False)
def lab_field_counts_from_core(pubs: pd.DataFrame, lab_ror: str, years: List[int]) -> pd.DataFrame:
    """Same as above but already filtered to a single lab."""
    df = lab_field_counts(pubs, years)
    return df[df["lab_ror"] == lab_ror].copy()


@st.cache_data(show_spinner=False)
def ul_field_counts_from_fields_table() -> pd.DataFrame:
    """
    UL total field mix from dict_fields.parquet.
    Returns: field_id, field_name, domain_name, count
    """
    ftab = load_fields_table()
    topics = load_topics()

    # Normalize field id/name if needed
    low = {c.lower(): c for c in ftab.columns}
    fid = low.get("field_id", "Field_ID")
    fname = low.get("field_name", "Field_Name")
    fcount = low.get("count", "Count")

    out = ftab.rename(columns={fid: "field_id", fname: "field_name", fcount: "count"})
    out["field_id"] = pd.to_numeric(out["field_id"], errors="coerce")

    look = topics.drop_duplicates("field_id")[["field_id", "domain_name"]]
    out = out.merge(look, on="field_id", how="left")
    out["domain_name"] = out["domain_name"].fillna("Other")
    return out[["field_id", "field_name", "domain_name", "count"]]


# -------------------------------------------------------------------
# Partners: unified view + parsing helpers
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def partners_joined() -> pd.DataFrame:
    """
    Join dict_top_partners (metrics) with dict_all_partners (name/type/country).
    Columns:
      partner_name, partner_type, country, inst_id, inst_ror, copubs,
      avg_fwci_fr, share_partner_output, share_of_ul_output,
      partner_total_works, fields_details, subfields_details
    """
    allp = load_all_partners()
    top = load_top_partners()

    out = pd.merge(
        top,
        allp[["partner_name", "partner_type", "country", "inst_id", "inst_ror"]],
        on=["inst_ror"],
        how="left",
        suffixes=("", "_all"),
    )
    # Where ROR join failed, try inst_id
    miss = out["partner_name"].isna()
    if miss.any():
        fill = out.loc[miss, ["inst_id"]].merge(
            allp[["partner_name", "partner_type", "country", "inst_id"]],
            on="inst_id",
            how="left",
        )
        out.loc[miss, ["partner_name", "partner_type", "country"]] = fill[
            ["partner_name", "partner_type", "country"]
        ].values

    for c in ["copubs", "avg_fwci_fr", "share_partner_output", "share_of_ul_output", "partner_total_works"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def explode_field_details(details: object) -> pd.DataFrame:
    """
    Parse strings like:
      "2505 (632 ; 1.33 ; 52 ; 2) | 2204 (359 ; 1.39 ; 61 ; 5)"
    → columns: id, count, fwci, top10, top1
    """
    rows = []
    if details is None or pd.isna(details) or not str(details).strip():
        return pd.DataFrame(columns=["id", "count", "fwci", "top10", "top1"])

    for tok in str(details).split("|"):
        tok = tok.strip()
        if not tok or "(" not in tok:
            continue
        left, right = tok.split("(", 1)
        key = left.strip()
        right = right.strip().rstrip(")")
        parts = [p.strip() for p in right.split(";")]
        # coerce fields
        def f2float(x):
            try:
                return float(str(x).replace(",", "."))
            except Exception:
                return None

        cnt = parts[0] if parts else None
        fwci = parts[1] if len(parts) > 1 else None
        t10 = parts[2] if len(parts) > 2 else None
        t01 = parts[3] if len(parts) > 3 else None

        try:
            key_num = int(float(key))
        except Exception:
            # if key is not numeric, just keep as string
            key_num = key

        try:
            cnt_val = int(float(str(cnt).replace(",", "."))) if cnt is not None else 0
        except Exception:
            cnt_val = 0

        rows.append(
            {
                "id": key_num,
                "count": cnt_val,
                "fwci": f2float(fwci),
                "top10": int(f2float(t10) or 0),
                "top1": int(f2float(t01) or 0),
            }
        )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Topline KPIs used in Home/Lab pages
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def topline_metrics(pubs: pd.DataFrame, internal: Optional[pd.DataFrame] = None) -> dict:
    # Number of labs (prefer dict_internal, fallback to pubs)
    if internal is not None and not internal.empty and "lab_ror" in internal.columns:
        n_labs = int(internal["lab_ror"].nunique())
    else:
        n_labs = int(explode_labs(pubs)["lab_ror"].nunique())

    df = pubs[(pubs["year"] >= YEAR_START) & (pubs["year"] <= YEAR_END)]
    any_lab_mask = df["labs_rors"].fillna("").astype(str).str.len() > 0 if "labs_rors" in df.columns else False
    covered = int(df.loc[any_lab_mask, "openalex_id"].nunique()) if isinstance(any_lab_mask, pd.Series) else 0
    total = int(df["openalex_id"].nunique()) if "openalex_id" in df.columns else 0
    pct = (covered / total) if total else 0.0

    return {
        "n_labs": n_labs,
        "%_covered_by_labs": float(pct),
        "n_pubs_total_19_23": total,
        "n_pubs_lab_19_23": covered,
    }
