# lib/data_io.py
from __future__ import annotations

import os
import re
import pandas as pd
import streamlit as st

from lib.constants import YEAR_START, YEAR_END, ROR_URL, OPENALEX_WORKS_FOR_ROR

DEFAULT_DATA_PATH = os.environ.get("DATA_PATH")
PIPE_SPLIT_RE = re.compile(r"\s*\|\s*")


@st.cache_data(show_spinner=False)
def load_parquet(name: str, data_path: str | None = None) -> pd.DataFrame:
    base = data_path or DEFAULT_DATA_PATH or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    path = os.path.join(base, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected file not found: {path}")
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_core(data_path: str | None = None) -> pd.DataFrame:
    df = load_parquet("pubs_final.parquet", data_path)
    rename_map = {
        "OpenAlex ID": "openalex_id",
        "Publication Year": "year",
        "Field-Weighted Citation Impact": "fwci",
        "FWCI": "fwci",
        "All Fields": "all_fields",
        "In_LUE": "in_lue",
        "Labs_RORs": "labs_rors",
        "Labs_Names": "labs_names",
    }
    rename_map.update({
        "Authors": "authors",
        "Authors ID": "authors_id",
        "Authors ORCID": "authors_orcid",
        "Publication Type": "pub_type",
        "Citation Count": "citation_count",
        "Title": "title",
        "Institution Types": "inst_types",
        "Institution Countries": "inst_countries",
    })


    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    needed = ["openalex_id", "year", "fwci", "all_fields", "in_lue", "labs_rors", "labs_names"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"pubs_final.parquet is missing columns: {miss}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["fwci"] = pd.to_numeric(df["fwci"], errors="coerce")
    df["in_lue"] = df["in_lue"].fillna(False).astype(bool)
    return df


def _parse_ratio_triplet(cell: object) -> float | None:
    """
    Accepts strings like '1071 ; 0.6106 ; 1.32' or '1071;0,6106;1,32'
    Returns the ratio (2nd element) as float in [0,1], or None.
    """
    if cell is None:
        return None
    s = str(cell)
    parts = [p.strip().replace(",", ".") for p in s.split(";")]
    if len(parts) >= 2:
        try:
            return float(parts[1])
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def load_internal(data_path: str | None = None) -> pd.DataFrame:
    """
    Load dict_internal.parquet and normalize columns we need.
    """
    df = load_parquet("dict_internal.parquet", data_path)

    # Column name normalization
    cols = {c.lower(): c for c in df.columns}
    unit_ror = cols.get("unit ror", cols.get("unit_ror", "Unit ROR"))
    inst_id = cols.get("institution id", cols.get("institution_id", "Institution ID"))
    labo = cols.get("laboratoire", "Laboratoire")
    stype = cols.get("structure type", cols.get("structure_type", "structure type"))
    pubs = cols.get("publications (unique)", "Publications (unique)")
    dont_lue = cols.get("dont lue", "dont LUE")
    fwci = cols.get("average fwci", "Average FWCI")
    share = cols.get("% of ul production", "% of UL production")
    intl = cols.get("international collaborations (count ; ratio ; avgfwci)",
                    "International collaborations (count ; ratio ; avgFWCI)")
    comp = cols.get("company collaborations (count ; ratio ; avgfwci)",
                    "Company collaborations (count ; ratio ; avgFWCI)")

    df = df.rename(
        columns={
            unit_ror: "unit_ror",
            inst_id: "institution_id",
            labo: "laboratoire",
            stype: "structure_type",
            pubs: "pubs_unique",
            dont_lue: "dont_lue",
            fwci: "avg_fwci",
            share: "share_ul",
            intl: "intl_triplet",
            comp: "company_triplet",
        }
    )

    # Keep only labs
    labs = df[df["structure_type"].astype(str).str.lower().eq("lab")].copy()

    # Types
    labs["pubs_unique"] = pd.to_numeric(labs["pubs_unique"], errors="coerce")
    labs["dont_lue"] = pd.to_numeric(labs["dont_lue"], errors="coerce")
    labs["avg_fwci"] = pd.to_numeric(labs["avg_fwci"], errors="coerce")

    # Convert share from strings like '6,61%' or decimals to 0..1
    labs["share_ul"] = (
        labs["share_ul"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    labs["share_ul"] = pd.to_numeric(labs["share_ul"], errors="coerce")
    # If they already are 0..1, keep; if they look like 6.61 -> divide by 100
    labs["share_ul"] = labs["share_ul"].where(labs["share_ul"] <= 1, labs["share_ul"] / 100.0)

    # Parse ratios
    labs["intl_ratio"] = labs["intl_triplet"].map(_parse_ratio_triplet)
    labs["company_ratio"] = labs["company_triplet"].map(_parse_ratio_triplet)

    return labs


def _openalex_ui_link(institution_id: str, year_min: int, year_max: int) -> str:
    """
    UI link with lineage filter and allowed types.
    Example: https://openalex.org/works?page=1&filter=authorships.institutions.lineage:i4210147295,type:types/article|types/book-chapter|types/review|types/book,publication_year:2019-2023
    """
    if not isinstance(institution_id, str) or not institution_id:
        return ""
    iid = institution_id.strip().lower()  # 'I421...' -> 'i421...'
    types = "types/article|types/book-chapter|types/review|types/book"
    return (
        f"https://openalex.org/works?page=1&filter="
        f"authorships.institutions.lineage:{iid},"
        f"type:{types},publication_year:{year_min}-{year_max}"
    )


@st.cache_data(show_spinner=False)
def explode_labs(df: pd.DataFrame) -> pd.DataFrame:
    base = df[["openalex_id", "year", "fwci", "in_lue", "all_fields", "labs_rors", "labs_names"]].copy()
    base = base[base["labs_rors"].notna() & (base["labs_rors"].astype(str).str.len() > 0)].reset_index(drop=True)

    rors_series = base["labs_rors"].astype(str).str.split(PIPE_SPLIT_RE)
    names_series = base["labs_names"].fillna("").astype(str).str.split(PIPE_SPLIT_RE)

    rows = []
    for i in range(len(base)):
        rlist = rors_series.iloc[i]
        nlist = names_series.iloc[i]
        if len(nlist) < len(rlist):
            nlist = nlist + [""] * (len(rlist) - len(nlist))
        for r, n in zip(rlist, nlist):
            if not r:
                continue
            rec = base.iloc[i].to_dict()
            rec.update({"lab_ror": r.strip(), "lab_name": n.strip()})
            rows.append(rec)

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def explode_fields(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df[["openalex_id", "all_fields"]].dropna().copy()
    tmp["all_fields"] = tmp["all_fields"].astype(str).str.split(PIPE_SPLIT_RE)
    tmp = tmp.explode("all_fields").dropna()
    tmp["field"] = tmp["all_fields"].astype(str).str.strip()
    tmp = tmp.drop(columns=["all_fields"]).drop_duplicates()
    return tmp


@st.cache_data(show_spinner=False)
def lab_field_counts(pubs: pd.DataFrame, years: list[int] | None = None) -> pd.DataFrame:
    """(lab_ror, field) counts + in_lue counts for selected years (defaults to YEAR_START..YEAR_END)."""
    lab_pubs = explode_labs(pubs)
    if years is None:
        years = list(range(YEAR_START, YEAR_END + 1))
    lab_pubs = lab_pubs[lab_pubs["year"].isin(years)].copy()

    fields = explode_fields(pubs)
    wf = lab_pubs.merge(fields, on="openalex_id", how="left").dropna(subset=["field"])

    grp = wf.groupby(["lab_ror", "field"], as_index=False).agg(
        count=("openalex_id", "nunique"),
        in_lue_count=("in_lue", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
    )
    return grp


@st.cache_data(show_spinner=False)
def lab_summary_table_from_internal(
    internal: pd.DataFrame, year_min: int, year_max: int
) -> pd.DataFrame:
    """
    Build the per-lab table directly from dict_internal.
    Keeps numeric ratios in 0..1 for logic, and we'll create 0..100 display series in the page.
    """
    g = internal.copy()

    g["lab_name"] = g["laboratoire"]
    g["pubs_19_23"] = (
        pd.to_numeric(g["pubs_unique"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # ratios in 0..1
    g["share_of_dataset_works"] = pd.to_numeric(g["share_ul"], errors="coerce").fillna(0.0)
    g["avg_fwci"] = pd.to_numeric(g["avg_fwci"], errors="coerce")

    # % LUE = dont_lue / pubs
    pubs_nonzero = g["pubs_19_23"].replace(0, pd.NA)
    g["lue_pct"] = (pd.to_numeric(g["dont_lue"], errors="coerce") / pubs_nonzero).fillna(0.0)

    # international & company ratios already parsed in load_internal()
    g["intl_pct"] = pd.to_numeric(g["intl_ratio"], errors="coerce").fillna(0.0)
    g["company_pct"] = pd.to_numeric(g["company_ratio"], errors="coerce").fillna(0.0)

    g["ror_url"] = g["unit_ror"].apply(lambda r: ROR_URL.format(ror=r))
    g["openalex_ui_url"] = g["institution_id"].apply(
        lambda iid: _openalex_ui_link(str(iid), year_min, year_max)
    )

    g = g[
        [
            "lab_name",
            "pubs_19_23",
            "share_of_dataset_works",
            "lue_pct",
            "intl_pct",
            "company_pct",
            "avg_fwci",
            "openalex_ui_url",
            "ror_url",
        ]
    ].sort_values("pubs_19_23", ascending=False)

    return g


@st.cache_data(show_spinner=False)
def topline_metrics(pubs: pd.DataFrame, internal: pd.DataFrame | None) -> dict:
    # Number of labs
    n_labs = int(internal["unit_ror"].nunique() if internal is not None and not internal.empty
                 else explode_labs(pubs)["lab_ror"].nunique())

    df = pubs[(pubs["year"] >= YEAR_START) & (pubs["year"] <= YEAR_END)]
    any_lab_mask = df["labs_rors"].fillna("").astype(str).str.len() > 0
    covered = int(df.loc[any_lab_mask, "openalex_id"].nunique())
    total = int(df["openalex_id"].nunique())
    pct = (covered / total) if total else 0.0

    return {
        "n_labs": n_labs,
        "%_covered_by_labs": float(pct),
        "n_pubs_total_19_23": total,
        "n_pubs_lab_19_23": covered,
    }

import re

PIPE_SPLIT_RE2 = re.compile(r"\s*\|\s*")
LEAD_IDX_RE = re.compile(r"^\[\d+\]\s*")

@st.cache_data(show_spinner=False)
def explode_authors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn (Authors, Authors ID) pipe-lists into rows: openalex_id, author_id, author_name.
    Keeps fwci, year, labs_names for later aggregates.
    """
    cols_needed = ["openalex_id", "authors", "authors_id", "fwci", "year", "labs_names"]
    for c in cols_needed:
        if c not in df.columns:
            df[c] = None

    rows = []
    for _, r in df[cols_needed].iterrows():
        names = str(r["authors"] or "").split("|")
        ids   = str(r["authors_id"] or "").split("|")
        # clean
        names = [LEAD_IDX_RE.sub("", n).strip() for n in names if n.strip()]
        ids   = [LEAD_IDX_RE.sub("", a).strip() for a in ids if a.strip()]
        # align lengths
        if len(names) < len(ids): names += [""]*(len(ids)-len(names))
        if len(ids)   < len(names): ids += [""]*(len(names)-len(ids))
        for nm, aid in zip(names, ids):
            if not aid and not nm: 
                continue
            rows.append({
                "openalex_id": r["openalex_id"],
                "author_id": aid,
                "author_name": nm,
                "fwci": r["fwci"],
                "year": r["year"],
                "labs_names": r["labs_names"],
            })
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def author_global_metrics(pubs: pd.DataFrame) -> pd.DataFrame:
    """
    Global metrics across the whole dataset per author_id:
      total_pubs, avg_fwci_overall, labs_concat (unique)
    """
    ea = explode_authors(pubs)
    if ea.empty:
        return pd.DataFrame(columns=["author_id","author_name","total_pubs","avg_fwci_overall","labs_concat"])
    g = ea.groupby(["author_id", "author_name"], as_index=False).agg(
        total_pubs=("openalex_id", "nunique"),
        avg_fwci_overall=("fwci", "mean"),
        labs_concat=("labs_names", lambda s: " | ".join(sorted({x.strip() for x in "|".join([str(v) for v in s]).split("|") if x.strip()}))),
    )
    return g

@st.cache_data(show_spinner=False)
def load_authors_lookup(data_path: str | None = None) -> pd.DataFrame | None:
    """
    Optional: load your authors dictionary (e.g., dict_authors.parquet).
    Expected (flexible) columns: Author ID, ORCID, Is Lorraine, Lab(s), Avg FWCI, Total pubs.
    """
    # try a few common names
    for fname in ["dict_authors.parquet", "authors_dict.parquet", "authors_lookup.parquet"]:
        try:
            df = load_parquet(fname, data_path)
            break
        except Exception:
            df = None
    if df is None:
        return None

    # normalize columns
    low = {c.lower(): c for c in df.columns}
    mapping = {
        low.get("author id", low.get("author_id", "Author ID")): "author_id",
        low.get("orcid", "ORCID"): "orcid",
        low.get("is lorraine", low.get("is_lorraine", "Is Lorraine")): "is_lorraine",
        low.get("lab(s)", low.get("labs", "Lab(s)")): "labs_from_dict",
        low.get("average fwci", low.get("avg fwci", "Average FWCI")): "avg_fwci_overall_dict",
        low.get("total publications", low.get("total pubs", "Total Publications")): "total_pubs_dict",
    }
    df = df.rename(columns=mapping)
    return df

# --- Partners: loader + parsing -----------------------------------------------------
import re
import json
from typing import Iterable

@st.cache_data(show_spinner=False)
def load_partners_ext(data_path: str | None = None) -> pd.DataFrame:
    """
    Load dict_partners_ext.parquet and normalize columns.
    """
    df = load_parquet("dict_partners_ext.parquet", data_path)
    rename = {
        "Institution Name": "partner_name",
        "Institution Type": "partner_type",
        "Country Name": "country",
        "Institutions ID": "inst_id",
        "Institutions ROR": "inst_ror",

        "Publications (unique)": "pubs_partner_ul",
        "dont LUE": "lue_count",
        "Average FWCI": "avg_fwci",
        "% of UL production": "share_of_ul",  # 0..1

        "Labs collaborating (count)": "labs_collab_count",
        "Labs collaborating (details)": "labs_collab_details",

        "Fields distribution (count ; FWCI)": "fields_details",

        "total_works_2019_23_articles_books_chapters_reviews": "partner_total_works",
        "fields_all_count_ratio": "fields_all_count_ratio",  # "Field (count ; ratio)"
        "relative_weight_to_total": "relative_weight_to_total",  # pubs_partner_ul / partner_total_works
        "field_relative_ratios": "field_relative_ratios",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    for c in ["pubs_partner_ul", "lue_count", "avg_fwci", "share_of_ul",
              "partner_total_works", "relative_weight_to_total"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _split_positions(s: str) -> list[str]:
    """Utility: remove [idx] prefixes, split by | and ;, and trim."""
    if s is None or pd.isna(s):
        return []
    raw = str(s)
    # Normalize delimiters
    raw = raw.replace(" ;", "|").replace("; ", "|").replace(";", "|")
    # Drop position markers like [1]
    raw = re.sub(r"\[\d+\]\s*", "", raw)
    toks = [t.strip() for t in raw.split("|")]
    return [t for t in toks if t]


@st.cache_data(show_spinner=False)
def explode_institutions(pubs: pd.DataFrame) -> pd.DataFrame:
    """
    Explode Institutions ID / ROR / Type / Country to one row per (work, institution).
    Returns: columns = openalex_id, inst_id, inst_ror, inst_type, inst_country, year, fwci, in_lue
    """
    cols = {
        "OpenAlex ID": "openalex_id",
        "Institutions ID": "inst_ids_raw",
        "Institutions ROR": "inst_rors_raw",
        "Institution Types": "inst_types_raw",
        "Institution Countries": "inst_countries_raw",
        "Publication Year": "year",
        "Field-Weighted Citation Impact": "fwci",
        "In_LUE": "in_lue",
    }
    df = pubs.rename(columns={k: v for k, v in cols.items() if k in pubs.columns}).copy()

    rows = []
    for _, r in df[["openalex_id", "inst_ids_raw", "inst_rors_raw", "inst_types_raw",
                    "inst_countries_raw", "year", "fwci", "in_lue"]].iterrows():
        ids = _split_positions(r["inst_ids_raw"])
        rors = _split_positions(r["inst_rors_raw"])
        tys = _split_positions(r["inst_types_raw"])
        ctys = _split_positions(r["inst_countries_raw"])

        # pad to max length
        L = max(len(ids), len(rors), len(tys), len(ctys), 1)
        ids += [""] * (L - len(ids))
        rors += [""] * (L - len(rors))
        tys += [""] * (L - len(tys))
        ctys += [""] * (L - len(ctys))

        for i in range(L):
            rows.append({
                "openalex_id": r["openalex_id"],
                "inst_id": ids[i] or None,
                "inst_ror": rors[i] or None,
                "inst_type": tys[i] or None,
                "inst_country": ctys[i] or None,
                "year": r["year"],
                "fwci": r["fwci"],
                "in_lue": bool(r.get("in_lue", False)),
            })
    out = pd.DataFrame(rows)
    return out


def explode_field_details(s: str, value_a: str, value_b: str, a_is_int: bool = True) -> pd.DataFrame:
    """
    Parse strings like:  "Field A (123 ; 1.46) | Field B (45 ; 0.92)"
    Returns DataFrame[field, <value_a>, <value_b>]
    """
    rows = []
    if s is None or pd.isna(s) or not str(s).strip():
        return pd.DataFrame(columns=["field", value_a, value_b])
    for tok in str(s).split("|"):
        tok = tok.strip()
        if not tok:
            continue
        # split 'Name (' and 'count ; val)'
        if "(" in tok:
            name, rest = tok.split("(", 1)
            name = name.strip()
            rest = rest.rstrip(")")
            parts = [p.strip() for p in rest.split(";")]
            a = pd.to_numeric(parts[0].replace(",", "."), errors="coerce")
            b = pd.to_numeric(parts[1].replace(",", "."), errors="coerce") if len(parts) > 1 else None
            if a_is_int:
                a = int(a) if pd.notna(a) else 0
            rows.append({"field": name, value_a: a, value_b: b})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def ul_field_counts_from_internal(internal: pd.DataFrame, prefer: str = "institution_ROR") -> pd.DataFrame:
    """
    Return UL's overall field mix as counts using dict_internal.
    Strategy:
      1) Prefer the row where 'structure type' == prefer (default 'institution_ROR').
      2) Fallback to 'structure type' == 'institution_full'.
      3) As last resort, pick the first row that has a non-empty 'Fields distribution (count ; FWCI)'.
    """
    df = internal.copy()

    # Normalize column names once
    lowmap = {c.lower(): c for c in df.columns}
    col_struct = lowmap.get("structure type")
    col_fields = lowmap.get("fields distribution (count ; fwci)")

    if col_fields is None:
        # Can't parse without the field string
        return pd.DataFrame(columns=["field", "count"])

    row = None
    if col_struct is not None:
        cand = df[df[col_struct].eq(prefer)]
        if cand.empty:
            cand = df[df[col_struct].eq("institution_full")]
        if not cand.empty:
            row = cand.iloc[0]

    if row is None:
        # last resort: first row that actually has the fields string
        nonempty = df[df[col_fields].astype(str).str.strip().ne("")]
        if not nonempty.empty:
            row = nonempty.iloc[0]

    if row is None:
        return pd.DataFrame(columns=["field", "count"])

    fields_s = row.get(col_fields, None)
    out = explode_field_details(fields_s, "count", "fwci", a_is_int=True)
    return out[["field", "count"]]

