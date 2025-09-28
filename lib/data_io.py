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
