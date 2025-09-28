# lib/data_io.py
from __future__ import annotations

import os
import re
import pandas as pd
import streamlit as st

from lib.constants import YEAR_START, YEAR_END, ROR_URL, OPENALEX_WORKS_FOR_ROR

DEFAULT_DATA_PATH = os.environ.get("DATA_PATH")

# Compiled splitter for pipe-separated strings (with spaces)
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
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    needed = ["openalex_id", "year", "fwci", "all_fields", "in_lue", "labs_rors", "labs_names"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"pubs_final.parquet is missing columns: {missing}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["fwci"] = pd.to_numeric(df["fwci"], errors="coerce")
    df["in_lue"] = df["in_lue"].fillna(False).astype(bool)

    return df


@st.cache_data(show_spinner=False)
def load_internal(data_path: str | None = None) -> pd.DataFrame:
    # optional file for friendly lab names & metrics
    df = load_parquet("dict_internal.parquet", data_path)

    cols = {c.lower(): c for c in df.columns}
    unit_ror = cols.get("unit ror", cols.get("unit_ror", "Unit ROR"))
    labo = cols.get("laboratoire", "Laboratoire")
    stype = cols.get("structure type", cols.get("structure_type", "structure type"))
    pubs = cols.get("publications (unique)", "Publications (unique)")
    fwci = cols.get("average fwci", "Average FWCI")
    share = cols.get("% of ul production", "% of UL production")

    df = df.rename(
        columns={
            unit_ror: "unit_ror",
            labo: "laboratoire",
            stype: "structure_type",
            pubs: "pubs_unique",
            fwci: "avg_fwci",
            share: "share_ul",
        }
    )

    labs = df[df["structure_type"].astype(str).str.lower().eq("lab")].copy()
    labs["avg_fwci"] = pd.to_numeric(labs["avg_fwci"], errors="coerce")
    labs["share_ul"] = (
        labs["share_ul"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    labs["share_ul"] = pd.to_numeric(labs["share_ul"], errors="coerce") / 100.0
    return labs


@st.cache_data(show_spinner=False)
def explode_labs(df: pd.DataFrame) -> pd.DataFrame:
    """Row-per-(work,lab) using Labs_RORs/Labs_Names."""
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
    """Explode All Fields per work; ensure unique fields per work."""
    tmp = df[["openalex_id", "all_fields"]].dropna().copy()
    tmp["all_fields"] = tmp["all_fields"].astype(str).str.split(PIPE_SPLIT_RE)
    tmp = tmp.explode("all_fields").dropna()
    tmp["field"] = tmp["all_fields"].astype(str).str.strip()
    tmp = tmp.drop(columns=["all_fields"]).drop_duplicates()
    return tmp


@st.cache_data(show_spinner=False)
def lab_field_counts(pubs: pd.DataFrame) -> pd.DataFrame:
    """(lab_ror, field) counts and in_lue counts for YEAR_START..YEAR_END."""
    lab_pubs = explode_labs(pubs)
    lab_pubs = lab_pubs[(lab_pubs["year"] >= YEAR_START) & (lab_pubs["year"] <= YEAR_END)].copy()

    fields = explode_fields(pubs)
    wf = lab_pubs.merge(fields, on="openalex_id", how="left").dropna(subset=["field"])

    grp = wf.groupby(["lab_ror", "field"], as_index=False).agg(
        count=("openalex_id", "nunique"),
        in_lue_count=("in_lue", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
    )
    return grp


@st.cache_data(show_spinner=False)
def lab_summary_table(pubs: pd.DataFrame, internal: pd.DataFrame | None) -> pd.DataFrame:
    """Per-lab summary: name, pubs 2019–23, share among lab-covered works, avg FWCI, ROR, links."""
    lab_pubs = explode_labs(pubs)
    lab_pubs = lab_pubs[(lab_pubs["year"] >= YEAR_START) & (lab_pubs["year"] <= YEAR_END)].copy()

    works_with_any_lab = lab_pubs["openalex_id"].nunique()

    g = lab_pubs.groupby(["lab_ror"], as_index=False).agg(
        pubs_19_23=("openalex_id", "nunique"),
        avg_fwci=("fwci", "mean"),
    )
    g["share_among_lab_works"] = g["pubs_19_23"].div(works_with_any_lab if works_with_any_lab else 1)

    if internal is not None and not internal.empty:
        lab_names = internal[["unit_ror", "laboratoire"]].drop_duplicates()
        g = g.merge(lab_names, left_on="lab_ror", right_on="unit_ror", how="left")
        g["lab_name"] = g.pop("laboratoire")
    else:
        first_names = (
            lab_pubs.groupby("lab_ror")["lab_name"]
            .agg(lambda s: next((x for x in s if x), ""))
            .reset_index()
        )
        g = g.merge(first_names, on="lab_ror", how="left")

    g["ror_url"] = g["lab_ror"].apply(lambda r: ROR_URL.format(ror=r))
    g["openalex_url"] = g["lab_ror"].apply(lambda r: OPENALEX_WORKS_FOR_ROR.format(ror=r))

    g = g[["lab_name", "pubs_19_23", "share_among_lab_works", "avg_fwci", "lab_ror", "ror_url", "openalex_url"]]
    g = g.sort_values("pubs_19_23", ascending=False)
    return g


@st.cache_data(show_spinner=False)
def topline_metrics(pubs: pd.DataFrame, internal: pd.DataFrame | None) -> dict:
    if internal is not None and not internal.empty:
        n_labs = internal["unit_ror"].nunique()
    else:
        n_labs = explode_labs(pubs)["lab_ror"].nunique()

    df = pubs[(pubs["year"] >= YEAR_START) & (pubs["year"] <= YEAR_END)]
    any_lab_mask = df["labs_rors"].fillna("").astype(str).str.len() > 0
    covered = df.loc[any_lab_mask, "openalex_id"].nunique()
    total = df["openalex_id"].nunique()
    pct = (covered / total) if total else 0.0

    return {
        "n_labs": int(n_labs),
        "%_covered_by_labs": float(pct),
        "n_pubs_total_19_23": int(total),
        "n_pubs_lab_19_23": int(covered),
    }
