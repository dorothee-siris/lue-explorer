# lib/transforms.py
from __future__ import annotations

import re
from typing import Iterable, Tuple

import pandas as pd

# --------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------

def filter_years(
    df: pd.DataFrame,
    year_col: str = "Publication Year",
    years: Iterable[int] = (2019, 2020, 2021, 2022, 2023),
) -> pd.DataFrame:
    """Filter dataframe to selected years (safe if column is str/int)."""
    years = {int(y) for y in years}
    return df[df[year_col].astype(int).isin(years)].copy()


# "id (value) | id (value)" parser (robust to spacing)
_TUPLE = re.compile(r"\s*([^\|]+?)\s*\(\s*([^)]+?)\s*\)\s*")


def _coerce(val: str, typ):
    try:
        return typ(val)
    except Exception:
        # allow things like '11 (970)' where typ=int but has commas/spaces
        try:
            return typ(str(val).replace(",", "").replace(" ", ""))
        except Exception:
            return val


def parse_id_count_series(
    series: pd.Series,
    id_type= str,
    count_type= int,
) -> pd.DataFrame:
    """
    Expand a Series whose cells look like:
      "11 (970) | 13 (1367) | 24 (173)"
    into a long-form DataFrame with columns:
      row_ix, id, count
    Types are coerced via id_type / count_type.
    """
    rows = []
    for ix, cell in series.items():
        if pd.isna(cell):
            continue
        for chunk in str(cell).split("|"):
            m = _TUPLE.search(chunk)
            if not m:
                continue
            _id = _coerce(m.group(1), id_type)
            _cnt = _coerce(m.group(2), count_type)
            rows.append({"row_ix": ix, "id": _id, "count": _cnt})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------
# Lab-related helpers (used in Lab View & collaborations)
# --------------------------------------------------------------------

def explode_labs(pubs: pd.DataFrame, labs_col: str = "Labs_RORs") -> pd.DataFrame:
    """
    One row per (work, lab_ror).
    Accepts lists or strings with '|' / ';' separators.
    """
    def _split(v):
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        return [x.strip() for x in str(v).replace(";", "|").split("|") if x.strip()]

    out = pubs.copy()
    out["lab_ror_list"] = out[labs_col].apply(_split)
    out = out.explode("lab_ror_list", ignore_index=False)
    out = out.rename(columns={"lab_ror_list": "lab_ror"})
    return out


def find_copubs_between_labs(
    pubs: pd.DataFrame, lab_a: str, lab_b: str, labs_col: str = "Labs_RORs"
) -> pd.DataFrame:
    """Return publications co-authored by both lab RORs."""
    e = explode_labs(pubs, labs_col=labs_col)
    works_a = set(e.index[e["lab_ror"] == lab_a])
    works_b = set(e.index[e["lab_ror"] == lab_b])
    idx = list(works_a & works_b)
    return pubs.loc[idx].copy()


def lab_field_counts_from_core(
    pubs: pd.DataFrame,
    lab_ror: str,
    years: Iterable[int],
    field_col_candidates: Tuple[str, ...] = ("Primary Field ID", "Field ID"),
) -> pd.DataFrame:
    """
    Count publications per Field ID for a given lab over selected years.
    Uses first existing column among field_col_candidates.
    """
    df = filter_years(pubs, years=years)
    e = explode_labs(df)
    has_lab = set(e.index[e["lab_ror"] == lab_ror])
    subset = df.loc[list(has_lab)].copy()

    field_col = next((c for c in field_col_candidates if c in subset.columns), None)
    if not field_col:
        return pd.DataFrame(columns=["Field ID", "count"])

    out = subset.groupby(field_col).size().reset_index(name="count")
    return out.rename(columns={field_col: "Field ID"})
