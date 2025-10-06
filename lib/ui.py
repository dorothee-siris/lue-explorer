# lib/ui.py
from __future__ import annotations

from typing import Iterable, List, Tuple


def year_multiselect(
    key: str = "years",
    default: Iterable[int] = (2019, 2020, 2021, 2022, 2023),
) -> List[int]:
    import streamlit as st
    return st.multiselect("Years", list(default), default=list(default), key=key)


def two_col_compare(label_left: str = "Left", label_right: str = "Right"):
    """
    Return two Streamlit containers side by side, with subheaders.
    Usage:
        c1, c2 = two_col_compare("Lab A", "Lab B")
        with c1: ...
        with c2: ...
    """
    import streamlit as st
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(label_left)
    with c2:
        st.subheader(label_right)
    return c1, c2
