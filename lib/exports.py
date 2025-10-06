# lib/exports.py
from __future__ import annotations

import pandas as pd


def download_csv_button(df: pd.DataFrame, label: str, filename: str) -> None:
    """
    Render a Streamlit download button to export a DataFrame as CSV.
    UTF-8 with BOM to play nicer with Excel users.
    """
    import streamlit as st  # lazy import

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
    )
