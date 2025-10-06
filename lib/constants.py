# constants.py
from __future__ import annotations
from typing import Final, Dict, List


DATA_DIR: Final = "data" # relative to Streamlit working dir
YEAR_RANGE: Final = list(range(2019, 2024)) # [2019..2023]


# Canonical domain ordering
DOMAIN_ORDER: Final[List[str]] = [
"Health Sciences",
"Life Sciences",
"Physical Sciences",
"Social Sciences",
"Other",
]


# Domain colors (cascade to children)
DOMAIN_COLORS: Final[Dict[str, str]] = {
"Health Sciences": "#F85C32",
"Life Sciences": "#0CA750",
"Physical Sciences": "#8190FF",
"Social Sciences": "#FFCB3A",
"Other": "#7f7f7f",
}


# Column keys (centralized to avoid typos)
COL = type("COL", (), {
# Core tables
"PUBS": "pubs_final.parquet",
"AUTHORS": "ul_authors_indicators.parquet",
"FIELDS": "ul_fields_indicators.parquet",
"DOMAINS": "ul_domains_indicators.parquet",
"PARTNERS": "ul_partners_indicators.parquet",
"UNITS": "ul_units_indicators.parquet",
"TOPICS": "all_topics.parquet",


# Common column names used across pages (abbreviated, map in transforms if needed)
"FIELD_ID": "Field ID",
"FIELD_NAME": "Field name",
"DOMAIN_ID": "Domain ID",
"DOMAIN_NAME": "Domain name",
"SUBFIELD_ID": "subfield_id",
"SUBFIELD_NAME": "subfield_name",
"YEAR": "Publication Year",
"TYPE": "Publication Type",
"FWCI_FR": "Avg FWCI (France)", # or dataset-specific key
})


# UI defaults
MIN_PARTNER_COPUBS: Final = 50
TOP_N_PARTNERS: Final = 20
TOP_N_AUTHORS: Final = 20
COUNT_LABEL_GUTTER_PX: Final = 80 # left gutter for count labels in bar charts