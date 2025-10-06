# lib/taxonomy.py
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

import pandas as pd

from .io import load_parquet

# Fixed domain display order (filtered to those present in data)
_DOMAIN_ORDER_CANON = [
    "Health Sciences",
    "Life Sciences",
    "Physical Sciences",
    "Social Sciences",
    "Other",
]

# Domain palette (cascades to fields/subfields)
_DOMAIN_COLORS = {
    "Health Sciences": "#F85C32",
    "Life Sciences": "#0CA750",
    "Physical Sciences": "#8190FF",
    "Social Sciences": "#FFCB3A",
    "Other": "#7f7f7f",
}


@lru_cache(maxsize=1)
def _topics() -> pd.DataFrame:
    df = load_parquet("all_topics").copy()
    # Normalize expected columns
    cols = {
        "domain_id": "domain_id",
        "domain_name": "domain_name",
        "field_id": "field_id",
        "field_name": "field_name",
        "subfield_id": "subfield_id",
        "subfield_name": "subfield_name",
        "topic_id": "topic_id",
        "topic_name": "topic_name",
    }
    df = df.rename(columns=cols)
    return df[list(cols.values()) + (["keywords"] if "keywords" in df.columns else [])]


@lru_cache(maxsize=1)
def build_taxonomy_lookups() -> Dict:
    """
    Build hierarchical lookups and canonical ordering for domains/fields/subfields.
    Returns dict with keys:
      - fields_by_domain: {domain_name: [field_name,...]}
      - subfields_by_field: {field_name: [subfield_name,...]}
      - canonical_fields: [field_name,...] (domain-grouped, alphabetic within domain)
      - canonical_subfields: [subfield_name,...] (grouped by field, alphabetic)
      - id2name: {str(id): name} for domain/field/subfield/topic
      - name2id: {name: str(id)} inverse mapping
      - domain_order: [domain_name,...] (filtered canonical)
    """
    t = _topics()

    # Domain order: use canonical filtered to what's present; append any extras by domain_id
    present_domains = (
        t[["domain_id", "domain_name"]]
        .drop_duplicates()
        .sort_values("domain_id")
        .reset_index(drop=True)
    )
    domain_order = [d for d in _DOMAIN_ORDER_CANON if d in present_domains["domain_name"].tolist()]
    extras = [d for d in present_domains["domain_name"].tolist() if d not in domain_order]
    domain_order += extras  # keep any extras at the end

    fields_by_domain: Dict[str, List[str]] = {}
    for d in domain_order:
        fields = (
            t.loc[t["domain_name"] == d, "field_name"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        fields_by_domain[d] = fields

    subfields_by_field: Dict[str, List[str]] = {}
    for f in t["field_name"].drop_duplicates().tolist():
        sub = t.loc[t["field_name"] == f, "subfield_name"].drop_duplicates().sort_values().tolist()
        subfields_by_field[f] = sub

    canonical_fields = [f for d in domain_order for f in fields_by_domain[d]]
    canonical_subfields: List[str] = []
    for f in canonical_fields:
        canonical_subfields.extend(subfields_by_field.get(f, []))

    id2name: Dict[str, str] = {}
    name2id: Dict[str, str] = {}
    for _, r in t.iterrows():
        for key_id, key_name in [
            ("domain_id", "domain_name"),
            ("field_id", "field_name"),
            ("subfield_id", "subfield_name"),
            ("topic_id", "topic_name"),
        ]:
            id2name[str(r[key_id])] = r[key_name]
            name2id[r[key_name]] = str(r[key_id])

    return {
        "fields_by_domain": fields_by_domain,
        "subfields_by_field": subfields_by_field,
        "canonical_fields": canonical_fields,
        "canonical_subfields": canonical_subfields,
        "id2name": id2name,
        "name2id": name2id,
        "domain_order": domain_order,
    }


@lru_cache(maxsize=None)
def get_domain_color(name_or_id: str) -> str:
    """Return hex color for a domain (by name or id)."""
    look = build_taxonomy_lookups()
    name = name_or_id
    if str(name_or_id).isdigit():  # id -> name
        name = look["id2name"].get(str(name_or_id), name_or_id)
    return _DOMAIN_COLORS.get(name, _DOMAIN_COLORS["Other"])


@lru_cache(maxsize=1)
def canonical_field_order() -> List[str]:
    return build_taxonomy_lookups()["canonical_fields"]


@lru_cache(maxsize=1)
def canonical_subfield_order() -> List[str]:
    return build_taxonomy_lookups()["canonical_subfields"]
