# lib/transforms.py
from __future__ import annotations
from lib.constants import FIELDS_TO_DOMAIN, DOMAIN_NAMES

def clamp(x: int) -> int:
    return max(0, min(255, x))

def hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % rgb

def darken(hex_color: str, factor: float = 0.75) -> str:
    r, g, b = hex_to_rgb(hex_color)
    r, g, b = (clamp(int(r * factor)), clamp(int(g * factor)), clamp(int(b * factor)))
    return rgb_to_hex((r, g, b))

def map_field_to_domain(field: str) -> str:
    return FIELDS_TO_DOMAIN.get(field, "Other")

def field_order(all_fields: list[str]) -> list[str]:
    """Return fields ordered by DOMAIN_NAMES, then alphabetical inside each domain."""
    all_fields = list(dict.fromkeys(all_fields))  # unique, keep order
    # bucket by domain
    buckets = {d: [] for d in DOMAIN_NAMES}
    buckets["Other"] = []
    for f in all_fields:
        d = map_field_to_domain(f)
        if d not in buckets:
            buckets["Other"].append(f)
        else:
            buckets[d].append(f)
    ordered = []
    for d in DOMAIN_NAMES + ["Other"]:
        ordered.extend(sorted(buckets.get(d, [])))
    return ordered

def all_fields_order() -> list[str]:
    """
    Canonical order of ALL fields: domain buckets (via FIELDS_TO_DOMAIN) then A→Z inside each.
    """
    from lib.constants import FIELDS_TO_DOMAIN
    return field_order(list(FIELDS_TO_DOMAIN.keys()))

# lib/transforms.py (append)
import pandas as pd
from functools import lru_cache

@lru_cache(maxsize=1)
def _domain_palette():
    # keep your existing colors (by domain name)
    return {
        "Health Sciences": "#F85C32",
        "Life Sciences": "#0CA750",
        "Physical Sciences": "#8190FF",
        "Social Sciences": "#FFCB3A",
        "Other": "#7f7f7f",
    }

@lru_cache(maxsize=1)
def domain_order():
    # fixed order for UI
    return ["Health Sciences", "Life Sciences", "Physical Sciences", "Social Sciences", "Other"]

def _safe_num(x):
    try:
        return float(x)
    except Exception:
        return None

def canonical_field_order(all_topics: pd.DataFrame) -> list[str]:
    """
    Produce a fixed, repeatable order of fields grouped by domain for comparisons.
    """
    df = all_topics[["field_id","field_name","domain_id","domain_name"]].drop_duplicates()
    # Map unknowns to "Other"
    df["domain_name"] = df["domain_name"].fillna("Other")
    dom_order = {d:i for i,d in enumerate(domain_order())}
    df["dom_rank"] = df["domain_name"].map(lambda d: dom_order.get(d, dom_order["Other"]))
    out = (df.sort_values(["dom_rank","field_name"])
             .drop_duplicates("field_name")["field_name"]
             .tolist())
    return out

def build_taxonomy_lookups(all_topics: pd.DataFrame):
    """
    Return dicts to translate IDs <-> names and field->domain.
    """
    t = all_topics.drop_duplicates()
    fid2fname = dict(zip(t["field_id"], t["field_name"]))
    fid2dname = dict(zip(t["field_id"], t["domain_name"]))
    sid2sname = dict(zip(t["subfield_id"], t["subfield_name"]))
    did2dname = dict(zip(t["domain_id"], t["domain_name"]))
    return {
        "field_id_to_name": fid2fname,
        "field_id_to_domain": fid2dname,
        "subfield_id_to_name": sid2sname,
        "domain_id_to_name": did2dname,
        "palette": _domain_palette(),
        "field_order": canonical_field_order(all_topics),
    }

