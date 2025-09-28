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
    """Canonical order of ALL fields (domain ordering, then A→Z inside each)."""
    return field_order(list(FIELDS_TO_DOMAIN.keys()))