# lib/transforms.py
from __future__ import annotations

from lib.constants import FIELDS_TO_DOMAIN


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
