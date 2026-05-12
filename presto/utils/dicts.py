"""Dictionary helpers."""

from typing import Any


def deep_update(data: dict[str, Any], overwrite: dict[str, Any]) -> dict[str, Any]:
    """Recursively update a mapping with values from overwrite."""
    updated = dict(data)
    for key, value in overwrite.items():
        if isinstance(updated.get(key), dict) and isinstance(value, dict):
            updated[key] = deep_update(updated[key], value)
        else:
            updated[key] = value
    return updated
