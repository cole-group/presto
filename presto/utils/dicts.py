"""Dictionary helpers."""

from typing import Any


def deep_update(data: dict[str, Any], overwrite: dict[str, Any]) -> dict[str, Any]:
    """Recursively update a mapping with values from overwrite.

    A list in ``data`` can be updated item by item with a dict keyed by list index,
    e.g. ``{"training_sampling_settings": {0: {"mlp_settings": {...}}}}``.
    """
    updated = dict(data)
    for key, value in overwrite.items():
        existing = updated.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            updated[key] = deep_update(existing, value)
        elif isinstance(existing, list) and isinstance(value, dict):
            updated[key] = _deep_update_list(key, existing, value)
        else:
            updated[key] = value
    return updated


def _deep_update_list(
    key: str, data: list[Any], overwrite: dict[Any, Any]
) -> list[Any]:
    """Update the items of a list from a dict keyed by list index.

    Negative indices are rejected so that each item has exactly one key, e.g. ``0``
    and ``-1`` cannot both address a one-item list with conflicting updates.
    """
    if not all(isinstance(index, int) for index in overwrite):
        raise ValueError(
            f"`{key}` is a list, so its overwrite must be keyed by list index, e.g. "
            f"{{{key!r}: {{0: {{...}}}}}}. Got keys {list(overwrite)}."
        )
    updated = list(data)
    for index, value in overwrite.items():
        if not 0 <= index < len(updated):
            raise IndexError(
                f"Overwrite index {index} is out of range for `{key}` with "
                f"{len(updated)} items. Indices must be non-negative."
            )
        if isinstance(updated[index], dict) and isinstance(value, dict):
            updated[index] = deep_update(updated[index], value)
        else:
            updated[index] = value
    return updated
