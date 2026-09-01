"""Shared category ordering. Plain alphabetical sort mis-orders realistic
timepoint labels (e.g. "14dpi" before "7dpi", or "baseline" landing in the
middle instead of first) -- this gives every group_col a sensible default
order instead."""

from __future__ import annotations

import re

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# Non-numeric labels that should sort before any numbered timepoint, in
# this priority order (baseline always comes first).
_LEADING_LABELS = ["baseline", "pre", "control", "day0", "d0"]


def natural_sort_key(label):
    """Sort key for a single category label: non-numeric "leading" labels
    (baseline, pre, control, ...) first, then anything with an embedded
    number in numeric order (so "7dpi" < "14dpi"), then everything else
    alphabetically. Works for plain categorical labels too -- a group_col
    with no numbers in its values just falls back to alphabetical."""
    text = str(label)
    lowered = text.lower()

    if lowered in _LEADING_LABELS:
        return (0, _LEADING_LABELS.index(lowered), 0.0, text)

    match = _NUMBER_RE.search(text)
    if match is not None:
        return (1, 0, float(match.group()), text)

    return (2, 0, 0.0, text)


def natural_sort(values):
    """Sort an iterable of category labels with natural_sort_key."""
    return sorted(values, key=natural_sort_key)
