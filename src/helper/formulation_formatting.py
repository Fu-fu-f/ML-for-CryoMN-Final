"""Shared helpers for rendering formulation rows as human-readable text."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


FEATURE_THRESHOLD = 1e-6


def format_formulation(row: pd.Series, feature_names: Sequence[str]) -> str:
    """Format a formulation row into a readable concentration string."""
    parts = []

    for feature_name in feature_names:
        value = row.get(feature_name, 0.0)
        if pd.isna(value) or float(value) <= FEATURE_THRESHOLD:
            continue

        if feature_name.endswith("_pct"):
            clean_name = feature_name.replace("_pct", "")
            parts.append(f"{float(value):.1f}% {clean_name}")
            continue

        clean_name = feature_name.replace("_M", "")
        concentration = float(value)
        if concentration >= 1.0:
            parts.append(f"{concentration:.2f}M {clean_name}")
        elif concentration >= 0.001:
            parts.append(f"{concentration * 1000:.1f}mM {clean_name}")
        else:
            parts.append(f"{concentration * 1e6:.1f}µM {clean_name}")

    return " + ".join(parts)
