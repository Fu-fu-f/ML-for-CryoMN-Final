"""Shared helpers for formulation identity normalization and rendering."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


FEATURE_THRESHOLD = 1e-6
PERCENT_NEGLIGIBLE_THRESHOLD = 0.1
MOLAR_NEGLIGIBLE_THRESHOLD = 0.001
EXPLICIT_PERCENTAGE_CAP = 100.0
EXPLICIT_PERCENTAGE_CAP_TOLERANCE = 1e-6


def negligible_threshold_for_feature(feature_name: str) -> float:
    """Return the practical presence floor for one formulation feature."""
    if feature_name.endswith("_pct"):
        return PERCENT_NEGLIGIBLE_THRESHOLD
    if feature_name.endswith("_M"):
        return MOLAR_NEGLIGIBLE_THRESHOLD
    return FEATURE_THRESHOLD


def is_negligible_feature_value(feature_name: str, value: object) -> bool:
    """Return True when a feature value should be treated as absent."""
    if value is None or pd.isna(value):
        return True
    return abs(float(value)) < negligible_threshold_for_feature(feature_name)


def normalize_formulation_row(row: pd.Series, feature_names: Sequence[str]) -> pd.Series:
    """Return a copy of one formulation row with negligible features zeroed."""
    normalized = row.copy()
    for feature_name in feature_names:
        value = row.get(feature_name, 0.0)
        normalized[feature_name] = 0.0 if is_negligible_feature_value(feature_name, value) else float(value)
    return normalized


def normalize_formulation_dataframe(df: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    """Return a copy of a formulation DataFrame with negligible features zeroed."""
    normalized = df.copy()
    for feature_name in feature_names:
        if feature_name not in normalized.columns:
            continue
        values = pd.to_numeric(normalized[feature_name], errors="coerce").fillna(0.0)
        floor = negligible_threshold_for_feature(feature_name)
        normalized[feature_name] = values.where(np.abs(values) >= floor, 0.0)
    return normalized


def normalize_formulation_vector(vector: Sequence[float], feature_names: Sequence[str]) -> np.ndarray:
    """Return one formulation vector with negligible features zeroed."""
    arr = np.asarray(vector, dtype=float).copy()
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D formulation vector, got shape {arr.shape}")
    if len(arr) != len(feature_names):
        raise ValueError(
            f"Vector length {len(arr)} does not match feature count {len(feature_names)}."
        )
    for idx, feature_name in enumerate(feature_names):
        if is_negligible_feature_value(feature_name, arr[idx]):
            arr[idx] = 0.0
    return arr


def normalize_formulation_matrix(matrix: Sequence[Sequence[float]], feature_names: Sequence[str]) -> np.ndarray:
    """Return a 2D formulation matrix with negligible features zeroed."""
    arr = np.asarray(matrix, dtype=float).copy()
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D formulation matrix, got shape {arr.shape}")
    if arr.shape[1] != len(feature_names):
        raise ValueError(
            f"Matrix feature dimension {arr.shape[1]} does not match feature count {len(feature_names)}."
        )
    for idx, feature_name in enumerate(feature_names):
        floor = negligible_threshold_for_feature(feature_name)
        arr[:, idx] = np.where(np.abs(arr[:, idx]) < floor, 0.0, arr[:, idx])
    return arr


def explicit_percentage_feature_names(feature_names: Sequence[str]) -> list[str]:
    """Return the explicit percentage-valued formulation features."""
    return [
        feature_name
        for feature_name in feature_names
        if feature_name.endswith("_pct") and feature_name != "dmso_percent"
    ]


def explicit_percentage_total_from_mapping(
    values: object,
    feature_names: Sequence[str],
) -> float:
    """Return the total explicit percentage contribution for one row-like object."""
    total = 0.0
    for feature_name in explicit_percentage_feature_names(feature_names):
        value = values.get(feature_name, 0.0) if hasattr(values, "get") else 0.0
        if is_negligible_feature_value(feature_name, value):
            continue
        total += float(value)
    return total


def explicit_percentage_totals_from_matrix(
    matrix: Sequence[Sequence[float]],
    feature_names: Sequence[str],
) -> np.ndarray:
    """Return explicit percentage totals for each normalized formulation row."""
    arr = normalize_formulation_matrix(matrix, feature_names)
    pct_indices = [
        idx
        for idx, feature_name in enumerate(feature_names)
        if feature_name.endswith("_pct") and feature_name != "dmso_percent"
    ]
    if not pct_indices:
        return np.zeros(arr.shape[0], dtype=float)
    return np.sum(arr[:, pct_indices], axis=1)


def explicit_percentage_cap_excess_from_matrix(
    matrix: Sequence[Sequence[float]],
    feature_names: Sequence[str],
    cap: float = EXPLICIT_PERCENTAGE_CAP,
    tolerance: float = EXPLICIT_PERCENTAGE_CAP_TOLERANCE,
) -> np.ndarray:
    """Return the amount by which each row exceeds the explicit percentage cap."""
    totals = explicit_percentage_totals_from_matrix(matrix, feature_names)
    return np.maximum(0.0, totals - cap - tolerance)


def exceeds_explicit_percentage_cap_vector(
    vector: Sequence[float],
    feature_names: Sequence[str],
    cap: float = EXPLICIT_PERCENTAGE_CAP,
    tolerance: float = EXPLICIT_PERCENTAGE_CAP_TOLERANCE,
) -> bool:
    """Return True when one formulation vector exceeds the explicit percentage cap."""
    arr = normalize_formulation_vector(vector, feature_names)
    pct_indices = [
        idx
        for idx, feature_name in enumerate(feature_names)
        if feature_name.endswith("_pct") and feature_name != "dmso_percent"
    ]
    if not pct_indices:
        return False
    return float(np.sum(arr[pct_indices])) > cap + tolerance


def exceeds_explicit_percentage_cap_mapping(
    values: object,
    feature_names: Sequence[str],
    cap: float = EXPLICIT_PERCENTAGE_CAP,
    tolerance: float = EXPLICIT_PERCENTAGE_CAP_TOLERANCE,
) -> bool:
    """Return True when one row-like object exceeds the explicit percentage cap."""
    total = explicit_percentage_total_from_mapping(values, feature_names)
    return total > cap + tolerance


def format_formulation(row: pd.Series, feature_names: Sequence[str]) -> str:
    """Format a formulation row into a readable concentration string."""
    normalized = normalize_formulation_row(row, feature_names)
    parts = []

    for feature_name in feature_names:
        value = normalized.get(feature_name, 0.0)
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
