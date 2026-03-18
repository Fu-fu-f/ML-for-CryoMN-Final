"""Shared helpers for formulation identity normalization and rendering."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


FEATURE_THRESHOLD = 1e-6
PERCENT_NEGLIGIBLE_THRESHOLD = 0.1
MOLAR_NEGLIGIBLE_THRESHOLD = 0.001


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
