#!/usr/bin/env python3
"""
Shared helpers for iteration-aware observed-context data.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from iteration_metadata import (
    PRIOR_MEAN_METHOD,
    STANDARD_METHOD,
    WEIGHTED_SIMPLE_METHOD,
    normalize_model_method,
)


OBSERVED_CONTEXT_FILENAME = 'observed_context.csv'
LEGACY_EVALUATION_FILENAME = 'evaluation_data.csv'
SOURCE_LITERATURE = 'literature'
SOURCE_WETLAB = 'wetlab'
CONTEXT_META_COLUMNS = [
    'viability_percent',
    'source',
    'context_weight',
    'model_method',
    'iteration',
    'iteration_dir',
]


def observed_context_iteration_path(project_root: str, iteration_dir: Optional[str]) -> Optional[str]:
    """Return the iteration-specific observed-context path."""
    if not iteration_dir:
        return None
    return os.path.join(project_root, 'models', iteration_dir, OBSERVED_CONTEXT_FILENAME)


def observed_context_active_path(project_root: str) -> str:
    """Return the active observed-context mirror path."""
    return os.path.join(project_root, 'models', OBSERVED_CONTEXT_FILENAME)


def legacy_evaluation_data_path(project_root: str) -> str:
    """Return the legacy compatibility path used by prior-mean iterations."""
    return os.path.join(project_root, 'data', 'processed', LEGACY_EVALUATION_FILENAME)


def _ordered_columns(feature_names: List[str]) -> List[str]:
    return list(feature_names) + CONTEXT_META_COLUMNS


def _empty_context_df(feature_names: List[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=_ordered_columns(feature_names))


def infer_wetlab_context_weight(
    model_method: Optional[str],
    metadata: Optional[Dict] = None,
) -> float:
    """Infer the intended wet-lab context weight for one iteration."""
    metadata = metadata or {}
    normalized = normalize_model_method(
        model_method or metadata.get('model_method') or metadata.get('weighting_method'),
        metadata.get('is_composite_model'),
    )
    if normalized == WEIGHTED_SIMPLE_METHOD:
        return float(metadata.get('weight_multiplier', 1.0))
    if normalized == PRIOR_MEAN_METHOD:
        if 'noise_ratio' in metadata:
            return float(metadata['noise_ratio'])
        alpha_lit = float(metadata.get('alpha_literature', 1.0))
        alpha_wet = float(metadata.get('alpha_wetlab', 1.0))
        return alpha_lit / alpha_wet if alpha_wet else 1.0
    return 1.0


def _build_source_df(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    source: str,
    context_weight: float,
    model_method: str,
    iteration: Optional[int],
    iteration_dir: Optional[str],
) -> pd.DataFrame:
    """Build one source-specific observed-context frame."""
    if X is None or len(X) == 0:
        return _empty_context_df(feature_names)

    df = pd.DataFrame(X, columns=feature_names)
    df['viability_percent'] = np.asarray(y, dtype=float)
    df['source'] = source
    df['context_weight'] = float(context_weight)
    df['model_method'] = model_method
    df['iteration'] = iteration
    df['iteration_dir'] = iteration_dir
    return df[_ordered_columns(feature_names)]


def build_observed_context_df(
    feature_names: List[str],
    X_literature: np.ndarray,
    y_literature: np.ndarray,
    X_wetlab: Optional[np.ndarray],
    y_wetlab: Optional[np.ndarray],
    model_method: str,
    iteration: Optional[int],
    iteration_dir: Optional[str],
    wetlab_context_weight: float,
) -> pd.DataFrame:
    """Build the canonical observed-context dataframe for one iteration."""
    normalized_method = normalize_model_method(model_method)
    frames = [
        _build_source_df(
            X_literature,
            y_literature,
            feature_names,
            SOURCE_LITERATURE,
            1.0,
            normalized_method,
            iteration,
            iteration_dir,
        )
    ]
    if X_wetlab is not None and len(X_wetlab):
        frames.append(
            _build_source_df(
                X_wetlab,
                y_wetlab,
                feature_names,
                SOURCE_WETLAB,
                wetlab_context_weight,
                normalized_method,
                iteration,
                iteration_dir,
            )
        )
    return pd.concat(frames, ignore_index=True)


def save_observed_context(output_dir: str, observed_df: pd.DataFrame):
    """Write the canonical observed-context artifact for one iteration."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, OBSERVED_CONTEXT_FILENAME)
    observed_df.to_csv(output_path, index=False)
    print(f"Observed context saved: {output_path}")


def save_legacy_evaluation_data(project_root: str, observed_df: pd.DataFrame, feature_names: List[str]):
    """Write the legacy prior-mean compatibility mirror."""
    legacy_df = observed_df[list(feature_names) + ['viability_percent', 'source']].copy()
    legacy_df['weight'] = observed_df['context_weight'].values
    output_path = legacy_evaluation_data_path(project_root)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    legacy_df.to_csv(output_path, index=False)
    print(f"Legacy evaluation data saved: {output_path}")


def load_literature_rows(project_root: str, feature_names: List[str]) -> pd.DataFrame:
    """Load literature rows aligned to the active feature set."""
    data_path = os.path.join(project_root, 'data', 'processed', 'parsed_formulations.csv')
    df = pd.read_csv(data_path)
    df = df[df['viability_percent'] <= 100].copy()
    literature = df.reindex(columns=feature_names, fill_value=0.0).fillna(0.0)
    literature['viability_percent'] = df['viability_percent'].values
    literature['source'] = SOURCE_LITERATURE
    return literature


def load_wetlab_rows(project_root: str, feature_names: List[str]) -> pd.DataFrame:
    """Load measured wet-lab rows aligned to the active feature set."""
    validation_path = os.path.join(project_root, 'data', 'validation', 'validation_results.csv')
    if not os.path.exists(validation_path):
        return pd.DataFrame(columns=list(feature_names) + ['viability_percent', 'source'])

    df = pd.read_csv(validation_path)
    df = df[df['viability_measured'].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=list(feature_names) + ['viability_percent', 'source'])

    wetlab = df.reindex(columns=feature_names, fill_value=0.0).fillna(0.0)
    wetlab['viability_percent'] = df['viability_measured'].astype(float).values
    wetlab['source'] = SOURCE_WETLAB
    return wetlab


def reconstruct_observed_context(
    project_root: str,
    feature_names: List[str],
    model_method: str,
    iteration: Optional[int],
    iteration_dir: Optional[str],
    metadata: Optional[Dict] = None,
) -> pd.DataFrame:
    """Reconstruct observed context from canonical literature and validation inputs."""
    metadata = metadata or {}
    literature = load_literature_rows(project_root, feature_names)
    wetlab = load_wetlab_rows(project_root, feature_names)
    wetlab_weight = infer_wetlab_context_weight(model_method, metadata)
    observed_df = build_observed_context_df(
        feature_names=feature_names,
        X_literature=literature[feature_names].values,
        y_literature=literature['viability_percent'].values,
        X_wetlab=wetlab[feature_names].values if len(wetlab) else None,
        y_wetlab=wetlab['viability_percent'].values if len(wetlab) else None,
        model_method=model_method,
        iteration=iteration,
        iteration_dir=iteration_dir,
        wetlab_context_weight=wetlab_weight,
    )
    print(
        f"  >>> Reconstructed observed context ({len(observed_df)} rows; "
        f"{len(literature)} literature + {len(wetlab)} wet lab)"
    )
    return observed_df


def _normalize_observed_context_df(
    df: pd.DataFrame,
    feature_names: List[str],
    model_method: str,
    iteration: Optional[int],
    iteration_dir: Optional[str],
) -> pd.DataFrame:
    """Normalize a context-like dataframe to the canonical schema."""
    normalized = df.copy()
    normalized_method = normalize_model_method(model_method)

    if 'viability_percent' not in normalized.columns and 'viability_measured' in normalized.columns:
        normalized['viability_percent'] = normalized['viability_measured']
    if 'context_weight' not in normalized.columns:
        if 'weight' in normalized.columns:
            normalized['context_weight'] = normalized['weight']
        else:
            normalized['context_weight'] = 1.0
    if 'source' not in normalized.columns:
        normalized['source'] = SOURCE_LITERATURE
    if 'model_method' not in normalized.columns:
        normalized['model_method'] = normalized_method
    else:
        normalized['model_method'] = normalized['model_method'].fillna(normalized_method)
    if 'iteration' not in normalized.columns:
        normalized['iteration'] = iteration
    else:
        normalized['iteration'] = normalized['iteration'].fillna(iteration)
    if 'iteration_dir' not in normalized.columns:
        normalized['iteration_dir'] = iteration_dir
    else:
        normalized['iteration_dir'] = normalized['iteration_dir'].fillna(iteration_dir)

    for name in feature_names:
        if name not in normalized.columns:
            normalized[name] = 0.0

    normalized = normalized[normalized['viability_percent'].notna()].copy()
    normalized['viability_percent'] = normalized['viability_percent'].astype(float)
    normalized['context_weight'] = normalized['context_weight'].astype(float)
    normalized[feature_names] = normalized[feature_names].fillna(0.0).astype(float)
    return normalized[_ordered_columns(feature_names)]


def _context_matches_resolution(
    df: pd.DataFrame,
    iteration: Optional[int],
    iteration_dir: Optional[str],
) -> bool:
    """Return True when a loaded artifact matches the resolved active iteration."""
    if iteration is not None and 'iteration' in df.columns:
        actual_iterations = set(df['iteration'].dropna().astype(int).tolist())
        if actual_iterations and actual_iterations != {int(iteration)}:
            return False
    if iteration_dir and 'iteration_dir' in df.columns:
        actual_dirs = set(df['iteration_dir'].dropna().astype(str).tolist())
        if actual_dirs and actual_dirs != {str(iteration_dir)}:
            return False
    return True


def _load_context_file(
    path: str,
    feature_names: List[str],
    model_method: str,
    iteration: Optional[int],
    iteration_dir: Optional[str],
    description: str,
) -> Optional[pd.DataFrame]:
    """Try to load and normalize one observed-context candidate file."""
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    normalized = _normalize_observed_context_df(
        df, feature_names, model_method, iteration, iteration_dir
    )
    if not _context_matches_resolution(normalized, iteration, iteration_dir):
        print(f"  >>> Ignoring {description}: iteration metadata does not match the active model")
        return None
    print(f"  >>> Loaded observed context from {description} ({len(normalized)} rows)")
    return normalized


def load_observed_context(
    project_root: str,
    feature_names: List[str],
    model_method: str,
    iteration: Optional[int],
    iteration_dir: Optional[str],
    metadata: Optional[Dict] = None,
) -> pd.DataFrame:
    """Load observed context via canonical artifact, compatibility mirror, or reconstruction."""
    metadata = metadata or {}

    observed_df = _load_context_file(
        observed_context_iteration_path(project_root, iteration_dir),
        feature_names,
        model_method,
        iteration,
        iteration_dir,
        'iteration artifact',
    )
    if observed_df is not None:
        return observed_df

    observed_df = _load_context_file(
        observed_context_active_path(project_root),
        feature_names,
        model_method,
        iteration,
        iteration_dir,
        'active mirror',
    )
    if observed_df is not None:
        return observed_df

    if normalize_model_method(model_method, metadata.get('is_composite_model')) == PRIOR_MEAN_METHOD:
        observed_df = _load_context_file(
            legacy_evaluation_data_path(project_root),
            feature_names,
            model_method,
            iteration,
            iteration_dir,
            'legacy evaluation_data.csv',
        )
        if observed_df is not None:
            return observed_df

    return reconstruct_observed_context(
        project_root,
        feature_names,
        model_method,
        iteration,
        iteration_dir,
        metadata,
    )


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Compute a weighted quantile over one-dimensional numeric values."""
    if len(values) == 0:
        return float('nan')

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    if not np.any(weights > 0):
        return float(np.quantile(values, quantile))

    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    threshold = quantile * cumulative[-1]
    index = int(np.searchsorted(cumulative, threshold, side='left'))
    index = min(index, len(sorted_values) - 1)
    return float(sorted_values[index])


def collapse_observed_context_for_bo(
    observed_df: pd.DataFrame,
    feature_names: List[str],
) -> pd.DataFrame:
    """Collapse repeated feature rows into unique BO geometry rows with analytic weights."""
    if observed_df.empty:
        return pd.DataFrame(
            columns=list(feature_names) + ['viability_percent', 'context_weight', 'source', 'n_context_rows']
        )

    subset = observed_df[list(feature_names) + ['viability_percent', 'context_weight', 'source']]
    rows = []
    for keys, group in subset.groupby(feature_names, sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        weights = group['context_weight'].to_numpy(dtype=float)
        viability = group['viability_percent'].to_numpy(dtype=float)
        weight_sum = float(np.sum(weights))
        weighted_viability = float(
            np.average(viability, weights=weights) if weight_sum > 0 else np.mean(viability)
        )
        row = dict(zip(feature_names, key_values))
        row.update({
            'viability_percent': weighted_viability,
            'context_weight': weight_sum,
            'source': '+'.join(sorted(group['source'].astype(str).unique())),
            'n_context_rows': len(group),
        })
        rows.append(row)
    return pd.DataFrame(rows)
