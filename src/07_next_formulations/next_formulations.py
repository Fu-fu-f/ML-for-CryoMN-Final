#!/usr/bin/env python3
"""
Generate the next wet-lab formulations with a fixed 10/10 exploit-explore split.

The script is intentionally strict:
- validate required inputs up front
- fail before writing if anything is missing or inconsistent
- generate exploration probes from residual blind spots rather than only from
  saved candidate CSVs
- validate the final 20 formulations before writing any outputs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NEXT_FORMULATIONS_DIR = RESULTS_DIR / "next_formulations"
VALIDATION_PATH = PROJECT_ROOT / "data" / "validation" / "validation_results.csv"
VALIDATION_LOOP_DIR = PROJECT_ROOT / "src" / "04_validation_loop"
BO_DIR = PROJECT_ROOT / "src" / "05_bo_optimization"
HELPER_DIR = PROJECT_ROOT / "src" / "helper"

for path in [PROJECT_ROOT, VALIDATION_LOOP_DIR, BO_DIR, HELPER_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from formulation_formatting import (  # noqa: E402
    EXPLICIT_PERCENTAGE_CAP,
    exceeds_explicit_percentage_cap_mapping,
    explicit_percentage_total_from_mapping,
    format_formulation,
    normalize_formulation_dataframe,
    normalize_formulation_row,
    normalize_formulation_vector,
)
from update_model_weighted_prior import CompositeGP  # noqa: F401,E402
from bo_optimizer import BOConfig, BayesianOptimizer  # noqa: E402


EXPERIMENT_ID_PATTERN = re.compile(r"(\d+)")
ITERATION_DIR_PATTERN = re.compile(r"^iteration_(\d+)(?:_[A-Za-z0-9_]+)?$")
FEATURE_THRESHOLD = 1e-6
GENERATION_SEED = 42
EXPLOIT_COUNT = 8
EXPLORE_COUNT = 12
TOTAL_COUNT = EXPLOIT_COUNT + EXPLORE_COUNT
LOCAL_RANK_PROBE_COUNT = 8
BLINDSPOT_PROBE_COUNT = 4
LOCAL_RANK_ANCHOR_COUNT = 4
LOCAL_RANK_PRIMARY_DELTA = 0.10
LOCAL_RANK_RETRY_DELTA = 0.20
BATCH_RECOMMENDATION_MIN = 6
BATCH_RECOMMENDATION_MAX = 12
POSITIVE_RESIDUAL_THRESHOLDS = [10.0, 8.0, 5.0, 2.0, 0.0]
MEANINGFUL_ACTUAL_MIN = 50.0
MIN_EXPLORATION_PREDICTION = 30.0
EXPLOIT_FAMILY_LIMIT = 3
EXPLORE_FAMILY_LIMIT = 2
BLINDSPOT_SIGNAL_MODE = "historical_residual_hybrid"
BLINDSPOT_RECENCY_DECAY = 0.7
BLINDSPOT_REGION_PREDICTION_THRESHOLD = 50.0
BLINDSPOT_REGION_WEIGHT = 1.5
BLINDSPOT_SUPPORT_EXPERIMENT_MIN = 2
BLINDSPOT_EXPERIMENT_SHRINK_OFFSET = 2.0
BLINDSPOT_BATCH_SHRINK_OFFSET = 1.0
REQUIRED_VALIDATION_COLUMNS = {"experiment_id", "experiment_date", "viability_measured"}
REQUIRED_CANDIDATE_COLUMNS = {
    "rank",
    "predicted_viability",
    "uncertainty",
    "dmso_percent",
    "n_ingredients",
}


class ValidationError(RuntimeError):
    """Raised when required inputs or outputs are invalid."""


@dataclass
class StageArtifacts:
    stage: int
    iteration_dir: str
    metadata: Dict
    feature_names: List[str]
    is_composite_model: bool
    model: object
    scaler: Optional[object]
    observed_context: pd.DataFrame


def round_or_none(value: float, digits: int = 4) -> Optional[float]:
    """Return a JSON-safe rounded number."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def normalize(values: pd.Series) -> pd.Series:
    """Min-max normalize a numeric series."""
    if values.empty:
        return values
    minimum = float(values.min())
    maximum = float(values.max())
    if math.isclose(minimum, maximum):
        return pd.Series(np.full(len(values), 0.5), index=values.index)
    return (values - minimum) / (maximum - minimum)


def parse_experiment_stage(experiment_id: str) -> Optional[int]:
    """Map EXP identifiers to the validation stage they belong to."""
    match = EXPERIMENT_ID_PATTERN.search(str(experiment_id))
    if not match:
        return None
    numeric = int(match.group(1))
    if numeric < 1000:
        return 0
    return numeric // 1000


def is_feature_column(name: str) -> bool:
    """Return True when a column is one model feature."""
    return name.endswith("_M") or name.endswith("_pct")


def feature_columns_from_df(df: pd.DataFrame) -> List[str]:
    """Extract formulation feature columns from a dataframe."""
    return [name for name in df.columns if is_feature_column(name)]


def format_stage_label(stage: int, iteration_dir: str) -> str:
    """Render a short stage label."""
    if stage == 0:
        return "literature_only"
    return iteration_dir


def active_features(row: pd.Series, feature_names: Sequence[str]) -> List[str]:
    """Return active features for one formulation row."""
    normalized = normalize_formulation_row(row, feature_names)
    names: List[str] = []
    for feature_name in feature_names:
        value = normalized.get(feature_name, 0.0)
        if pd.isna(value):
            continue
        if abs(float(value)) > FEATURE_THRESHOLD:
            names.append(feature_name)
    return names


def top_features_by_magnitude(row: pd.Series, feature_names: Sequence[str], limit: int = 2) -> List[str]:
    """Return the largest active features in descending magnitude."""
    normalized = normalize_formulation_row(row, feature_names)
    ranked: List[Tuple[float, str]] = []
    for feature_name in feature_names:
        value = normalized.get(feature_name, 0.0)
        if pd.isna(value):
            continue
        value = float(value)
        if abs(value) <= FEATURE_THRESHOLD:
            continue
        ranked.append((abs(value), feature_name))
    ranked.sort(reverse=True)
    return [name for _, name in ranked[:limit]]


def chemistry_family(row: pd.Series, feature_names: Sequence[str]) -> str:
    """Coarse family label used to keep the final output diverse."""
    top_names = top_features_by_magnitude(row, feature_names, limit=2)
    if not top_names:
        return "empty"
    cleaned = [name.replace("_M", "").replace("_pct", "") for name in top_names]
    return "+".join(sorted(cleaned))


def predict(model, scaler, X: np.ndarray, is_composite_model: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Predict mean and uncertainty for an unscaled feature matrix."""
    if is_composite_model:
        return model.predict(X, return_std=True)
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled, return_std=True)


def discover_iteration_dirs(stage: int) -> List[str]:
    """List saved iteration folders for one stage."""
    matches: List[Tuple[str, str]] = []
    prefix = f"iteration_{stage}"
    if not MODELS_DIR.exists():
        return []
    for entry in sorted(os.listdir(MODELS_DIR)):
        if not entry.startswith(prefix):
            continue
        metadata_path = MODELS_DIR / entry / "model_metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        sort_key = str(metadata.get("updated_at") or metadata.get("trained_at") or entry)
        matches.append((entry, sort_key))
    matches.sort(key=lambda item: item[1])
    return [entry for entry, _ in matches]


def choose_iteration_dir(stage: int, preferred_iteration_dir: Optional[str] = None) -> str:
    """Resolve the saved iteration directory for one stage."""
    if stage == 0:
        if not (MODELS_DIR / "literature_only" / "model_metadata.json").exists():
            raise ValidationError("Missing literature-only checkpoint at models/literature_only/")
        return "literature_only"

    if preferred_iteration_dir:
        preferred_path = MODELS_DIR / preferred_iteration_dir / "model_metadata.json"
        if preferred_path.exists():
            return preferred_iteration_dir

    matches = discover_iteration_dirs(stage)
    if not matches:
        raise ValidationError(f"No saved model directory found for iteration {stage}.")
    return matches[-1]


def load_stage_artifacts(
    stage: int,
    preferred_iteration_dir: Optional[str] = None,
    require_observed_context: bool = True,
) -> StageArtifacts:
    """Load one saved stage checkpoint plus its observed context."""
    iteration_dir = choose_iteration_dir(stage, preferred_iteration_dir)
    model_dir = MODELS_DIR / iteration_dir
    metadata_path = model_dir / "model_metadata.json"
    if not metadata_path.exists():
        raise ValidationError(f"Missing metadata file: {metadata_path}")

    metadata = json.loads(metadata_path.read_text())
    feature_names = list(metadata.get("feature_names") or [])
    if not feature_names:
        raise ValidationError(f"{metadata_path} does not define feature_names.")

    is_composite_model = bool(metadata.get("is_composite_model", False))
    if is_composite_model:
        model_path = model_dir / "composite_model.pkl"
        if not model_path.exists():
            raise ValidationError(f"Missing composite model artifact: {model_path}")
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
        scaler = None
    else:
        model_path = model_dir / "gp_model.pkl"
        scaler_path = model_dir / "scaler.pkl"
        if not model_path.exists():
            raise ValidationError(f"Missing GP model artifact: {model_path}")
        if not scaler_path.exists():
            raise ValidationError(f"Missing scaler artifact: {scaler_path}")
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
        with scaler_path.open("rb") as handle:
            scaler = pickle.load(handle)

    observed_context_path = model_dir / "observed_context.csv"
    if observed_context_path.exists():
        observed_context = pd.read_csv(observed_context_path)
        missing_observed = [name for name in feature_names if name not in observed_context.columns]
        if missing_observed:
            raise ValidationError(
                f"{observed_context_path} is missing feature columns: {', '.join(missing_observed)}"
            )
        if "viability_percent" not in observed_context.columns:
            raise ValidationError(f"{observed_context_path} is missing viability_percent.")
    elif require_observed_context:
        raise ValidationError(f"Missing observed context artifact: {observed_context_path}")
    else:
        observed_context = pd.DataFrame(columns=[*feature_names, "viability_percent"])

    return StageArtifacts(
        stage=stage,
        iteration_dir=iteration_dir,
        metadata=metadata,
        feature_names=feature_names,
        is_composite_model=is_composite_model,
        model=model,
        scaler=scaler,
        observed_context=observed_context,
    )


def align_candidate_df(df: pd.DataFrame, feature_names: Sequence[str], path: Path) -> pd.DataFrame:
    """Align a BO candidate file to the active feature space."""
    missing_required = sorted(REQUIRED_CANDIDATE_COLUMNS - set(df.columns))
    if missing_required:
        raise ValidationError(f"{path} is missing required columns: {', '.join(missing_required)}")

    candidate_features = feature_columns_from_df(df)
    unknown_features = sorted(name for name in candidate_features if name not in feature_names)
    if unknown_features:
        raise ValidationError(
            f"{path} contains feature columns not present in the active model: {', '.join(unknown_features)}"
        )

    aligned = df.copy()
    for feature_name in feature_names:
        if feature_name not in aligned.columns:
            aligned[feature_name] = 0.0

    for column in feature_names:
        aligned[column] = aligned[column].fillna(0.0).astype(float)
    aligned["rank"] = aligned["rank"].astype(int)
    aligned["predicted_viability"] = aligned["predicted_viability"].astype(float)
    aligned["uncertainty"] = aligned["uncertainty"].astype(float)
    aligned["dmso_percent"] = aligned["dmso_percent"].fillna(0.0).astype(float)
    aligned["n_ingredients"] = aligned["n_ingredients"].fillna(0).astype(int)
    if "acquisition_value" in aligned.columns:
        aligned["acquisition_value"] = aligned["acquisition_value"].astype(float)
    aligned = normalize_formulation_dataframe(aligned, feature_names)
    aligned["dmso_percent"] = aligned.get("dmso_M", pd.Series(np.zeros(len(aligned)))) * 78.13 / (1.10 * 10.0)
    aligned["n_ingredients"] = [
        len(active_features(row, feature_names)) for _, row in aligned.iterrows()
    ]
    return aligned


def load_validation_df(feature_names: Sequence[str]) -> pd.DataFrame:
    """Load measured wet-lab results with parsed stage labels."""
    if not VALIDATION_PATH.exists():
        raise ValidationError(f"Missing validation results file: {VALIDATION_PATH}")

    validation_df = pd.read_csv(VALIDATION_PATH)
    if validation_df.empty:
        raise ValidationError(f"{VALIDATION_PATH} is empty.")

    missing_required = sorted(REQUIRED_VALIDATION_COLUMNS - set(validation_df.columns))
    if missing_required:
        raise ValidationError(
            f"{VALIDATION_PATH} is missing required columns: {', '.join(missing_required)}"
        )

    missing_features = [name for name in feature_names if name not in validation_df.columns]
    if missing_features:
        raise ValidationError(
            f"{VALIDATION_PATH} is missing feature columns required by the active model: "
            f"{', '.join(missing_features)}"
        )

    validation_df = validation_df.copy()
    validation_df["viability_measured"] = pd.to_numeric(
        validation_df["viability_measured"], errors="coerce"
    )
    validation_df = validation_df[validation_df["viability_measured"].notna()].copy()
    if validation_df.empty:
        raise ValidationError(f"{VALIDATION_PATH} has no measured validation rows.")

    validation_df["stage"] = validation_df["experiment_id"].map(parse_experiment_stage)
    if validation_df["stage"].isna().any():
        bad_ids = sorted(validation_df.loc[validation_df["stage"].isna(), "experiment_id"].astype(str).unique())
        raise ValidationError(f"Could not parse stages from experiment IDs: {', '.join(bad_ids)}")

    for feature_name in feature_names:
        validation_df[feature_name] = pd.to_numeric(validation_df[feature_name], errors="coerce").fillna(0.0)
    return validation_df


def build_tested_signatures(validation_df: pd.DataFrame, feature_names: Sequence[str]) -> set[str]:
    """Build the canonical tested-formulation signature set."""
    return {
        format_formulation(row, feature_names)
        for _, row in validation_df.iterrows()
    }


def build_bo_context(active_stage: StageArtifacts) -> BayesianOptimizer:
    """Construct a BO helper so the probe generator uses the same bounds and sparsity rules."""
    config = BOConfig(random_seed=GENERATION_SEED)
    optimizer = BayesianOptimizer(
        active_stage.model,
        active_stage.scaler,
        active_stage.feature_names,
        config=config,
        is_composite=active_stage.is_composite_model,
    )
    optimizer._fit_search_context(active_stage.observed_context.copy())
    return optimizer


def load_bo_candidate_pool(
    active_stage: StageArtifacts,
    validation_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Path]]:
    """Load active-stage BO candidate files and remove already tested formulations."""
    iteration_tag = active_stage.iteration_dir
    candidate_paths = [
        RESULTS_DIR / f"bo_candidates_general_{iteration_tag}.csv",
        RESULTS_DIR / f"bo_candidates_dmso_free_{iteration_tag}.csv",
    ]
    missing = [str(path) for path in candidate_paths if not path.exists()]
    if missing:
        raise ValidationError(f"Missing BO candidate files: {', '.join(missing)}")

    tested_signatures = build_tested_signatures(validation_df, active_stage.feature_names)
    rows: List[Dict] = []

    for path in candidate_paths:
        candidate_df = align_candidate_df(pd.read_csv(path), active_stage.feature_names, path)
        for _, row in candidate_df.iterrows():
            signature = format_formulation(row, active_stage.feature_names)
            if signature in tested_signatures:
                continue
            record = {name: float(row.get(name, 0.0)) for name in active_stage.feature_names}
            dmso_percent = float(record.get("dmso_M", 0.0)) * 78.13 / (1.10 * 10.0)
            record.update(
                {
                    "origin": "bo_candidate",
                    "source_file": path.name,
                    "source_rank": int(row["rank"]),
                    "source_kind": "bo",
                    "predicted_viability": float(row["predicted_viability"]),
                    "uncertainty": float(row["uncertainty"]),
                    "acquisition_value": round_or_none(row.get("acquisition_value")),
                    "dmso_percent": dmso_percent,
                    "n_ingredients": len(active_features(pd.Series(record), active_stage.feature_names)),
                    "signature": signature,
                    "chemistry_family": chemistry_family(row, active_stage.feature_names),
                    "anchor_stage": None,
                    "anchor_experiments": "",
                    "rationale": "",
                }
            )
            rows.append(record)

    pool = pd.DataFrame(rows)
    if pool.empty:
        raise ValidationError("No untested BO candidates remain for the active stage.")

    pool = pool.drop_duplicates(subset=["signature"]).reset_index(drop=True)
    return pool, candidate_paths


def compute_stage_batch(
    validation_df: pd.DataFrame,
    stage_artifacts: StageArtifacts,
    active_stage: Optional[StageArtifacts] = None,
) -> pd.DataFrame:
    """Load one completed stage batch and attach frozen-model residuals."""
    batch = validation_df[validation_df["stage"] == stage_artifacts.stage].copy()
    if batch.empty:
        raise ValidationError(f"No validation rows found for completed stage {stage_artifacts.stage}.")

    stage_features = [name for name in stage_artifacts.feature_names if name in batch.columns]
    missing_stage_features = [name for name in stage_artifacts.feature_names if name not in batch.columns]
    if missing_stage_features:
        raise ValidationError(
            f"Validation data is missing feature columns required by stage {stage_artifacts.stage}: "
            f"{', '.join(missing_stage_features)}"
        )

    X_stage = batch.reindex(columns=stage_features, fill_value=0.0).fillna(0.0).to_numpy(float)
    predicted, uncertainty = predict(
        stage_artifacts.model,
        stage_artifacts.scaler,
        X_stage,
        stage_artifacts.is_composite_model,
    )
    batch["predicted_stage_model"] = predicted
    batch["uncertainty_stage_model"] = uncertainty
    batch["residual"] = batch["viability_measured"].astype(float) - batch["predicted_stage_model"].astype(float)
    batch["signature"] = [
        format_formulation(row, stage_artifacts.feature_names) for _, row in batch.iterrows()
    ]
    batch["frozen_stage"] = int(stage_artifacts.stage)
    batch["frozen_iteration_dir"] = str(stage_artifacts.iteration_dir)

    if active_stage is not None:
        X_active = batch.reindex(columns=active_stage.feature_names, fill_value=0.0).fillna(0.0).to_numpy(float)
        active_predicted, active_uncertainty = predict(
            active_stage.model,
            active_stage.scaler,
            X_active,
            active_stage.is_composite_model,
        )
        batch["predicted_active_model"] = active_predicted
        batch["uncertainty_active_model"] = active_uncertainty

    return batch


def compute_previous_stage_batch(
    validation_df: pd.DataFrame,
    previous_stage: StageArtifacts,
) -> pd.DataFrame:
    """Load the most recent completed batch and attach previous-stage residuals."""
    batch = compute_stage_batch(validation_df, previous_stage)
    batch["predicted_previous_stage"] = batch["predicted_stage_model"]
    batch["uncertainty_previous_stage"] = batch["uncertainty_stage_model"]
    return batch


def build_historical_residual_df(
    validation_df: pd.DataFrame,
    active_stage: StageArtifacts,
    completed_stages: Sequence[int],
) -> pd.DataFrame:
    """Assemble all completed wet-lab batches scored by their own frozen models."""
    historical_batches: List[pd.DataFrame] = []
    for stage in completed_stages:
        stage_artifacts = load_stage_artifacts(int(stage), require_observed_context=False)
        historical_batches.append(compute_stage_batch(validation_df, stage_artifacts, active_stage=active_stage))

    if not historical_batches:
        raise ValidationError("No completed validation stages are available for historical blind-spot scoring.")

    historical_df = pd.concat(historical_batches, ignore_index=True)
    return historical_df


def build_wetlab_coverage_counts(
    validation_df: pd.DataFrame,
    feature_names: Sequence[str],
) -> Tuple[Dict[str, int], Dict[Tuple[str, str], int]]:
    """Count feature and pair coverage across all measured wet-lab history."""
    wetlab_feature_counts = {name: 0 for name in feature_names}
    wetlab_pair_counts: Dict[Tuple[str, str], int] = {}
    for _, row in validation_df.iterrows():
        active = active_features(row, feature_names)
        for feature_name in active:
            wetlab_feature_counts[feature_name] += 1
        for pair in combinations(sorted(active), 2):
            wetlab_pair_counts[pair] = wetlab_pair_counts.get(pair, 0) + 1
    return wetlab_feature_counts, wetlab_pair_counts


def finalize_blindspot_details(
    raw_signal_totals: Dict,
    weight_totals: Dict,
    experiment_support: Dict,
    batch_support: Dict,
) -> Tuple[Dict, Dict]:
    """Convert raw weighted residual accumulators into final support-adjusted signals."""
    final_signal = {key: 0.0 for key in raw_signal_totals}
    details: Dict = {}

    for key, raw_total in raw_signal_totals.items():
        total_weight = float(weight_totals.get(key, 0.0))
        experiment_count = len(experiment_support.get(key, set()))
        batch_count = len(batch_support.get(key, set()))
        raw_signal = raw_total / total_weight if total_weight > 0.0 else 0.0
        experiment_shrink = (
            experiment_count / (experiment_count + BLINDSPOT_EXPERIMENT_SHRINK_OFFSET)
            if experiment_count
            else 0.0
        )
        batch_shrink = (
            batch_count / (batch_count + BLINDSPOT_BATCH_SHRINK_OFFSET)
            if batch_count
            else 0.0
        )
        passes_support_filter = experiment_count >= BLINDSPOT_SUPPORT_EXPERIMENT_MIN
        adjusted_signal = (
            raw_signal * experiment_shrink * batch_shrink
            if passes_support_filter
            else 0.0
        )
        final_signal[key] = adjusted_signal
        details[key] = {
            "raw_signal": raw_signal,
            "final_signal": adjusted_signal,
            "positive_residual_experiment_support": experiment_count,
            "positive_residual_batch_support": batch_count,
            "experiment_shrink": experiment_shrink,
            "batch_shrink": batch_shrink,
            "passes_support_filter": passes_support_filter,
        }

    return final_signal, details


def top_signal_entries(
    signal_details: Dict,
    *,
    is_pair: bool = False,
    limit: int = 10,
) -> List[Dict]:
    """Return the top support-adjusted feature or pair signal entries."""
    ranked = sorted(
        signal_details.items(),
        key=lambda item: (
            float(item[1]["final_signal"]),
            float(item[1]["raw_signal"]),
            int(item[1]["positive_residual_experiment_support"]),
            int(item[1]["positive_residual_batch_support"]),
        ),
        reverse=True,
    )

    output: List[Dict] = []
    for key, detail in ranked:
        if float(detail["final_signal"]) <= 0.0:
            continue
        entry = {
            "score": round_or_none(detail["final_signal"]),
            "final_signal": round_or_none(detail["final_signal"]),
            "raw_signal": round_or_none(detail["raw_signal"]),
            "positive_residual_experiment_support": int(detail["positive_residual_experiment_support"]),
            "positive_residual_batch_support": int(detail["positive_residual_batch_support"]),
            "experiment_shrink": round_or_none(detail["experiment_shrink"]),
            "batch_shrink": round_or_none(detail["batch_shrink"]),
        }
        if is_pair:
            entry["pair"] = list(key)
        else:
            entry["feature"] = str(key)
        output.append(entry)
        if len(output) >= limit:
            break
    return output


def compute_blindspot_signals(
    historical_residual_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_names: Sequence[str],
    last_completed_stage: int,
) -> Tuple[
    Dict[str, float],
    Dict[Tuple[str, str], float],
    Dict[str, int],
    Dict[Tuple[str, str], int],
    Dict[str, Dict[str, float]],
    Dict[Tuple[str, str], Dict[str, float]],
    Dict[str, object],
]:
    """Summarize support-adjusted blind spots from historical positive residuals."""
    feature_raw_totals = {name: 0.0 for name in feature_names}
    feature_weight_totals = {name: 0.0 for name in feature_names}
    feature_experiment_support = {name: set() for name in feature_names}
    feature_batch_support = {name: set() for name in feature_names}

    pair_raw_totals: Dict[Tuple[str, str], float] = {}
    pair_weight_totals: Dict[Tuple[str, str], float] = {}
    pair_experiment_support: Dict[Tuple[str, str], set[str]] = {}
    pair_batch_support: Dict[Tuple[str, str], set[int]] = {}

    positive_rows = historical_residual_df[historical_residual_df["residual"].astype(float) > 0.0].copy()
    for _, row in positive_rows.iterrows():
        active = active_features(row, feature_names)
        if not active:
            continue

        residual = float(row["residual"])
        stage = int(row["stage"])
        experiment_id = str(row["experiment_id"])
        recency_weight = BLINDSPOT_RECENCY_DECAY ** max(0, last_completed_stage - stage)
        region_weight = (
            BLINDSPOT_REGION_WEIGHT
            if float(row.get("predicted_active_model", 0.0)) >= BLINDSPOT_REGION_PREDICTION_THRESHOLD
            else 1.0
        )
        row_weight = recency_weight * region_weight

        feature_share = residual / len(active)
        for feature_name in active:
            feature_raw_totals[feature_name] += row_weight * feature_share
            feature_weight_totals[feature_name] += row_weight
            feature_experiment_support[feature_name].add(experiment_id)
            feature_batch_support[feature_name].add(stage)

        pair_share = residual / max(1, len(active) - 1)
        for pair in combinations(sorted(active), 2):
            pair_raw_totals[pair] = pair_raw_totals.get(pair, 0.0) + row_weight * pair_share
            pair_weight_totals[pair] = pair_weight_totals.get(pair, 0.0) + row_weight
            pair_experiment_support.setdefault(pair, set()).add(experiment_id)
            pair_batch_support.setdefault(pair, set()).add(stage)

    feature_signal, feature_details = finalize_blindspot_details(
        feature_raw_totals,
        feature_weight_totals,
        feature_experiment_support,
        feature_batch_support,
    )
    pair_signal, pair_details = finalize_blindspot_details(
        pair_raw_totals,
        pair_weight_totals,
        pair_experiment_support,
        pair_batch_support,
    )

    wetlab_feature_counts, wetlab_pair_counts = build_wetlab_coverage_counts(validation_df, feature_names)

    blindspot_audit = {
        "blindspot_signal_mode": BLINDSPOT_SIGNAL_MODE,
        "blindspot_stage_range": [
            int(historical_residual_df["stage"].min()),
            int(historical_residual_df["stage"].max()),
        ],
        "blindspot_recency_decay": float(BLINDSPOT_RECENCY_DECAY),
        "blindspot_region_prediction_threshold": float(BLINDSPOT_REGION_PREDICTION_THRESHOLD),
        "blindspot_region_weight": float(BLINDSPOT_REGION_WEIGHT),
        "blindspot_support_experiment_min": int(BLINDSPOT_SUPPORT_EXPERIMENT_MIN),
        "blindspot_experiment_shrink_offset": float(BLINDSPOT_EXPERIMENT_SHRINK_OFFSET),
        "blindspot_batch_shrink_offset": float(BLINDSPOT_BATCH_SHRINK_OFFSET),
        "historical_positive_residual_rows": int(len(positive_rows)),
        "top_positive_features": top_signal_entries(feature_details, is_pair=False),
        "top_positive_pairs": top_signal_entries(pair_details, is_pair=True),
    }

    return (
        feature_signal,
        pair_signal,
        wetlab_feature_counts,
        wetlab_pair_counts,
        feature_details,
        pair_details,
        blindspot_audit,
    )


def agree_within_ten_percent(left: float, right: float) -> bool:
    """Return True when two positive concentrations are effectively the same."""
    if abs(left) <= FEATURE_THRESHOLD or abs(right) <= FEATURE_THRESHOLD:
        return False
    scale = max(abs(left), abs(right), FEATURE_THRESHOLD)
    return abs(left - right) <= 0.10 * scale


def vector_to_record(
    vector: np.ndarray,
    feature_names: Sequence[str],
    origin: str,
    anchor_stage: int,
    anchor_experiments: Sequence[str],
    source_file: Optional[str] = None,
    source_rank: Optional[int] = None,
) -> Dict:
    """Convert one formulation vector into the common record shape."""
    normalized_vector = normalize_formulation_vector(vector, feature_names)
    record = {name: float(normalized_vector[idx]) for idx, name in enumerate(feature_names)}
    row = pd.Series(record)
    dmso_molar = float(record.get("dmso_M", 0.0))
    dmso_percent = dmso_molar * 78.13 / (1.10 * 10.0)
    record.update(
        {
            "origin": origin,
            "source_file": source_file or "",
            "source_rank": source_rank,
            "source_kind": "generated" if origin in {"generated_probe", "blindspot_probe", "local_rank_probe"} else "bo",
            "signature": format_formulation(row, feature_names),
            "chemistry_family": chemistry_family(row, feature_names),
            "anchor_stage": anchor_stage,
            "anchor_experiments": ";".join(sorted(set(anchor_experiments))),
            "dmso_percent": dmso_percent,
            "n_ingredients": int(sum(abs(record[name]) > FEATURE_THRESHOLD for name in feature_names)),
        }
    )
    return record


def sparsify_vector(
    vector: np.ndarray,
    feature_names: Sequence[str],
    max_ingredients: int,
    protected_features: Sequence[str],
) -> np.ndarray:
    """Limit active ingredient count while preserving the requested protected features."""
    x = vector.copy()
    if max_ingredients <= 0:
        return x

    active_idx = [idx for idx, value in enumerate(x) if abs(value) > FEATURE_THRESHOLD]
    if len(active_idx) <= max_ingredients:
        return x

    protected_idx = {
        feature_names.index(name) for name in protected_features if name in feature_names
    }
    removable = [idx for idx in active_idx if idx not in protected_idx]
    removable.sort(key=lambda idx: abs(x[idx]))

    for idx in removable:
        if sum(abs(value) > FEATURE_THRESHOLD for value in x) <= max_ingredients:
            break
        x[idx] = 0.0

    if sum(abs(value) > FEATURE_THRESHOLD for value in x) > max_ingredients:
        for idx in active_idx:
            if idx in protected_idx:
                continue
            if sum(abs(value) > FEATURE_THRESHOLD for value in x) <= max_ingredients:
                break
            x[idx] = 0.0
    return x


def finalise_generated_vector(
    vector: np.ndarray,
    feature_names: Sequence[str],
    optimizer: BayesianOptimizer,
    protected_features: Sequence[str],
) -> np.ndarray:
    """Clip and sparsify a generated probe using BO's search constraints."""
    x = normalize_formulation_vector(optimizer._clip_to_bounds(vector.copy()), feature_names)
    x = sparsify_vector(x, feature_names, optimizer.effective_max_ingredients, protected_features)
    return normalize_formulation_vector(optimizer._clip_to_bounds(x), feature_names)


def build_midpoint_probe(
    pair: Tuple[str, str],
    anchors: Sequence[pd.Series],
    feature_names: Sequence[str],
    optimizer: BayesianOptimizer,
) -> Optional[np.ndarray]:
    """Generate one midpoint interpolation probe from two anchor rows."""
    if len(anchors) < 2:
        return None

    first, second = anchors[0], anchors[1]
    midpoint = np.zeros(len(feature_names), dtype=float)
    for idx, feature_name in enumerate(feature_names):
        left = float(first.get(feature_name, 0.0))
        right = float(second.get(feature_name, 0.0))
        if feature_name in pair:
            midpoint[idx] = (left + right) / 2.0
        elif agree_within_ten_percent(left, right):
            midpoint[idx] = (left + right) / 2.0

    midpoint = finalise_generated_vector(midpoint, feature_names, optimizer, pair)
    if sum(abs(value) > FEATURE_THRESHOLD for value in midpoint) < len(pair):
        return None
    return midpoint


def build_scaled_probe(
    pair: Tuple[str, str],
    anchor: pd.Series,
    feature_names: Sequence[str],
    optimizer: BayesianOptimizer,
    scale: float,
) -> Optional[np.ndarray]:
    """Generate one local perturbation probe around a positive-residual anchor."""
    vector = np.array([float(anchor.get(feature_name, 0.0)) for feature_name in feature_names], dtype=float)
    active_pair = False
    for feature_name in pair:
        if feature_name not in feature_names:
            continue
        idx = feature_names.index(feature_name)
        if abs(vector[idx]) > FEATURE_THRESHOLD:
            vector[idx] *= scale
            active_pair = True
    if not active_pair:
        return None
    vector = finalise_generated_vector(vector, feature_names, optimizer, pair)
    return vector


def build_local_rank_probe(
    anchor_row: Dict,
    scaled_features: Sequence[str],
    feature_names: Sequence[str],
    optimizer: BayesianOptimizer,
    scale_factor: float,
) -> Optional[np.ndarray]:
    """Generate one local rank-resolution probe around an exploit anchor."""
    vector = np.array([float(anchor_row.get(feature_name, 0.0)) for feature_name in feature_names], dtype=float)
    active_features_scaled = False
    for feature_name in scaled_features:
        if feature_name not in feature_names:
            continue
        idx = feature_names.index(feature_name)
        if abs(vector[idx]) <= FEATURE_THRESHOLD:
            continue
        vector[idx] *= scale_factor
        active_features_scaled = True
    if not active_features_scaled:
        return None
    return finalise_generated_vector(vector, feature_names, optimizer, scaled_features)


def compute_blindspot_score(
    row: pd.Series,
    feature_names: Sequence[str],
    feature_signal: Dict[str, float],
    pair_signal: Dict[Tuple[str, str], float],
) -> float:
    """Score how strongly a formulation hits positive residual blind spots."""
    active = active_features(row, feature_names)
    if not active:
        return 0.0
    feature_component = sum(max(feature_signal.get(name, 0.0), 0.0) for name in active) / len(active)
    pairs = list(combinations(sorted(active), 2))
    if not pairs:
        return feature_component
    pair_component = sum(max(pair_signal.get(pair, 0.0), 0.0) for pair in pairs) / len(pairs)
    return feature_component + 0.75 * pair_component


def compute_novelty_score(
    row: pd.Series,
    feature_names: Sequence[str],
    wetlab_feature_counts: Dict[str, int],
    wetlab_pair_counts: Dict[Tuple[str, str], int],
) -> float:
    """Score how weakly a formulation is represented in wet-lab history."""
    active = active_features(row, feature_names)
    values: List[float] = []
    for feature_name in active:
        values.append(1.0 / (1.0 + wetlab_feature_counts.get(feature_name, 0)))
    for pair in combinations(sorted(active), 2):
        values.append(1.0 / (1.0 + wetlab_pair_counts.get(pair, 0)))
    if not values:
        return 0.0
    return float(np.mean(values))


def score_records_with_active_model(
    records: List[Dict],
    active_stage: StageArtifacts,
    feature_signal: Dict[str, float],
    pair_signal: Dict[Tuple[str, str], float],
    wetlab_feature_counts: Dict[str, int],
    wetlab_pair_counts: Dict[Tuple[str, str], int],
) -> pd.DataFrame:
    """Attach prediction, blind-spot, novelty, and family information to a record list."""
    if not records:
        return pd.DataFrame()

    scored = pd.DataFrame(records)
    X = scored.reindex(columns=active_stage.feature_names, fill_value=0.0).fillna(0.0).to_numpy(float)
    pred_mean, pred_std = predict(
        active_stage.model,
        active_stage.scaler,
        X,
        active_stage.is_composite_model,
    )
    scored["predicted_viability"] = pred_mean
    scored["uncertainty"] = pred_std

    blindspot_scores = []
    novelty_scores = []
    families = []
    for _, row in scored.iterrows():
        blindspot_scores.append(
            compute_blindspot_score(row, active_stage.feature_names, feature_signal, pair_signal)
        )
        novelty_scores.append(
            compute_novelty_score(row, active_stage.feature_names, wetlab_feature_counts, wetlab_pair_counts)
        )
        families.append(chemistry_family(row, active_stage.feature_names))
    scored["blindspot_score"] = blindspot_scores
    scored["novelty_score"] = novelty_scores
    scored["chemistry_family"] = families
    scored["signature"] = [
        format_formulation(row, active_stage.feature_names) for _, row in scored.iterrows()
    ]
    scored["dmso_percent"] = scored.get("dmso_percent", pd.Series(dtype=float)).fillna(
        scored.get("dmso_M", pd.Series(np.zeros(len(scored)))) * 78.13 / (1.10 * 10.0)
    )
    scored["n_ingredients"] = [
        len(active_features(row, active_stage.feature_names)) for _, row in scored.iterrows()
    ]
    return scored


def build_local_rank_probe_rows(
    exploit_rows: Sequence[Dict],
    active_stage: StageArtifacts,
    optimizer: BayesianOptimizer,
    feature_signal: Dict[str, float],
    pair_signal: Dict[Tuple[str, str], float],
    wetlab_feature_counts: Dict[str, int],
    wetlab_pair_counts: Dict[Tuple[str, str], int],
    tested_signatures: set[str],
) -> Tuple[List[Dict], Dict[str, object]]:
    """Generate local rank-resolution probes around the top exploit anchors."""
    anchor_rows = [dict(row) for row in exploit_rows[:LOCAL_RANK_ANCHOR_COUNT]]
    if not anchor_rows:
        return [], {"local_rank_anchor_count": 0, "local_rank_probe_count": 0}

    generated_records: List[Dict] = []
    seen_signatures = set(tested_signatures) | {str(row["signature"]) for row in anchor_rows}

    for anchor_index, anchor in enumerate(anchor_rows, start=1):
        scaled_features = top_features_by_magnitude(pd.Series(anchor), active_stage.feature_names, limit=2)
        if len(scaled_features) < 2:
            continue

        source_file = str(anchor.get("source_file", ""))
        source_rank = int(anchor.get("source_rank", 0) or 0)
        for direction, primary_factor, retry_factor in (
            ("down", 1.0 - LOCAL_RANK_PRIMARY_DELTA, 1.0 - LOCAL_RANK_RETRY_DELTA),
            ("up", 1.0 + LOCAL_RANK_PRIMARY_DELTA, 1.0 + LOCAL_RANK_RETRY_DELTA),
        ):
            selected_record: Optional[Dict] = None
            for scale_factor in (primary_factor, retry_factor):
                probe = build_local_rank_probe(
                    anchor,
                    scaled_features,
                    active_stage.feature_names,
                    optimizer,
                    scale_factor,
                )
                if probe is None:
                    continue
                probe_record = vector_to_record(
                    probe,
                    active_stage.feature_names,
                    origin="local_rank_probe",
                    anchor_stage=active_stage.stage,
                    anchor_experiments=[],
                    source_file=source_file,
                    source_rank=source_rank,
                )
                signature = str(probe_record["signature"])
                if signature in seen_signatures:
                    continue
                probe_record["probe_order"] = len(generated_records)
                delta_pct = abs(scale_factor - 1.0) * 100.0
                probe_record["rationale"] = (
                    "Local rank-resolution probe around exploit anchor "
                    f"{source_file} rank {source_rank}; "
                    f"scaled {scaled_features[0].replace('_M', '').replace('_pct', '')} + "
                    f"{scaled_features[1].replace('_M', '').replace('_pct', '')} "
                    f"{direction} by {delta_pct:.0f}%."
                )
                selected_record = probe_record
                seen_signatures.add(signature)
                break
            if selected_record is not None:
                generated_records.append(selected_record)

    if not generated_records:
        return [], {
            "local_rank_anchor_count": int(len(anchor_rows)),
            "local_rank_probe_count": 0,
        }

    scored = score_records_with_active_model(
        generated_records,
        active_stage,
        feature_signal,
        pair_signal,
        wetlab_feature_counts,
        wetlab_pair_counts,
    )
    scored = scored[
        (scored["predicted_viability"] >= MIN_EXPLORATION_PREDICTION)
        & (scored["dmso_percent"] <= optimizer.config.max_dmso_percent + 1e-6)
    ].copy()
    if scored.empty:
        return [], {
            "local_rank_anchor_count": int(len(anchor_rows)),
            "local_rank_probe_count": 0,
        }

    scored = scored.sort_values("probe_order", kind="mergesort").head(LOCAL_RANK_PROBE_COUNT).copy()
    return scored.to_dict(orient="records"), {
        "local_rank_anchor_count": int(len(anchor_rows)),
        "local_rank_probe_count": int(len(scored)),
    }


def rank_generated_probes(
    generated: pd.DataFrame,
) -> pd.DataFrame:
    """Rank generated probes for exploration selection."""
    if generated.empty:
        return generated

    ranked = generated.copy()
    ranked["pred_norm"] = normalize(ranked["predicted_viability"])
    ranked["unc_norm"] = normalize(ranked["uncertainty"])
    ranked["blindspot_norm"] = normalize(ranked["blindspot_score"].clip(lower=0.0))
    ranked["novelty_norm"] = normalize(ranked["novelty_score"])
    ranked["exploration_score"] = (
        0.35 * ranked["unc_norm"]
        + 0.35 * ranked["blindspot_norm"]
        + 0.20 * ranked["novelty_norm"]
        + 0.10 * ranked["pred_norm"]
    )
    return ranked.sort_values(
        ["exploration_score", "blindspot_score", "uncertainty", "predicted_viability"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def build_generated_probe_pool(
    active_stage: StageArtifacts,
    historical_residual_df: pd.DataFrame,
    optimizer: BayesianOptimizer,
    feature_signal: Dict[str, float],
    pair_signal: Dict[Tuple[str, str], float],
    wetlab_feature_counts: Dict[str, int],
    wetlab_pair_counts: Dict[Tuple[str, str], int],
    tested_signatures: set[str],
    min_positive_residual: float,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Generate calibration probes from positive-residual blind spots."""
    anchor_df = historical_residual_df.copy()
    anchor_df["actual_priority"] = anchor_df["viability_measured"].astype(float) >= MEANINGFUL_ACTUAL_MIN
    anchor_df["promising_region"] = (
        anchor_df.get("predicted_active_model", pd.Series(np.zeros(len(anchor_df)), index=anchor_df.index))
        .astype(float)
        >= BLINDSPOT_REGION_PREDICTION_THRESHOLD
    )
    anchor_df["is_positive_anchor"] = anchor_df["residual"].astype(float) > min_positive_residual

    positive_rows = anchor_df[anchor_df["is_positive_anchor"]].copy()
    if positive_rows.empty:
        return pd.DataFrame(), {
            "anchor_count": 0,
            "generated_anchor_stage_counts": {},
            "generated_anchor_stage_range": [],
            "generated_top_anchor_pairs": [],
        }

    anchor_rows = positive_rows.sort_values(
        ["promising_region", "actual_priority", "residual", "stage", "viability_measured"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    pair_to_rows: Dict[Tuple[str, str], List[pd.Series]] = {}
    for _, row in anchor_rows.iterrows():
        active = active_features(row, active_stage.feature_names)
        for pair in combinations(sorted(active), 2):
            pair_to_rows.setdefault(pair, []).append(row)

    generated_records: List[Dict] = []
    seen_signatures: set[str] = set()
    generated_anchor_stages: List[int] = []
    generated_anchor_pairs: List[Tuple[str, str]] = []
    ranked_pairs = [
        pair
        for pair, score in sorted(pair_signal.items(), key=lambda item: item[1], reverse=True)
        if score > 0.0
    ]

    for pair in ranked_pairs:
        anchors = pair_to_rows.get(pair, [])
        if not anchors:
            continue

        if len(anchors) >= 2:
            midpoint = build_midpoint_probe(pair, anchors, active_stage.feature_names, optimizer)
            if midpoint is not None:
                midpoint_record = vector_to_record(
                    midpoint,
                    active_stage.feature_names,
                    origin="blindspot_probe",
                    anchor_stage=int(anchors[0]["stage"]),
                    anchor_experiments=[
                        str(anchors[0]["experiment_id"]),
                        str(anchors[1]["experiment_id"]),
                    ],
                )
                midpoint_record["rationale"] = (
                    "Midpoint calibration probe from underpredicted "
                    f"{pair[0].replace('_M', '').replace('_pct', '')} + "
                    f"{pair[1].replace('_M', '').replace('_pct', '')} anchors."
                )
                if midpoint_record["signature"] not in tested_signatures:
                    generated_records.append(midpoint_record)
                    seen_signatures.add(midpoint_record["signature"])
                    generated_anchor_stages.extend([int(anchors[0]["stage"]), int(anchors[1]["stage"])])
                    generated_anchor_pairs.append(pair)

        anchor = anchors[0]
        for scale in (0.75, 1.25):
            probe = build_scaled_probe(pair, anchor, active_stage.feature_names, optimizer, scale)
            if probe is None:
                continue
            record = vector_to_record(
                probe,
                active_stage.feature_names,
                origin="blindspot_probe",
                anchor_stage=int(anchor["stage"]),
                anchor_experiments=[str(anchor["experiment_id"])],
            )
            if record["signature"] in tested_signatures or record["signature"] in seen_signatures:
                continue
            record["rationale"] = (
                "Local calibration probe around underpredicted "
                f"{pair[0].replace('_M', '').replace('_pct', '')} + "
                f"{pair[1].replace('_M', '').replace('_pct', '')} chemistry."
            )
            generated_records.append(record)
            seen_signatures.add(record["signature"])
            generated_anchor_stages.append(int(anchor["stage"]))
            generated_anchor_pairs.append(pair)

    if not generated_records:
        return pd.DataFrame(), {
            "anchor_count": int(len(anchor_rows)),
            "generated_anchor_stage_counts": {},
            "generated_anchor_stage_range": [],
            "generated_top_anchor_pairs": [],
        }

    generated = score_records_with_active_model(
        generated_records,
        active_stage,
        feature_signal,
        pair_signal,
        wetlab_feature_counts,
        wetlab_pair_counts,
    )
    generated = generated[
        (generated["predicted_viability"] >= MIN_EXPLORATION_PREDICTION)
        & (generated["dmso_percent"] <= optimizer.config.max_dmso_percent + 1e-6)
    ].copy()
    generated = generated.drop_duplicates(subset=["signature"]).reset_index(drop=True)
    ranked_generated = rank_generated_probes(generated)

    stage_counts = pd.Series(generated_anchor_stages).value_counts().sort_index()
    pair_counts = pd.Series(generated_anchor_pairs).value_counts()
    generated_audit = {
        "anchor_count": int(len(anchor_rows)),
        "generated_anchor_stage_counts": {str(int(stage)): int(count) for stage, count in stage_counts.items()},
        "generated_anchor_stage_range": (
            [int(min(generated_anchor_stages)), int(max(generated_anchor_stages))]
            if generated_anchor_stages
            else []
        ),
        "generated_top_anchor_pairs": [
            {
                "pair": list(pair),
                "count": int(count),
            }
            for pair, count in pair_counts.items()
        ][:10],
    }
    return ranked_generated, generated_audit


def build_bo_exploration_fallback_pool(
    candidate_pool: pd.DataFrame,
    active_stage: StageArtifacts,
    feature_signal: Dict[str, float],
    pair_signal: Dict[Tuple[str, str], float],
    wetlab_feature_counts: Dict[str, int],
    wetlab_pair_counts: Dict[Tuple[str, str], int],
) -> pd.DataFrame:
    """Score BO candidates as a fallback exploration pool."""
    scored = candidate_pool.copy()
    blindspots = []
    novelties = []
    for _, row in scored.iterrows():
        blindspots.append(compute_blindspot_score(row, active_stage.feature_names, feature_signal, pair_signal))
        novelties.append(compute_novelty_score(row, active_stage.feature_names, wetlab_feature_counts, wetlab_pair_counts))
    scored["blindspot_score"] = blindspots
    scored["novelty_score"] = novelties
    scored["pred_norm"] = normalize(scored["predicted_viability"])
    scored["unc_norm"] = normalize(scored["uncertainty"])
    scored["blindspot_norm"] = normalize(scored["blindspot_score"].clip(lower=0.0))
    scored["novelty_norm"] = normalize(scored["novelty_score"])
    scored["exploration_score"] = (
        0.35 * scored["unc_norm"]
        + 0.35 * scored["blindspot_norm"]
        + 0.20 * scored["novelty_norm"]
        + 0.10 * scored["pred_norm"]
    )
    return scored.sort_values(
        ["exploration_score", "blindspot_score", "uncertainty", "predicted_viability"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def select_diverse_rows(
    scored: pd.DataFrame,
    count: int,
    score_column: str,
    family_limit: int,
    min_predicted_viability: float,
    disallowed_signatures: Optional[set[str]] = None,
) -> List[Dict]:
    """Select diverse rows using a simple family cap."""
    if scored.empty:
        return []

    required_columns = [score_column, "predicted_viability", "uncertainty", "signature", "chemistry_family"]
    missing_columns = [column for column in required_columns if column not in scored.columns]
    if missing_columns:
        raise ValidationError(
            "Selection pool is missing required columns: " + ", ".join(sorted(missing_columns))
        )

    selected: List[Dict] = []
    family_counts: Dict[str, int] = {}
    blocked_signatures = set(disallowed_signatures or set())

    ranked = scored.sort_values(
        [score_column, "predicted_viability", "uncertainty"],
        ascending=[False, False, True],
    )
    for _, row in ranked.iterrows():
        signature = str(row["signature"])
        if signature in blocked_signatures:
            continue
        if float(row["predicted_viability"]) < min_predicted_viability:
            continue
        family = str(row["chemistry_family"])
        if family_counts.get(family, 0) >= family_limit:
            continue
        selected.append(row.to_dict())
        blocked_signatures.add(signature)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) >= count:
            break
    return selected


def select_generated_exploration_rows(
    generated_pool: pd.DataFrame,
    already_selected_signatures: set[str],
    count: int = BLINDSPOT_PROBE_COUNT,
) -> Tuple[List[Dict], Dict[str, int]]:
    """Select generated exploration rows before consulting BO fallback."""
    strict_selected = select_diverse_rows(
        generated_pool,
        count=count,
        score_column="exploration_score",
        family_limit=EXPLORE_FAMILY_LIMIT,
        min_predicted_viability=MIN_EXPLORATION_PREDICTION,
        disallowed_signatures=set(already_selected_signatures),
    )
    selected = list(strict_selected)
    if len(selected) < count:
        disallowed = set(already_selected_signatures) | {str(row["signature"]) for row in selected}
        relaxed_topup = select_diverse_rows(
            generated_pool,
            count=count - len(selected),
            score_column="exploration_score",
            family_limit=count,
            min_predicted_viability=MIN_EXPLORATION_PREDICTION,
            disallowed_signatures=disallowed,
        )
        selected.extend(relaxed_topup)
    return selected, {
        "strict_selected_rows": len(strict_selected),
        "relaxed_selected_rows": len(selected),
    }


def choose_generated_exploration_rows(
    active_stage: StageArtifacts,
    historical_residual_df: pd.DataFrame,
    optimizer: BayesianOptimizer,
    feature_signal: Dict[str, float],
    pair_signal: Dict[Tuple[str, str], float],
    wetlab_feature_counts: Dict[str, int],
    wetlab_pair_counts: Dict[Tuple[str, str], int],
    tested_signatures: set[str],
    already_selected_signatures: set[str],
    positive_residual_thresholds: Sequence[float] = POSITIVE_RESIDUAL_THRESHOLDS,
) -> Dict[str, object]:
    """Select the best generated exploration slate across configured thresholds."""
    threshold_values = [float(value) for value in positive_residual_thresholds]
    if not threshold_values:
        raise ValidationError("No positive residual thresholds configured for exploration generation.")

    attempts: List[Dict[str, object]] = []
    best_attempt: Optional[Dict[str, object]] = None
    residuals = historical_residual_df["residual"].astype(float)

    for threshold in threshold_values:
        anchor_count = int((residuals > threshold).sum())
        generated_pool, generated_audit = build_generated_probe_pool(
            active_stage,
            historical_residual_df,
            optimizer,
            feature_signal,
            pair_signal,
            wetlab_feature_counts,
            wetlab_pair_counts,
            tested_signatures,
            min_positive_residual=threshold,
        )
        selected_rows, selection_counts = select_generated_exploration_rows(
            generated_pool,
            already_selected_signatures,
            count=BLINDSPOT_PROBE_COUNT,
        )
        attempt = {
            "positive_residual_threshold": threshold,
            "anchor_count": anchor_count,
            "generated_pool_rows": int(len(generated_pool)),
            "strict_selected_rows": int(selection_counts["strict_selected_rows"]),
            "relaxed_selected_rows": int(selection_counts["relaxed_selected_rows"]),
            "selected_rows": [dict(row) for row in selected_rows],
            "generated_anchor_stage_counts": dict(generated_audit["generated_anchor_stage_counts"]),
            "generated_anchor_stage_range": list(generated_audit["generated_anchor_stage_range"]),
            "generated_top_anchor_pairs": list(generated_audit["generated_top_anchor_pairs"]),
        }
        attempts.append(attempt)

        if best_attempt is None:
            best_attempt = attempt
            continue

        attempt_key = (attempt["relaxed_selected_rows"], attempt["strict_selected_rows"])
        best_key = (best_attempt["relaxed_selected_rows"], best_attempt["strict_selected_rows"])
        if attempt_key > best_key:
            best_attempt = attempt

    assert best_attempt is not None

    attempt_summaries = [
        {
            "positive_residual_threshold": float(attempt["positive_residual_threshold"]),
            "anchor_count": int(attempt["anchor_count"]),
            "generated_pool_rows": int(attempt["generated_pool_rows"]),
            "strict_selected_rows": int(attempt["strict_selected_rows"]),
            "relaxed_selected_rows": int(attempt["relaxed_selected_rows"]),
        }
        for attempt in attempts
    ]

    return {
        "selected_rows": [dict(row) for row in best_attempt["selected_rows"]],
        "audit": {
            "positive_residual_thresholds_tried": threshold_values,
            "selected_positive_residual_threshold": float(best_attempt["positive_residual_threshold"]),
            "anchor_count_at_selected_threshold": int(best_attempt["anchor_count"]),
            "generated_pool_rows_at_selected_threshold": int(best_attempt["generated_pool_rows"]),
            "generated_explore_count": int(len(best_attempt["selected_rows"])),
            "generated_anchor_stage_counts": dict(best_attempt["generated_anchor_stage_counts"]),
            "generated_anchor_stage_range": list(best_attempt["generated_anchor_stage_range"]),
            "generated_top_anchor_pairs": list(best_attempt["generated_top_anchor_pairs"]),
            "positive_residual_threshold_attempts": attempt_summaries,
        },
    }


def build_exploitation_selection(
    candidate_pool: pd.DataFrame,
) -> List[Dict]:
    """Select the final exploitation rows."""
    scored = candidate_pool.copy()
    scored["pred_norm"] = normalize(scored["predicted_viability"])
    scored["unc_norm"] = normalize(scored["uncertainty"])
    if "acquisition_value" in scored.columns:
        scored["acq_norm"] = normalize(scored["acquisition_value"].fillna(scored["acquisition_value"].min()))
    else:
        scored["acq_norm"] = 0.0
    scored["exploitation_score"] = (
        0.75 * scored["pred_norm"]
        + 0.15 * (1.0 - scored["unc_norm"])
        + 0.10 * scored["acq_norm"]
    )
    selected = select_diverse_rows(
        scored,
        count=EXPLOIT_COUNT,
        score_column="exploitation_score",
        family_limit=EXPLOIT_FAMILY_LIMIT,
        min_predicted_viability=0.0,
    )
    if len(selected) < EXPLOIT_COUNT:
        selected = select_diverse_rows(
            scored,
            count=EXPLOIT_COUNT,
            score_column="exploitation_score",
            family_limit=EXPLOIT_COUNT,
            min_predicted_viability=0.0,
        )
    for row in selected:
        row["rationale"] = "Top BO exploitation candidate with strong predicted viability."
    return selected


def build_exploration_selection(
    local_rank_rows: List[Dict],
    blindspot_rows: List[Dict],
    fallback_pool: pd.DataFrame,
    already_selected_signatures: set[str],
) -> Tuple[List[Dict], int]:
    """Select the final exploration rows, falling back to BO rows if needed."""
    selected = [dict(row) for row in local_rank_rows]
    selected.extend(dict(row) for row in blindspot_rows[:BLINDSPOT_PROBE_COUNT])
    if len(selected) >= EXPLORE_COUNT:
        return selected[:EXPLORE_COUNT], 0

    disallowed = {str(row["signature"]) for row in selected} | set(already_selected_signatures)
    selected_families = {str(row["chemistry_family"]) for row in selected}
    remaining = EXPLORE_COUNT - len(selected)
    filtered_fallback = fallback_pool[~fallback_pool["chemistry_family"].isin(selected_families)].copy()
    fallback_rows = select_diverse_rows(
        filtered_fallback,
        count=remaining,
        score_column="exploration_score",
        family_limit=EXPLORE_FAMILY_LIMIT,
        min_predicted_viability=MIN_EXPLORATION_PREDICTION,
        disallowed_signatures=disallowed,
    )
    if len(fallback_rows) < remaining:
        disallowed |= {str(row["signature"]) for row in fallback_rows}
        fallback_rows.extend(
            select_diverse_rows(
            fallback_pool,
            count=remaining - len(fallback_rows),
            score_column="exploration_score",
            family_limit=EXPLORE_COUNT,
            min_predicted_viability=MIN_EXPLORATION_PREDICTION,
            disallowed_signatures=disallowed,
        )
        )
    for row in fallback_rows:
        row["origin"] = "explore_fallback"
        if not row.get("rationale"):
            row["rationale"] = "High-uncertainty BO fallback in a blind-spot chemistry family."
    selected.extend(fallback_rows)
    return selected, len(fallback_rows)


def validate_output_rows(
    rows: List[Dict],
    active_stage: StageArtifacts,
    optimizer: BayesianOptimizer,
    tested_signatures: set[str],
) -> None:
    """Validate the final 20 outputs before writing anything."""
    if len(rows) != TOTAL_COUNT:
        raise ValidationError(f"Expected {TOTAL_COUNT} final rows, found {len(rows)}.")

    counts = pd.Series([row["recommendation_type"] for row in rows]).value_counts().to_dict()
    if counts.get("exploit", 0) != EXPLOIT_COUNT or counts.get("explore", 0) != EXPLORE_COUNT:
        raise ValidationError(
            f"Expected {EXPLOIT_COUNT} exploit and {EXPLORE_COUNT} explore rows, found {counts}."
        )

    signatures = [str(row["formulation"]) for row in rows]
    if len(signatures) != len(set(signatures)):
        raise ValidationError("Final recommendations contain duplicate formulations.")

    overlap = sorted(set(signatures) & tested_signatures)
    if overlap:
        raise ValidationError(
            "Final recommendations include already tested formulations: " + ", ".join(overlap[:5])
        )

    output_df = pd.DataFrame(rows)
    X = output_df.reindex(columns=active_stage.feature_names, fill_value=0.0).fillna(0.0).to_numpy(float)
    pred_mean, pred_std = predict(
        active_stage.model,
        active_stage.scaler,
        X,
        active_stage.is_composite_model,
    )

    for idx, row in enumerate(rows):
        if not row.get("rationale"):
            raise ValidationError(f"Row {idx + 1} is missing rationale text.")
        if row.get("origin") not in {"bo_candidate", "local_rank_probe", "blindspot_probe", "explore_fallback"}:
            raise ValidationError(f"Row {idx + 1} has unexpected origin: {row.get('origin')}")
        if not math.isclose(float(row["predicted_viability"]), float(pred_mean[idx]), rel_tol=1e-6, abs_tol=1e-6):
            raise ValidationError(f"Row {idx + 1} prediction does not match active model scoring.")
        if not math.isclose(float(row["uncertainty"]), float(pred_std[idx]), rel_tol=1e-6, abs_tol=1e-6):
            raise ValidationError(f"Row {idx + 1} uncertainty does not match active model scoring.")

        n_ingredients = sum(abs(float(row.get(name, 0.0))) > FEATURE_THRESHOLD for name in active_stage.feature_names)
        if int(row["n_ingredients"]) != n_ingredients:
            raise ValidationError(f"Row {idx + 1} n_ingredients is inconsistent.")
        if n_ingredients > optimizer.effective_max_ingredients:
            raise ValidationError(
                f"Row {idx + 1} exceeds the BO effective max ingredient count ({optimizer.effective_max_ingredients})."
            )

        dmso_percent = float(row.get("dmso_percent", 0.0))
        if dmso_percent > optimizer.config.max_dmso_percent + 1e-6:
            raise ValidationError(f"Row {idx + 1} exceeds the DMSO cap.")

        if exceeds_explicit_percentage_cap_mapping(row, active_stage.feature_names):
            pct_total = explicit_percentage_total_from_mapping(row, active_stage.feature_names)
            raise ValidationError(
                f"Row {idx + 1} exceeds the explicit percentage cap "
                f"({pct_total:.4f}% > {EXPLICIT_PERCENTAGE_CAP:.1f}%)."
            )

        for feature_index, feature_name in enumerate(active_stage.feature_names):
            low, high = optimizer.bounds[feature_index]
            value = float(row.get(feature_name, 0.0))
            if value < low - 1e-9 or value > high + 1e-9:
                raise ValidationError(f"Row {idx + 1} violates bounds for {feature_name}.")


def build_summary_text(
    active_stage: StageArtifacts,
    previous_stage: StageArtifacts,
    exploit_rows: List[Dict],
    explore_rows: List[Dict],
    feature_signal: Dict[str, float],
    pair_signal: Dict[Tuple[str, str], float],
    exploration_audit: Dict[str, object],
    blindspot_audit: Dict[str, object],
    batch_recommendations: Sequence[Dict[str, object]],
) -> str:
    """Render a human-readable summary."""
    top_features = [
        name.replace("_M", "").replace("_pct", "")
        for name, score in sorted(feature_signal.items(), key=lambda item: item[1], reverse=True)
        if score > 0.0
    ][:5]
    top_pairs = [
        " + ".join(part.replace("_M", "").replace("_pct", "") for part in pair)
        for pair, score in sorted(pair_signal.items(), key=lambda item: item[1], reverse=True)
        if score > 0.0
    ][:5]
    thresholds_tried = [float(value) for value in exploration_audit["positive_residual_thresholds_tried"]]
    default_threshold = thresholds_tried[0]
    selected_threshold = float(exploration_audit["selected_positive_residual_threshold"])
    local_rank_probe_count = int(exploration_audit.get("local_rank_probe_count", 0))
    blindspot_probe_count = int(exploration_audit["generated_explore_count"])
    fallback_explore_count = int(exploration_audit["fallback_explore_count"])
    if math.isclose(selected_threshold, default_threshold):
        threshold_note = f"none (kept default {default_threshold:.1f})"
    else:
        threshold_note = f"relaxed from {default_threshold:.1f} to {selected_threshold:.1f}"
    blindspot_stage_range = blindspot_audit["blindspot_stage_range"]
    generated_anchor_stage_range = exploration_audit.get("generated_anchor_stage_range", [])
    generated_anchor_stage_counts = exploration_audit.get("generated_anchor_stage_counts", {})

    lines = [
        "=" * 80,
        "CryoMN Next Formulations",
        "=" * 80,
        f"Target stage: {format_stage_label(active_stage.stage, active_stage.iteration_dir)}",
        f"Residual feedback stage: {format_stage_label(previous_stage.stage, previous_stage.iteration_dir)}",
        f"Blind-spot signal stages: {blindspot_stage_range[0]} to {blindspot_stage_range[1]}",
        f"Output split: {EXPLOIT_COUNT} exploitation + {EXPLORE_COUNT} exploration",
        "Positive residual thresholds tried: " + ", ".join(f"{threshold:.1f}" for threshold in thresholds_tried),
        f"Selected positive residual threshold: {selected_threshold:.1f}",
        f"Residual threshold relaxation: {threshold_note}",
        f"Local rank-resolution probes: {local_rank_probe_count}",
        f"Blind-spot probes: {blindspot_probe_count}",
        f"BO fallback exploration rows: {fallback_explore_count}",
        (
            "Generated anchor stage range: "
            f"{generated_anchor_stage_range[0]} to {generated_anchor_stage_range[1]}"
            if generated_anchor_stage_range
            else "Generated anchor stage range: none"
        ),
        (
            "Generated anchor stage counts: "
            + ", ".join(f"stage {stage}={count}" for stage, count in generated_anchor_stage_counts.items())
            if generated_anchor_stage_counts
            else "Generated anchor stage counts: none"
        ),
        "Blind-spot signal: historical positive residuals with recency/support adjustment",
        "Novelty score: full wet-lab history coverage, kept separate from blind-spot support",
        "Exploration anchors may come from any historical positive-residual stage",
        f"Top blind-spot features: {', '.join(top_features) if top_features else 'none detected'}",
        f"Top blind-spot pairs: {', '.join(top_pairs) if top_pairs else 'none detected'}",
        "",
        "Exploitation picks:",
    ]

    for row in exploit_rows:
        lines.extend(
            [
                f"- {row['formulation']}",
                f"  origin: {row['origin']} ({row['source_file']} rank {row['source_rank']})",
                f"  predicted viability: {row['predicted_viability']:.2f}% +/- {row['uncertainty']:.2f}%",
                f"  rationale: {row['rationale']}",
            ]
        )

    lines.append("")
    lines.append("Exploration / calibration picks:")
    for row in explore_rows:
        anchor_bits = []
        if row.get("anchor_stage") is not None:
            anchor_bits.append(f"stage {row['anchor_stage']}")
        if row.get("anchor_experiments"):
            anchor_bits.append(str(row["anchor_experiments"]))
        anchor_text = f" ({', '.join(anchor_bits)})" if anchor_bits else ""
        source_text = ""
        if row.get("source_file"):
            source_text = f" from {row['source_file']} rank {row['source_rank']}"
        lines.extend(
            [
                f"- {row['formulation']}",
                f"  origin: {row['origin']}{source_text}{anchor_text}",
                f"  predicted viability: {row['predicted_viability']:.2f}% +/- {row['uncertainty']:.2f}%",
                f"  rationale: {row['rationale']}",
            ]
        )

    lines.append("")
    lines.append("Recommended smaller batch subsets:")
    for recommendation in batch_recommendations:
        lines.append(
            f"- batch size {recommendation['batch_size']}:"
            f" score={recommendation['heuristic_score']},"
            f" exploit={recommendation['selected_counts']['exploit']},"
            f" local_rank={recommendation['selected_counts']['local_rank']},"
            f" blindspot={recommendation['selected_counts']['blindspot']},"
            f" mean_pred={recommendation['mean_predicted_viability']}"
        )
        for row in recommendation.get("rows", []):
            lines.extend(
                [
                    (
                        f"  {row['selection_order']}. {row['formulation']}"
                        f" [{row['recommendation_type']}/{row['origin']}]"
                    ),
                    (
                        f"     predicted viability: {float(row['predicted_viability']):.2f}%"
                        f" +/- {float(row['uncertainty']):.2f}%"
                    ),
                    f"     utility: {float(row['batch_utility']):.4f}",
                ]
            )
        lines.append("")
    return "\n".join(lines)


def preflight_report(
    active_stage: StageArtifacts,
    previous_stage: StageArtifacts,
    validation_df: pd.DataFrame,
    previous_batch: pd.DataFrame,
    candidate_paths: Sequence[Path],
    optimizer: BayesianOptimizer,
    exploration_audit: Dict[str, object],
    blindspot_audit: Dict[str, object],
) -> Dict:
    """Build the machine-readable input validation report."""
    return {
        "status": "passed",
        "target_stage": active_stage.stage,
        "target_iteration_dir": active_stage.iteration_dir,
        "feedback_stage": previous_stage.stage,
        "feedback_iteration_dir": previous_stage.iteration_dir,
        "validation_rows_total": int(len(validation_df)),
        "validation_rows_feedback_stage": int(len(previous_batch)),
        "candidate_files": [str(path.relative_to(PROJECT_ROOT)) for path in candidate_paths],
        "feature_count": len(active_stage.feature_names),
        "active_model_method": active_stage.metadata.get("model_method"),
        "feedback_model_method": previous_stage.metadata.get("model_method"),
        "active_is_composite": active_stage.is_composite_model,
        "feedback_is_composite": previous_stage.is_composite_model,
        "observed_context_rows": int(len(active_stage.observed_context)),
        "effective_max_ingredients": int(optimizer.effective_max_ingredients),
        "dmso_cap_percent": float(optimizer.config.max_dmso_percent),
        "feature_names": list(active_stage.feature_names),
        "positive_residual_thresholds_tried": list(exploration_audit["positive_residual_thresholds_tried"]),
        "selected_positive_residual_threshold": float(exploration_audit["selected_positive_residual_threshold"]),
        "anchor_count_at_selected_threshold": int(exploration_audit["anchor_count_at_selected_threshold"]),
        "generated_pool_rows_at_selected_threshold": int(
            exploration_audit["generated_pool_rows_at_selected_threshold"]
        ),
        "local_rank_anchor_count": int(exploration_audit.get("local_rank_anchor_count", 0)),
        "local_rank_probe_count": int(exploration_audit.get("local_rank_probe_count", 0)),
        "generated_explore_count": int(exploration_audit["generated_explore_count"]),
        "fallback_explore_count": int(exploration_audit["fallback_explore_count"]),
        "generated_anchor_stage_counts": dict(exploration_audit.get("generated_anchor_stage_counts", {})),
        "generated_anchor_stage_range": list(exploration_audit.get("generated_anchor_stage_range", [])),
        "generated_top_anchor_pairs": list(exploration_audit.get("generated_top_anchor_pairs", [])),
        "positive_residual_threshold_attempts": list(exploration_audit["positive_residual_threshold_attempts"]),
        "blindspot_signal_mode": str(blindspot_audit["blindspot_signal_mode"]),
        "blindspot_stage_range": list(blindspot_audit["blindspot_stage_range"]),
        "blindspot_recency_decay": float(blindspot_audit["blindspot_recency_decay"]),
        "blindspot_region_prediction_threshold": float(blindspot_audit["blindspot_region_prediction_threshold"]),
        "blindspot_region_weight": float(blindspot_audit["blindspot_region_weight"]),
        "blindspot_support_experiment_min": int(blindspot_audit["blindspot_support_experiment_min"]),
        "blindspot_experiment_shrink_offset": float(blindspot_audit["blindspot_experiment_shrink_offset"]),
        "blindspot_batch_shrink_offset": float(blindspot_audit["blindspot_batch_shrink_offset"]),
        "historical_positive_residual_rows": int(blindspot_audit["historical_positive_residual_rows"]),
        "top_positive_features": list(blindspot_audit["top_positive_features"]),
        "top_positive_pairs": list(blindspot_audit["top_positive_pairs"]),
    }


def to_output_rows(
    active_stage: StageArtifacts,
    exploit_rows: List[Dict],
    explore_rows: List[Dict],
) -> List[Dict]:
    """Convert scored selections into the final CSV row shape."""
    rows: List[Dict] = []
    for recommendation_type, bucket_rows in [("exploit", exploit_rows), ("explore", explore_rows)]:
        for rank, row in enumerate(bucket_rows, start=1):
            output_row = {
                "recommendation_type": recommendation_type,
                "bucket_rank": rank,
                "origin": row["origin"],
                "source_file": row.get("source_file", ""),
                "source_rank": row.get("source_rank"),
                "anchor_stage": row.get("anchor_stage"),
                "anchor_experiments": row.get("anchor_experiments", ""),
                "predicted_viability": float(row["predicted_viability"]),
                "uncertainty": float(row["uncertainty"]),
                "blindspot_score": float(row.get("blindspot_score", 0.0)),
                "novelty_score": float(row.get("novelty_score", 0.0)),
                "dmso_percent": float(row["dmso_percent"]),
                "n_ingredients": int(row["n_ingredients"]),
            }
            for feature_name in active_stage.feature_names:
                output_row[feature_name] = float(row.get(feature_name, 0.0))
            output_row["formulation"] = str(row["signature"])
            output_row["rationale"] = str(row["rationale"])
            rows.append(output_row)
    return rows


def score_batch_recommendation_rows(
    output_rows: List[Dict],
    active_stage: StageArtifacts,
) -> pd.DataFrame:
    """Attach heuristic utility scores used to recommend smaller test batches."""
    scored = pd.DataFrame(output_rows).copy()
    if scored.empty:
        return scored

    scored["recommendation_id"] = np.arange(1, len(scored) + 1)
    scored["chemistry_family"] = [
        chemistry_family(row, active_stage.feature_names) for _, row in scored.iterrows()
    ]
    scored["pred_norm"] = normalize(scored["predicted_viability"])
    scored["unc_norm"] = normalize(scored["uncertainty"])
    scored["blindspot_norm"] = normalize(scored["blindspot_score"].clip(lower=0.0))
    scored["novelty_norm"] = normalize(scored["novelty_score"])
    scored["confidence_norm"] = 1.0 - scored["unc_norm"]

    scored["subset_role"] = "blindspot"
    scored.loc[scored["recommendation_type"] == "exploit", "subset_role"] = "exploit"
    scored.loc[scored["origin"] == "local_rank_probe", "subset_role"] = "local_rank"
    scored.loc[scored["origin"] == "explore_fallback", "subset_role"] = "fallback"

    scored["local_anchor_key"] = scored.apply(
        lambda row: f"{row['source_file']}::{int(row['source_rank'])}"
        if row["origin"] == "local_rank_probe" and pd.notna(row.get("source_rank"))
        else "",
        axis=1,
    )

    row_scores = []
    for _, row in scored.iterrows():
        if row["recommendation_type"] == "exploit":
            score = (
                0.70 * row["pred_norm"]
                + 0.20 * row["confidence_norm"]
                + 0.10 * row["novelty_norm"]
            )
        elif row["origin"] == "local_rank_probe":
            score = (
                0.40 * row["pred_norm"]
                + 0.30 * row["unc_norm"]
                + 0.20 * row["blindspot_norm"]
                + 0.10 * row["novelty_norm"]
            )
        elif row["origin"] == "blindspot_probe":
            score = (
                0.20 * row["pred_norm"]
                + 0.35 * row["unc_norm"]
                + 0.35 * row["blindspot_norm"]
                + 0.10 * row["novelty_norm"]
            )
        else:
            score = (
                0.15 * row["pred_norm"]
                + 0.35 * row["unc_norm"]
                + 0.25 * row["blindspot_norm"]
                + 0.15 * row["novelty_norm"]
                - 0.05
            )
        row_scores.append(float(score))
    scored["batch_utility"] = row_scores
    return scored


def target_batch_role_counts(batch_size: int) -> Dict[str, int]:
    """Return the intended exploit/local/blind split for one smaller wet-lab batch."""
    exploit_target = int(round(batch_size * EXPLOIT_COUNT / TOTAL_COUNT))
    local_target = int(round(batch_size * LOCAL_RANK_PROBE_COUNT / TOTAL_COUNT))
    blind_target = batch_size - exploit_target - local_target
    return {
        "exploit": max(exploit_target, 0),
        "local_rank": max(local_target, 0),
        "blindspot": max(blind_target, 0),
    }


def build_batch_recommendations(
    output_rows: List[Dict],
    active_stage: StageArtifacts,
) -> Tuple[List[Dict[str, object]], pd.DataFrame, Dict[str, object]]:
    """Recommend the best subset for each feasible batch size from 6 to 12."""
    scored_rows = score_batch_recommendation_rows(output_rows, active_stage)
    if scored_rows.empty:
        return [], pd.DataFrame(), {}

    row_records = scored_rows.to_dict(orient="records")
    utility_values = scored_rows["batch_utility"].to_numpy(dtype=float)
    predicted_values = scored_rows["predicted_viability"].to_numpy(dtype=float)
    uncertainty_values = scored_rows["uncertainty"].to_numpy(dtype=float)
    role_values = scored_rows["subset_role"].astype(str).to_numpy()
    family_codes, _ = pd.factorize(scored_rows["chemistry_family"].astype(str), sort=False)
    local_anchor_labels = scored_rows["local_anchor_key"].astype(str)
    local_anchor_codes = np.full(len(scored_rows), -1, dtype=int)
    nonempty_local_anchor = local_anchor_labels != ""
    if nonempty_local_anchor.any():
        encoded_local_anchors, _ = pd.factorize(local_anchor_labels[nonempty_local_anchor], sort=False)
        local_anchor_codes[nonempty_local_anchor.to_numpy()] = encoded_local_anchors.astype(int)

    recommendations: List[Dict[str, object]] = []
    flattened_rows: List[Dict[str, object]] = []

    for batch_size in range(BATCH_RECOMMENDATION_MIN, BATCH_RECOMMENDATION_MAX + 1):
        best_score = -float("inf")
        best_tie_break = (-float("inf"), -float("inf"), -float("inf"))
        best_subset_indices: Tuple[int, ...] = ()
        target_counts = target_batch_role_counts(batch_size)

        for index_combo in combinations(range(len(row_records)), batch_size):
            combo_indices = np.fromiter(index_combo, dtype=int)
            combo_roles = role_values[combo_indices]
            exploit_count = int(np.count_nonzero(combo_roles == "exploit"))
            local_count = int(np.count_nonzero(combo_roles == "local_rank"))
            blind_count = int(np.count_nonzero((combo_roles == "blindspot") | (combo_roles == "fallback")))

            family_count = int(np.unique(family_codes[combo_indices]).size)
            local_codes = local_anchor_codes[combo_indices]
            local_codes = local_codes[local_codes >= 0]
            local_anchor_count = int(np.unique(local_codes).size)

            base_score = float(utility_values[combo_indices].sum())
            family_bonus = 0.10 * float(family_count)
            local_anchor_bonus = 0.08 * float(local_anchor_count)
            type_penalty = (
                0.45 * abs(exploit_count - target_counts["exploit"])
                + 0.35 * abs(local_count - target_counts["local_rank"])
                + 0.25 * abs(blind_count - target_counts["blindspot"])
            )
            subset_score = base_score + family_bonus + local_anchor_bonus - type_penalty
            tie_break = (
                float(predicted_values[combo_indices].mean()),
                float(family_count),
                -float(uncertainty_values[combo_indices].mean()),
            )
            if subset_score > best_score + 1e-12 or (
                math.isclose(subset_score, best_score, rel_tol=1e-12, abs_tol=1e-12)
                and tie_break > best_tie_break
            ):
                best_score = subset_score
                best_tie_break = tie_break
                best_subset_indices = index_combo

        best_subset_records = [row_records[idx] for idx in best_subset_indices]
        best_subset_df = pd.DataFrame(best_subset_records)
        role_counts = best_subset_df["subset_role"].value_counts().to_dict()
        recommendation = {
            "batch_size": int(batch_size),
            "heuristic_score": round_or_none(best_score),
            "target_counts": dict(target_counts),
            "selected_counts": {
                "exploit": int(role_counts.get("exploit", 0)),
                "local_rank": int(role_counts.get("local_rank", 0)),
                "blindspot": int(role_counts.get("blindspot", 0) + role_counts.get("fallback", 0)),
            },
            "mean_predicted_viability": round_or_none(best_subset_df["predicted_viability"].mean()),
            "mean_uncertainty": round_or_none(best_subset_df["uncertainty"].mean()),
            "unique_chemistry_families": int(best_subset_df["chemistry_family"].nunique()),
            "rows": [],
        }
        for selection_order, (_, row) in enumerate(best_subset_df.sort_values(["recommendation_type", "bucket_rank", "recommendation_id"]).iterrows(), start=1):
            recommendation["rows"].append(
                {
                    "selection_order": int(selection_order),
                    "recommendation_id": int(row["recommendation_id"]),
                    "recommendation_type": str(row["recommendation_type"]),
                    "origin": str(row["origin"]),
                    "bucket_rank": int(row["bucket_rank"]),
                    "predicted_viability": round_or_none(row["predicted_viability"]),
                    "uncertainty": round_or_none(row["uncertainty"]),
                    "batch_utility": round_or_none(row["batch_utility"]),
                    "formulation": str(row["formulation"]),
                }
            )
            flattened_rows.append(
                {
                    "batch_size": int(batch_size),
                    "selection_order": int(selection_order),
                    "recommendation_id": int(row["recommendation_id"]),
                    "recommendation_type": str(row["recommendation_type"]),
                    "origin": str(row["origin"]),
                    "bucket_rank": int(row["bucket_rank"]),
                    "predicted_viability": float(row["predicted_viability"]),
                    "uncertainty": float(row["uncertainty"]),
                    "batch_utility": float(row["batch_utility"]),
                    "formulation": str(row["formulation"]),
                }
            )
        recommendations.append(recommendation)

    scoring_config = {
        "batch_size_range": [BATCH_RECOMMENDATION_MIN, BATCH_RECOMMENDATION_MAX],
        "target_mix_basis": {
            "exploit_count": EXPLOIT_COUNT,
            "local_rank_probe_count": LOCAL_RANK_PROBE_COUNT,
            "blindspot_probe_count": BLINDSPOT_PROBE_COUNT,
            "total_count": TOTAL_COUNT,
        },
        "row_score_weights": {
            "exploit": {"pred_norm": 0.70, "confidence_norm": 0.20, "novelty_norm": 0.10},
            "local_rank_probe": {"pred_norm": 0.40, "unc_norm": 0.30, "blindspot_norm": 0.20, "novelty_norm": 0.10},
            "blindspot_probe": {"pred_norm": 0.20, "unc_norm": 0.35, "blindspot_norm": 0.35, "novelty_norm": 0.10},
            "explore_fallback": {"pred_norm": 0.15, "unc_norm": 0.35, "blindspot_norm": 0.25, "novelty_norm": 0.15, "penalty": -0.05},
        },
        "subset_score_adjustments": {
            "chemistry_family_bonus": 0.10,
            "local_anchor_bonus": 0.08,
            "type_penalty_weights": {"exploit": 0.45, "local_rank": 0.35, "blindspot": 0.25},
        },
    }
    return recommendations, pd.DataFrame(flattened_rows), scoring_config


def write_atomic_text(path: Path, content: str) -> None:
    """Write one text file atomically."""
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(content)
    os.replace(tmp_path, path)


def write_atomic_json(path: Path, payload: object) -> None:
    """Write one JSON file atomically."""
    write_atomic_text(path, json.dumps(payload, indent=2))


def write_atomic_csv(path: Path, df: pd.DataFrame) -> None:
    """Write one CSV file atomically."""
    tmp_path = path.with_name(path.name + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def ensure_output_paths(output_dir: Path, overwrite: bool) -> Dict[str, Path]:
    """Validate output file paths before writing."""
    paths = {
        "csv": output_dir / "next_formulations.csv",
        "summary": output_dir / "next_formulations_summary.txt",
        "metadata": output_dir / "next_formulations_metadata.json",
        "input_validation": output_dir / "input_validation.json",
        "batch_recommendations_json": output_dir / "batch_recommendations.json",
        "batch_recommendations_csv": output_dir / "batch_recommendations.csv",
    }
    if not overwrite:
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing:
            raise ValidationError(
                "Output files already exist. Re-run with --overwrite to replace them: "
                + ", ".join(existing)
            )
    return paths


def generate_next_formulations(stage_override: Optional[int], overwrite: bool) -> Dict[str, object]:
    """Run the full recommendation pipeline."""
    if not (MODELS_DIR / "model_metadata.json").exists():
        raise ValidationError("Missing active model metadata at models/model_metadata.json")

    active_metadata = json.loads((MODELS_DIR / "model_metadata.json").read_text())
    target_stage = int(stage_override if stage_override is not None else active_metadata.get("iteration"))
    if target_stage < 1:
        raise ValidationError("Target stage must be at least 1.")

    target_iteration_dir = None
    if stage_override is None:
        target_iteration_dir = active_metadata.get("iteration_dir")
        if active_metadata.get("iteration") != target_stage:
            raise ValidationError("Active metadata iteration is inconsistent with the requested target stage.")

    active_stage = load_stage_artifacts(target_stage, target_iteration_dir)
    validation_df = load_validation_df(active_stage.feature_names)
    completed_stages = sorted({int(stage) for stage in validation_df["stage"].unique()})
    last_completed_stage = completed_stages[-1]
    if last_completed_stage != target_stage - 1:
        raise ValidationError(
            f"Expected the latest completed validation stage to be {target_stage - 1}, "
            f"found {last_completed_stage}."
        )

    previous_stage = load_stage_artifacts(last_completed_stage, require_observed_context=False)
    if previous_stage.feature_names != active_stage.feature_names:
        raise ValidationError("Active and previous-stage feature spaces do not match.")

    previous_batch = compute_previous_stage_batch(validation_df, previous_stage)
    historical_residual_df = build_historical_residual_df(validation_df, active_stage, completed_stages)
    optimizer = build_bo_context(active_stage)
    candidate_pool, candidate_paths = load_bo_candidate_pool(active_stage, validation_df)
    tested_signatures = build_tested_signatures(validation_df, active_stage.feature_names)

    (
        feature_signal,
        pair_signal,
        wetlab_feature_counts,
        wetlab_pair_counts,
        _feature_details,
        _pair_details,
        blindspot_audit,
    ) = compute_blindspot_signals(
        historical_residual_df,
        validation_df,
        active_stage.feature_names,
        last_completed_stage,
    )

    exploit_rows = build_exploitation_selection(candidate_pool)
    if len(exploit_rows) != EXPLOIT_COUNT:
        raise ValidationError(f"Unable to select {EXPLOIT_COUNT} exploitation rows from active BO candidates.")

    exploit_signatures = {str(row["signature"]) for row in exploit_rows}
    local_rank_rows, local_rank_audit = build_local_rank_probe_rows(
        exploit_rows,
        active_stage,
        optimizer,
        feature_signal,
        pair_signal,
        wetlab_feature_counts,
        wetlab_pair_counts,
        tested_signatures,
    )
    local_rank_signatures = {str(row["signature"]) for row in local_rank_rows}
    generated_selection = choose_generated_exploration_rows(
        active_stage,
        historical_residual_df,
        optimizer,
        feature_signal,
        pair_signal,
        wetlab_feature_counts,
        wetlab_pair_counts,
        tested_signatures,
        exploit_signatures | local_rank_signatures,
    )
    fallback_pool = build_bo_exploration_fallback_pool(
        candidate_pool,
        active_stage,
        feature_signal,
        pair_signal,
        wetlab_feature_counts,
        wetlab_pair_counts,
    )

    explore_rows, fallback_explore_count = build_exploration_selection(
        local_rank_rows,
        generated_selection["selected_rows"],
        fallback_pool,
        exploit_signatures,
    )
    if len(explore_rows) != EXPLORE_COUNT:
        raise ValidationError(f"Unable to select {EXPLORE_COUNT} exploration rows.")
    exploration_audit = dict(generated_selection["audit"])
    exploration_audit["local_rank_anchor_count"] = int(local_rank_audit["local_rank_anchor_count"])
    exploration_audit["local_rank_probe_count"] = int(local_rank_audit["local_rank_probe_count"])
    exploration_audit["fallback_explore_count"] = int(fallback_explore_count)

    output_rows = to_output_rows(active_stage, exploit_rows, explore_rows)
    validate_output_rows(output_rows, active_stage, optimizer, tested_signatures)
    batch_recommendations, batch_recommendations_df, batch_recommendation_scoring = build_batch_recommendations(
        output_rows,
        active_stage,
    )

    output_dir = NEXT_FORMULATIONS_DIR / active_stage.iteration_dir
    output_paths = ensure_output_paths(output_dir, overwrite=overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    preflight = preflight_report(
        active_stage,
        previous_stage,
        validation_df,
        previous_batch,
        candidate_paths,
        optimizer,
        exploration_audit,
        blindspot_audit,
    )

    summary_text = build_summary_text(
        active_stage,
        previous_stage,
        [dict(row) for row in output_rows if row["recommendation_type"] == "exploit"],
        [dict(row) for row in output_rows if row["recommendation_type"] == "explore"],
        feature_signal,
        pair_signal,
        exploration_audit,
        blindspot_audit,
        batch_recommendations,
    )
    metadata = {
        "target_stage": active_stage.stage,
        "target_iteration_dir": active_stage.iteration_dir,
        "feedback_stage": previous_stage.stage,
        "feedback_iteration_dir": previous_stage.iteration_dir,
        "generation_seed": GENERATION_SEED,
        "positive_residual_thresholds_tried": list(exploration_audit["positive_residual_thresholds_tried"]),
        "selected_positive_residual_threshold": float(exploration_audit["selected_positive_residual_threshold"]),
        "anchor_count_at_selected_threshold": int(exploration_audit["anchor_count_at_selected_threshold"]),
        "generated_pool_rows_at_selected_threshold": int(
            exploration_audit["generated_pool_rows_at_selected_threshold"]
        ),
        "local_rank_anchor_count": int(exploration_audit.get("local_rank_anchor_count", 0)),
        "local_rank_probe_count": int(exploration_audit.get("local_rank_probe_count", 0)),
        "generated_explore_count": int(exploration_audit["generated_explore_count"]),
        "fallback_explore_count": int(exploration_audit["fallback_explore_count"]),
        "generated_anchor_stage_counts": dict(exploration_audit.get("generated_anchor_stage_counts", {})),
        "generated_anchor_stage_range": list(exploration_audit.get("generated_anchor_stage_range", [])),
        "generated_top_anchor_pairs": list(exploration_audit.get("generated_top_anchor_pairs", [])),
        "meaningful_actual_min": MEANINGFUL_ACTUAL_MIN,
        "min_exploration_prediction": MIN_EXPLORATION_PREDICTION,
        "exploit_count": EXPLOIT_COUNT,
        "explore_count": EXPLORE_COUNT,
        "effective_max_ingredients": int(optimizer.effective_max_ingredients),
        "dmso_cap_percent": float(optimizer.config.max_dmso_percent),
        "candidate_files": [str(path.relative_to(PROJECT_ROOT)) for path in candidate_paths],
        "used_exploration_fallback": bool(exploration_audit["fallback_explore_count"]),
        "blindspot_signal_mode": str(blindspot_audit["blindspot_signal_mode"]),
        "blindspot_stage_range": list(blindspot_audit["blindspot_stage_range"]),
        "blindspot_recency_decay": float(blindspot_audit["blindspot_recency_decay"]),
        "blindspot_region_prediction_threshold": float(blindspot_audit["blindspot_region_prediction_threshold"]),
        "blindspot_region_weight": float(blindspot_audit["blindspot_region_weight"]),
        "blindspot_support_experiment_min": int(blindspot_audit["blindspot_support_experiment_min"]),
        "blindspot_experiment_shrink_offset": float(blindspot_audit["blindspot_experiment_shrink_offset"]),
        "blindspot_batch_shrink_offset": float(blindspot_audit["blindspot_batch_shrink_offset"]),
        "historical_positive_residual_rows": int(blindspot_audit["historical_positive_residual_rows"]),
        "top_positive_features": list(blindspot_audit["top_positive_features"]),
        "top_positive_pairs": list(blindspot_audit["top_positive_pairs"]),
        "batch_recommendation_scoring": batch_recommendation_scoring,
        "batch_recommendations": batch_recommendations,
    }

    output_df = pd.DataFrame(output_rows)
    ordered_columns = [
        "recommendation_type",
        "bucket_rank",
        "origin",
        "source_file",
        "source_rank",
        "anchor_stage",
        "anchor_experiments",
        "predicted_viability",
        "uncertainty",
        "blindspot_score",
        "novelty_score",
        "dmso_percent",
        "n_ingredients",
        *active_stage.feature_names,
        "formulation",
        "rationale",
    ]
    output_df = output_df.reindex(columns=ordered_columns)

    write_atomic_csv(output_paths["csv"], output_df)
    write_atomic_text(output_paths["summary"], summary_text)
    write_atomic_json(output_paths["metadata"], metadata)
    write_atomic_json(output_paths["input_validation"], preflight)
    write_atomic_json(output_paths["batch_recommendations_json"], batch_recommendations)
    write_atomic_csv(output_paths["batch_recommendations_csv"], batch_recommendations_df)

    return {
        "output_dir": str(output_dir),
        "preflight": preflight,
        "metadata": metadata,
        "output_df": output_df,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help="Optional explicit iteration stage to target. Defaults to the active iteration.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files under results/next_formulations/<iteration_tag>/",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    result = generate_next_formulations(stage_override=args.stage, overwrite=args.overwrite)
    output_dir = result["output_dir"]
    metadata = result["metadata"]
    print("Input validation passed.")
    print(f"Target stage: iteration {metadata['target_stage']} ({metadata['target_iteration_dir']})")
    print(f"Feedback stage: iteration {metadata['feedback_stage']} ({metadata['feedback_iteration_dir']})")
    print(f"Wrote next formulations to: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
