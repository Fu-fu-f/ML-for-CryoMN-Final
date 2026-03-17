#!/usr/bin/env python3
"""
Generate the next wet-lab formulations with a fixed 5/5 exploit-explore split.

The script is intentionally strict:
- validate required inputs up front
- fail before writing if anything is missing or inconsistent
- generate exploration probes from residual blind spots rather than only from
  saved candidate CSVs
- validate the final 10 formulations before writing any outputs
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

for path in [PROJECT_ROOT, VALIDATION_LOOP_DIR, BO_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from filter_tested_candidates import format_formulation  # noqa: E402
from update_model_weighted_prior import CompositeGP  # noqa: F401,E402
from bo_optimizer import BOConfig, BayesianOptimizer  # noqa: E402


EXPERIMENT_ID_PATTERN = re.compile(r"(\d+)")
ITERATION_DIR_PATTERN = re.compile(r"^iteration_(\d+)(?:_[A-Za-z0-9_]+)?$")
FEATURE_THRESHOLD = 1e-6
GENERATION_SEED = 42
EXPLOIT_COUNT = 5
EXPLORE_COUNT = 5
TOTAL_COUNT = EXPLOIT_COUNT + EXPLORE_COUNT
MIN_POSITIVE_RESIDUAL = 10.0
MEANINGFUL_ACTUAL_MIN = 50.0
MIN_EXPLORATION_PREDICTION = 30.0
EXPLOIT_FAMILY_LIMIT = 2
EXPLORE_FAMILY_LIMIT = 1
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
    names: List[str] = []
    for feature_name in feature_names:
        value = row.get(feature_name, 0.0)
        if pd.isna(value):
            continue
        if abs(float(value)) > FEATURE_THRESHOLD:
            names.append(feature_name)
    return names


def top_features_by_magnitude(row: pd.Series, feature_names: Sequence[str], limit: int = 2) -> List[str]:
    """Return the largest active features in descending magnitude."""
    ranked: List[Tuple[float, str]] = []
    for feature_name in feature_names:
        value = row.get(feature_name, 0.0)
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


def load_stage_artifacts(stage: int, preferred_iteration_dir: Optional[str] = None) -> StageArtifacts:
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
    if not observed_context_path.exists():
        raise ValidationError(f"Missing observed context artifact: {observed_context_path}")
    observed_context = pd.read_csv(observed_context_path)
    missing_observed = [name for name in feature_names if name not in observed_context.columns]
    if missing_observed:
        raise ValidationError(
            f"{observed_context_path} is missing feature columns: {', '.join(missing_observed)}"
        )
    if "viability_percent" not in observed_context.columns:
        raise ValidationError(f"{observed_context_path} is missing viability_percent.")

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
            record.update(
                {
                    "origin": "bo_candidate",
                    "source_file": path.name,
                    "source_rank": int(row["rank"]),
                    "source_kind": "bo",
                    "predicted_viability": float(row["predicted_viability"]),
                    "uncertainty": float(row["uncertainty"]),
                    "acquisition_value": round_or_none(row.get("acquisition_value")),
                    "dmso_percent": float(row.get("dmso_percent", 0.0)),
                    "n_ingredients": int(row.get("n_ingredients", len(active_features(row, active_stage.feature_names)))),
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


def compute_previous_stage_batch(
    validation_df: pd.DataFrame,
    previous_stage: StageArtifacts,
) -> pd.DataFrame:
    """Load the most recent completed batch and attach previous-stage residuals."""
    batch = validation_df[validation_df["stage"] == previous_stage.stage].copy()
    if batch.empty:
        raise ValidationError(f"No validation rows found for completed stage {previous_stage.stage}.")

    X = batch.reindex(columns=previous_stage.feature_names, fill_value=0.0).fillna(0.0).to_numpy(float)
    predicted, uncertainty = predict(
        previous_stage.model,
        previous_stage.scaler,
        X,
        previous_stage.is_composite_model,
    )
    batch["predicted_previous_stage"] = predicted
    batch["uncertainty_previous_stage"] = uncertainty
    batch["residual"] = batch["viability_measured"].astype(float) - batch["predicted_previous_stage"].astype(float)
    batch["signature"] = [
        format_formulation(row, previous_stage.feature_names) for _, row in batch.iterrows()
    ]
    return batch


def compute_blindspot_signals(
    previous_batch: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_names: Sequence[str],
) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float], Dict[str, int], Dict[Tuple[str, str], int]]:
    """Summarize positive residual blind spots at feature and pair level."""
    feature_signal = {name: 0.0 for name in feature_names}
    pair_signal: Dict[Tuple[str, str], float] = {}
    feature_counts = {name: 0 for name in feature_names}
    pair_counts: Dict[Tuple[str, str], int] = {}

    for _, row in previous_batch.iterrows():
        residual = max(float(row["residual"]), 0.0)
        if residual <= 0.0:
            continue
        active = active_features(row, feature_names)
        if not active:
            continue
        share = residual / len(active)
        for feature_name in active:
            feature_signal[feature_name] += share
            feature_counts[feature_name] += 1
        for pair in combinations(sorted(active), 2):
            pair_signal[pair] = pair_signal.get(pair, 0.0) + residual / max(1, len(active) - 1)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    for feature_name, count in feature_counts.items():
        if count:
            feature_signal[feature_name] /= count
    for pair, count in list(pair_counts.items()):
        if count:
            pair_signal[pair] /= count

    wetlab_feature_counts = {name: 0 for name in feature_names}
    wetlab_pair_counts: Dict[Tuple[str, str], int] = {}
    for _, row in validation_df.iterrows():
        active = active_features(row, feature_names)
        for feature_name in active:
            wetlab_feature_counts[feature_name] += 1
        for pair in combinations(sorted(active), 2):
            wetlab_pair_counts[pair] = wetlab_pair_counts.get(pair, 0) + 1

    return feature_signal, pair_signal, wetlab_feature_counts, wetlab_pair_counts


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
    record = {name: float(vector[idx]) for idx, name in enumerate(feature_names)}
    row = pd.Series(record)
    dmso_molar = float(record.get("dmso_M", 0.0))
    dmso_percent = dmso_molar * 78.13 / (1.10 * 10.0)
    record.update(
        {
            "origin": origin,
            "source_file": source_file or "",
            "source_rank": source_rank,
            "source_kind": "generated" if origin == "generated_probe" else "bo",
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
    x = optimizer._clip_to_bounds(vector.copy())
    x = sparsify_vector(x, feature_names, optimizer.effective_max_ingredients, protected_features)
    return optimizer._clip_to_bounds(x)


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
    previous_stage: StageArtifacts,
    previous_batch: pd.DataFrame,
    validation_df: pd.DataFrame,
    optimizer: BayesianOptimizer,
    feature_signal: Dict[str, float],
    pair_signal: Dict[Tuple[str, str], float],
    wetlab_feature_counts: Dict[str, int],
    wetlab_pair_counts: Dict[Tuple[str, str], int],
    tested_signatures: set[str],
) -> pd.DataFrame:
    """Generate calibration probes from positive-residual blind spots."""
    previous_batch = previous_batch.copy()
    previous_batch["actual_priority"] = previous_batch["viability_measured"].astype(float) >= MEANINGFUL_ACTUAL_MIN
    previous_batch["is_positive_anchor"] = previous_batch["residual"].astype(float) > MIN_POSITIVE_RESIDUAL

    positive_rows = previous_batch[previous_batch["is_positive_anchor"]].copy()
    if positive_rows.empty:
        return pd.DataFrame()

    anchor_rows = positive_rows.sort_values(
        ["actual_priority", "residual", "viability_measured"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    pair_to_rows: Dict[Tuple[str, str], List[pd.Series]] = {}
    for _, row in anchor_rows.iterrows():
        active = active_features(row, previous_stage.feature_names)
        for pair in combinations(sorted(active), 2):
            pair_to_rows.setdefault(pair, []).append(row)

    generated_records: List[Dict] = []
    seen_signatures: set[str] = set()
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
                    origin="generated_probe",
                    anchor_stage=previous_stage.stage,
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

        anchor = anchors[0]
        for scale in (0.75, 1.25):
            probe = build_scaled_probe(pair, anchor, active_stage.feature_names, optimizer, scale)
            if probe is None:
                continue
            record = vector_to_record(
                probe,
                active_stage.feature_names,
                origin="generated_probe",
                anchor_stage=previous_stage.stage,
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

    if not generated_records:
        return pd.DataFrame()

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
    return rank_generated_probes(generated)


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
    selected: List[Dict] = []
    family_counts: Dict[str, int] = {}
    disallowed_signatures = disallowed_signatures or set()

    ranked = scored.sort_values(
        [score_column, "predicted_viability", "uncertainty"],
        ascending=[False, False, True],
    )
    for _, row in ranked.iterrows():
        signature = str(row["signature"])
        if signature in disallowed_signatures:
            continue
        if float(row["predicted_viability"]) < min_predicted_viability:
            continue
        family = str(row["chemistry_family"])
        if family_counts.get(family, 0) >= family_limit:
            continue
        selected.append(row.to_dict())
        disallowed_signatures.add(signature)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) >= count:
            break
    return selected


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
    generated_pool: pd.DataFrame,
    fallback_pool: pd.DataFrame,
    already_selected_signatures: set[str],
) -> List[Dict]:
    """Select the final exploration rows, falling back to BO rows if needed."""
    selected = select_diverse_rows(
        generated_pool,
        count=EXPLORE_COUNT,
        score_column="exploration_score",
        family_limit=EXPLORE_FAMILY_LIMIT,
        min_predicted_viability=MIN_EXPLORATION_PREDICTION,
        disallowed_signatures=set(already_selected_signatures),
    )
    if len(selected) >= EXPLORE_COUNT:
        return selected

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
        fallback_rows = select_diverse_rows(
            fallback_pool,
            count=remaining,
            score_column="exploration_score",
            family_limit=EXPLORE_COUNT,
            min_predicted_viability=MIN_EXPLORATION_PREDICTION,
            disallowed_signatures=disallowed,
        )
    for row in fallback_rows:
        row["origin"] = "explore_fallback"
        if not row.get("rationale"):
            row["rationale"] = "High-uncertainty BO fallback in a blind-spot chemistry family."
    selected.extend(fallback_rows)
    return selected


def ensure_dmso_probe_present(explore_rows: List[Dict]) -> None:
    """Fail if exploration missed the known DMSO+sucrose blind spot in the current data."""
    for row in explore_rows:
        active = active_features(pd.Series(row), [name for name in row.keys() if is_feature_column(name)])
        if "dmso_M" in active and "sucrose_M" in active:
            return
    raise ValidationError(
        "Exploration selection did not include a DMSO+sucrose probe, which should be present "
        "given the current positive-residual blind spot."
    )


def validate_output_rows(
    rows: List[Dict],
    active_stage: StageArtifacts,
    optimizer: BayesianOptimizer,
    tested_signatures: set[str],
) -> None:
    """Validate the final 10 outputs before writing anything."""
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
        if row.get("origin") not in {"bo_candidate", "generated_probe", "explore_fallback"}:
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

    lines = [
        "=" * 80,
        "CryoMN Next Formulations",
        "=" * 80,
        f"Target stage: {format_stage_label(active_stage.stage, active_stage.iteration_dir)}",
        f"Residual feedback stage: {format_stage_label(previous_stage.stage, previous_stage.iteration_dir)}",
        f"Output split: {EXPLOIT_COUNT} exploitation + {EXPLORE_COUNT} exploration",
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
        if row["origin"] != "generated_probe":
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
    return "\n".join(lines)


def preflight_report(
    active_stage: StageArtifacts,
    previous_stage: StageArtifacts,
    validation_df: pd.DataFrame,
    previous_batch: pd.DataFrame,
    candidate_paths: Sequence[Path],
    optimizer: BayesianOptimizer,
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

    previous_stage = load_stage_artifacts(last_completed_stage)
    if previous_stage.feature_names != active_stage.feature_names:
        raise ValidationError("Active and previous-stage feature spaces do not match.")

    previous_batch = compute_previous_stage_batch(validation_df, previous_stage)
    optimizer = build_bo_context(active_stage)
    candidate_pool, candidate_paths = load_bo_candidate_pool(active_stage, validation_df)
    tested_signatures = build_tested_signatures(validation_df, active_stage.feature_names)

    feature_signal, pair_signal, wetlab_feature_counts, wetlab_pair_counts = compute_blindspot_signals(
        previous_batch,
        validation_df,
        active_stage.feature_names,
    )

    exploit_rows = build_exploitation_selection(candidate_pool)
    if len(exploit_rows) != EXPLOIT_COUNT:
        raise ValidationError(f"Unable to select {EXPLOIT_COUNT} exploitation rows from active BO candidates.")

    generated_pool = build_generated_probe_pool(
        active_stage,
        previous_stage,
        previous_batch,
        validation_df,
        optimizer,
        feature_signal,
        pair_signal,
        wetlab_feature_counts,
        wetlab_pair_counts,
        tested_signatures,
    )
    fallback_pool = build_bo_exploration_fallback_pool(
        candidate_pool,
        active_stage,
        feature_signal,
        pair_signal,
        wetlab_feature_counts,
        wetlab_pair_counts,
    )

    exploit_signatures = {str(row["signature"]) for row in exploit_rows}
    explore_rows = build_exploration_selection(generated_pool, fallback_pool, exploit_signatures)
    if len(explore_rows) != EXPLORE_COUNT:
        raise ValidationError(f"Unable to select {EXPLORE_COUNT} exploration rows.")
    ensure_dmso_probe_present(explore_rows)

    output_rows = to_output_rows(active_stage, exploit_rows, explore_rows)
    validate_output_rows(output_rows, active_stage, optimizer, tested_signatures)

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
    )

    summary_text = build_summary_text(
        active_stage,
        previous_stage,
        [dict(row) for row in output_rows if row["recommendation_type"] == "exploit"],
        [dict(row) for row in output_rows if row["recommendation_type"] == "explore"],
        feature_signal,
        pair_signal,
    )
    metadata = {
        "target_stage": active_stage.stage,
        "target_iteration_dir": active_stage.iteration_dir,
        "feedback_stage": previous_stage.stage,
        "feedback_iteration_dir": previous_stage.iteration_dir,
        "generation_seed": GENERATION_SEED,
        "min_positive_residual": MIN_POSITIVE_RESIDUAL,
        "meaningful_actual_min": MEANINGFUL_ACTUAL_MIN,
        "min_exploration_prediction": MIN_EXPLORATION_PREDICTION,
        "exploit_count": EXPLOIT_COUNT,
        "explore_count": EXPLORE_COUNT,
        "effective_max_ingredients": int(optimizer.effective_max_ingredients),
        "dmso_cap_percent": float(optimizer.config.max_dmso_percent),
        "candidate_files": [str(path.relative_to(PROJECT_ROOT)) for path in candidate_paths],
        "used_exploration_fallback": any(row["origin"] == "explore_fallback" for row in output_rows),
        "top_positive_features": [
            {"feature": feature_name, "score": round_or_none(score)}
            for feature_name, score in sorted(feature_signal.items(), key=lambda item: item[1], reverse=True)
            if score > 0.0
        ][:10],
        "top_positive_pairs": [
            {
                "pair": list(pair),
                "score": round_or_none(score),
            }
            for pair, score in sorted(pair_signal.items(), key=lambda item: item[1], reverse=True)
            if score > 0.0
        ][:10],
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
