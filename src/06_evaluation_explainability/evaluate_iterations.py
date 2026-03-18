#!/usr/bin/env python3
"""
Evaluate saved candidate/result sets against the wet-lab batches they produced.

Workflow encoded here:
1. Literature-only model -> result files without an iteration suffix -> EXP101-EXP306
2. Iteration 1 model -> iteration_1 result files -> EXP1101-EXP1206
3. Iteration 2 model -> iteration_2 result files -> EXP2101-EXP2106
4. Iteration 3 model -> iteration_3 result files -> pending / future EXP31xx...

The script therefore scores each stage against the specific validation batch
that was chosen from that stage's frozen rankings. Refreshed later rankings do
not overwrite the evidence for earlier stages.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "cryomn-matplotlib"))
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LITERATURE_ONLY_DIR = os.path.join(MODELS_DIR, "literature_only")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
VALIDATION_PATH = os.path.join(PROJECT_ROOT, "data", "validation", "validation_results.csv")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "evaluation")
PLOT_PATH = os.path.join(OUTPUT_DIR, "stage_performance.png")
HELPER_DIR = os.path.join(PROJECT_ROOT, "src", "helper")
VALIDATION_LOOP_DIR = os.path.join(PROJECT_ROOT, "src", "04_validation_loop")


if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if HELPER_DIR not in sys.path:
    sys.path.insert(0, HELPER_DIR)
if VALIDATION_LOOP_DIR not in sys.path:
    sys.path.insert(0, VALIDATION_LOOP_DIR)

from formulation_formatting import (  # noqa: E402
    format_formulation,
    normalize_formulation_dataframe,
)
from update_model_weighted_prior import CompositeGP  # noqa: F401,E402


@dataclass
class StageRecord:
    stage: int
    label: str
    iteration_dir: Optional[str]
    metadata_path: Optional[str]
    timestamp: Optional[datetime]
    model_method: str
    is_composite_model: bool
    feature_names: List[str]
    model_loader: str


EXPERIMENT_ID_PATTERN = re.compile(r"(\d+)")
CANDIDATE_FILE_PATTERN = re.compile(
    r"^(?P<basename>(?:bo_)?candidates_(?:general|dmso_free))"
    r"(?:_(?P<tag>iteration_(?P<stage>\d+)(?:_[A-Za-z0-9_]+)?))?\.csv$"
)


def parse_timestamp(value: str) -> datetime:
    """Parse ISO timestamps stored in metadata/history."""
    return datetime.fromisoformat(value)


def stage_from_iteration_dir(iteration_dir: str) -> Optional[int]:
    """Extract the stage number from an iteration directory name."""
    match = re.search(r"iteration_(\d+)", str(iteration_dir))
    if not match:
        return None
    return int(match.group(1))


def parse_validation_dates(series: pd.Series) -> pd.Series:
    """Parse wet-lab dates using the repository's month/day/year format."""
    parsed = pd.to_datetime(series, format="%m/%d/%y", errors="coerce")
    if parsed.isna().any():
        fallback_mask = parsed.isna()
        parsed.loc[fallback_mask] = pd.to_datetime(
            series.loc[fallback_mask], errors="coerce"
        )
    return parsed


def round_or_none(value: float, digits: int = 4) -> Optional[float]:
    """Return a JSON-safe rounded float."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def load_validation_df() -> pd.DataFrame:
    """Load measured wet-lab rows with parsed dates."""
    validation_df = pd.read_csv(VALIDATION_PATH)
    validation_df = validation_df[validation_df["viability_measured"].notna()].copy()
    validation_df["parsed_date"] = parse_validation_dates(validation_df["experiment_date"])
    validation_df = validation_df[validation_df["parsed_date"].notna()].copy()
    validation_df["parsed_date"] = validation_df["parsed_date"].dt.normalize()
    return validation_df


def discover_iteration_checkpoints() -> List[StageRecord]:
    """Collect saved iteration checkpoints with metadata."""
    records: List[StageRecord] = []

    if not os.path.isdir(MODELS_DIR):
        return records

    for entry in sorted(os.listdir(MODELS_DIR)):
        if not entry.startswith("iteration_"):
            continue
        metadata_path = os.path.join(MODELS_DIR, entry, "model_metadata.json")
        if not os.path.exists(metadata_path):
            continue
        with open(metadata_path, "r") as handle:
            metadata = json.load(handle)
        timestamp_raw = metadata.get("updated_at") or metadata.get("trained_at")
        if not timestamp_raw:
            continue
        stage = metadata.get("iteration")
        if stage is None:
            stage = stage_from_iteration_dir(entry)
        stage = int(stage) if stage is not None else -1
        records.append(
            StageRecord(
                stage=stage,
                label=f"iteration_{stage}",
                iteration_dir=entry,
                metadata_path=metadata_path,
                timestamp=parse_timestamp(timestamp_raw),
                model_method=str(metadata.get("model_method") or metadata.get("weighting_method") or "unknown"),
                is_composite_model=bool(metadata.get("is_composite_model", False)),
                feature_names=list(metadata["feature_names"]),
                model_loader="composite" if bool(metadata.get("is_composite_model", False)) else "standard",
            )
        )

    records.sort(key=lambda record: (record.stage, record.timestamp or datetime.min))
    return records


def build_stage_records() -> List[StageRecord]:
    """Build literature baseline plus saved iteration checkpoints."""
    iteration_records = discover_iteration_checkpoints()
    if not iteration_records:
        return []

    if os.path.exists(os.path.join(LITERATURE_ONLY_DIR, "model_metadata.json")):
        with open(os.path.join(LITERATURE_ONLY_DIR, "model_metadata.json"), "r") as handle:
            metadata = json.load(handle)
        literature_stage = StageRecord(
            stage=0,
            label="literature_only",
            iteration_dir="literature_only",
            metadata_path=os.path.join(LITERATURE_ONLY_DIR, "model_metadata.json"),
            timestamp=parse_timestamp(
                metadata.get("updated_at") or metadata.get("trained_at")
            ),
            model_method="literature_only",
            is_composite_model=False,
            feature_names=list(metadata["feature_names"]),
            model_loader="standard",
        )
    else:
        first_iteration = iteration_records[0]
        literature_stage = StageRecord(
            stage=0,
            label="literature_only",
            iteration_dir=first_iteration.iteration_dir,
            metadata_path=first_iteration.metadata_path,
            timestamp=first_iteration.timestamp,
            model_method="literature_only",
            is_composite_model=False,
            feature_names=first_iteration.feature_names,
            model_loader="literature_component",
        )
    return [literature_stage] + iteration_records


def load_model(stage_record: StageRecord):
    """Load one saved checkpoint."""
    if not stage_record.iteration_dir:
        raise FileNotFoundError(f"No model directory configured for stage {stage_record.label}")

    model_dir = os.path.join(MODELS_DIR, stage_record.iteration_dir)
    if stage_record.model_loader == "composite":
        with open(os.path.join(model_dir, "composite_model.pkl"), "rb") as handle:
            return pickle.load(handle), None
    with open(os.path.join(model_dir, "gp_model.pkl"), "rb") as handle:
        gp = pickle.load(handle)
    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as handle:
        scaler = pickle.load(handle)
    return gp, scaler


def predict(model, scaler, X: np.ndarray, is_composite_model: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Predict mean and standard deviation for one feature matrix."""
    if is_composite_model:
        return model.predict(X, return_std=True)
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled, return_std=True)


def evaluate_predictions(
    model,
    scaler,
    feature_names: Sequence[str],
    eval_df: pd.DataFrame,
    is_composite_model: bool,
) -> Dict[str, Optional[float]]:
    """Compute predictive metrics for one held-out wet-lab slice."""
    if eval_df.empty:
        return {
            "n_rows": 0,
            "rmse": None,
            "mae": None,
            "r2": None,
            "spearman_rho": None,
            "kendall_tau": None,
            "mean_uncertainty": None,
            "coverage_1sigma": None,
            "coverage_2sigma": None,
            "hit_rate_ge_50": None,
            "hit_rate_ge_70": None,
        }

    X = eval_df.reindex(columns=feature_names, fill_value=0.0).fillna(0.0).to_numpy(float)
    y = eval_df["viability_measured"].to_numpy(float)
    pred_mean, pred_std = predict(model, scaler, X, is_composite_model)

    rmse = float(np.sqrt(np.mean((y - pred_mean) ** 2)))
    mae = float(np.mean(np.abs(y - pred_mean)))
    ss_res = float(np.sum((y - pred_mean) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = None if ss_tot == 0 else float(1 - ss_res / ss_tot)
    spearman_value = spearmanr(y, pred_mean).statistic if len(y) > 1 else np.nan
    kendall_value = kendalltau(y, pred_mean).statistic if len(y) > 1 else np.nan

    return {
        "n_rows": int(len(eval_df)),
        "rmse": round_or_none(rmse),
        "mae": round_or_none(mae),
        "r2": round_or_none(r2),
        "spearman_rho": round_or_none(spearman_value),
        "kendall_tau": round_or_none(kendall_value),
        "mean_uncertainty": round_or_none(np.mean(pred_std)),
        "coverage_1sigma": round_or_none(np.mean(np.abs(y - pred_mean) <= pred_std)),
        "coverage_2sigma": round_or_none(np.mean(np.abs(y - pred_mean) <= 2 * pred_std)),
        "hit_rate_ge_50": round_or_none(np.mean((pred_mean >= 50.0) == (y >= 50.0))),
        "hit_rate_ge_70": round_or_none(np.mean((pred_mean >= 70.0) == (y >= 70.0))),
    }


def stage_from_experiment_id(experiment_id: str) -> Optional[int]:
    """Map EXP IDs to experiment stages used in this project."""
    match = EXPERIMENT_ID_PATTERN.search(str(experiment_id))
    if not match:
        return None
    value = int(match.group(1))
    if value < 1000:
        return 0
    return value // 1000


def candidate_files_for_stage(stage: int) -> List[str]:
    """Return candidate CSVs saved for one modeling stage."""
    candidates: List[str] = []
    for entry in sorted(os.listdir(RESULTS_DIR)):
        match = CANDIDATE_FILE_PATTERN.match(entry)
        if not match:
            continue
        file_stage = int(match.group("stage")) if match.group("stage") is not None else 0
        if file_stage == stage:
            candidates.append(os.path.join(RESULTS_DIR, entry))
    return candidates


def build_signature_lookup(validation_df: pd.DataFrame, feature_names: Sequence[str]) -> Dict[str, List[dict]]:
    """Map formulation signatures to later wet-lab outcomes."""
    lookup: Dict[str, List[dict]] = {}
    for _, row in validation_df.iterrows():
        signature = format_formulation(row, feature_names)
        record = {
            "experiment_id": str(row.get("experiment_id", "")),
            "experiment_date": str(row.get("experiment_date", "")),
            "viability_measured": float(row["viability_measured"]),
        }
        lookup.setdefault(signature, []).append(record)
    return lookup


def align_candidate_df(candidate_df: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    """Fill in missing feature columns before signature generation."""
    aligned = candidate_df.copy()
    for feature_name in feature_names:
        if feature_name not in aligned.columns:
            aligned[feature_name] = 0.0
        aligned[feature_name] = pd.to_numeric(aligned[feature_name], errors="coerce").fillna(0.0)
    return normalize_formulation_dataframe(aligned, feature_names)


def rescore_candidate_df(
    candidate_df: pd.DataFrame,
    feature_names: Sequence[str],
    model,
    scaler,
    is_composite_model: bool,
) -> pd.DataFrame:
    """Normalize candidate rows, recompute scores, and derive an effective rank."""
    rescored = align_candidate_df(candidate_df, feature_names).copy()
    if "rank" in rescored.columns:
        source_rank = pd.to_numeric(rescored["rank"], errors="coerce")
        source_rank = source_rank.where(source_rank.notna(), pd.Series(range(1, len(rescored) + 1)))
    else:
        source_rank = pd.Series(range(1, len(rescored) + 1), index=rescored.index, dtype=float)
    rescored["source_rank"] = source_rank.astype(int)

    X = rescored.reindex(columns=feature_names, fill_value=0.0).fillna(0.0).to_numpy(float)
    pred_mean, pred_std = predict(model, scaler, X, is_composite_model)
    rescored["predicted_viability"] = pred_mean
    rescored["uncertainty"] = pred_std
    rescored = rescored.sort_values(
        ["predicted_viability", "source_rank"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    rescored["effective_rank"] = np.arange(1, len(rescored) + 1)
    return rescored


def summarize_candidate_hits(
    candidate_path: str,
    future_validation_df: pd.DataFrame,
    feature_names: Sequence[str],
    model,
    scaler,
    is_composite_model: bool,
) -> Dict[str, object]:
    """Evaluate one frozen candidate file against later wet-lab matches."""
    candidate_df = pd.read_csv(candidate_path)
    candidate_df = rescore_candidate_df(
        candidate_df,
        feature_names,
        model,
        scaler,
        is_composite_model,
    ).fillna(0.0)
    lookup = build_signature_lookup(future_validation_df, feature_names)

    matched_rows: List[dict] = []
    for _, row in candidate_df.iterrows():
        signature = format_formulation(row, feature_names)
        tested_matches = lookup.get(signature, [])
        if not tested_matches:
            continue
        for match in tested_matches:
            matched_rows.append(
                {
                    "rank": int(row["effective_rank"]),
                    "source_rank": int(row["source_rank"]),
                    "effective_rank": int(row["effective_rank"]),
                    "predicted_viability": float(row["predicted_viability"]),
                    "uncertainty": float(row["uncertainty"]),
                    "experiment_id": match["experiment_id"],
                    "experiment_date": match["experiment_date"],
                    "actual_viability": match["viability_measured"],
                    "signature": signature,
                }
            )

    matched_df = pd.DataFrame(matched_rows)
    summary: Dict[str, object] = {
        "file": os.path.basename(candidate_path),
        "n_candidates": int(len(candidate_df)),
        "n_tested_later": int(len(matched_df)),
        "tested_examples": [],
        "top_k": {},
    }

    for k in (3, 5, 10):
        subset = matched_df[matched_df["effective_rank"] <= k] if not matched_df.empty else matched_df
        summary["top_k"][f"top_{k}"] = {
            "tested_count": int(len(subset)),
            "tested_fraction": round_or_none(len(subset) / min(k, len(candidate_df))) if len(candidate_df) else None,
            "mean_actual_viability": round_or_none(subset["actual_viability"].mean()) if len(subset) else None,
            "best_actual_viability": round_or_none(subset["actual_viability"].max()) if len(subset) else None,
            "hit_rate_ge_50": round_or_none((subset["actual_viability"] >= 50.0).mean()) if len(subset) else None,
            "hit_rate_ge_70": round_or_none((subset["actual_viability"] >= 70.0).mean()) if len(subset) else None,
        }

    if not matched_df.empty:
        preview = matched_df.sort_values(["effective_rank", "experiment_date", "experiment_id"]).head(10)
        summary["tested_examples"] = preview.to_dict(orient="records")

    return summary


def validation_batch_for_stage(validation_df: pd.DataFrame, stage: int) -> pd.DataFrame:
    """Return validation rows belonging to one experimental stage."""
    stage_series = validation_df["experiment_id"].map(stage_from_experiment_id)
    return validation_df[stage_series == stage].copy()


def evaluate_stage(stage_record: StageRecord, validation_df: pd.DataFrame) -> Dict[str, object]:
    """Evaluate one modeling stage against its corresponding validation batch."""
    model, scaler = load_model(stage_record)
    batch_df = validation_batch_for_stage(validation_df, stage_record.stage)
    candidate_summaries = [
        summarize_candidate_hits(
            path,
            batch_df,
            stage_record.feature_names,
            model,
            scaler,
            stage_record.is_composite_model,
        )
        for path in candidate_files_for_stage(stage_record.stage)
    ]

    return {
        "stage": stage_record.stage,
        "label": stage_record.label,
        "iteration_dir": stage_record.iteration_dir,
        "timestamp": None if stage_record.timestamp is None else stage_record.timestamp.isoformat(),
        "model_method": stage_record.model_method,
        "is_composite_model": stage_record.is_composite_model,
        "batch_dates": sorted({str(value.date()) for value in batch_df["parsed_date"].dropna().tolist()}),
        "batch_rows": int(len(batch_df)),
        "batch_metrics": evaluate_predictions(
            model,
            scaler,
            stage_record.feature_names,
            batch_df,
            stage_record.is_composite_model,
        ),
        "candidate_evaluation": candidate_summaries,
    }


def print_summary(results: Sequence[Dict[str, object]]):
    """Render a concise human-readable report."""
    print("=" * 80)
    print("CryoMN Stage-Based Evaluation")
    print("=" * 80)
    print("Each modeling stage is scored against the wet-lab batch chosen from that stage.")
    print("")

    for result in results:
        print(f"{result['label']}  [{result['model_method']}]")
        if result["timestamp"]:
            print(f"  Model timestamp: {result['timestamp']}")
        print(f"  Validation batch dates: {', '.join(result['batch_dates']) if result['batch_dates'] else 'N/A'}")

        batch_metrics = result["batch_metrics"]
        if batch_metrics["n_rows"]:
            print(
                "  Batch metrics:"
                f" rows={batch_metrics['n_rows']},"
                f" RMSE={batch_metrics['rmse']},"
                f" Spearman={batch_metrics['spearman_rho']},"
                f" hit@50={batch_metrics['hit_rate_ge_50']},"
                f" mean_std={batch_metrics['mean_uncertainty']}"
            )
        else:
            print("  Batch metrics: no validation rows found for this stage")

        if result["candidate_evaluation"]:
            print("  Candidate rank cross-reference:")
            for candidate_summary in result["candidate_evaluation"]:
                top10 = candidate_summary["top_k"]["top_10"]
                print(
                    f"    {candidate_summary['file']}:"
                    f" tested_in_batch={candidate_summary['n_tested_later']},"
                    f" top10_tested={top10['tested_count']},"
                    f" top10_hit@50={top10['hit_rate_ge_50']},"
                    f" top10_best={top10['best_actual_viability']}"
                )
        else:
            print("  Candidate rank cross-reference: no candidate CSVs found")

        print("")


def write_outputs(results: Sequence[Dict[str, object]]):
    """Persist evaluation artifacts for later comparison."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_path = os.path.join(OUTPUT_DIR, "iteration_prospective_summary.json")
    with open(json_path, "w") as handle:
        json.dump(list(results), handle, indent=2)

    rows: List[dict] = []
    for result in results:
        batch_metrics = result["batch_metrics"]
        rows.append(
            {
                "stage": result["stage"],
                "label": result["label"],
                "iteration_dir": result["iteration_dir"],
                "timestamp": result["timestamp"],
                "model_method": result["model_method"],
                "batch_dates": ",".join(result["batch_dates"]),
                "batch_rows": batch_metrics["n_rows"],
                "batch_rmse": batch_metrics["rmse"],
                "batch_spearman": batch_metrics["spearman_rho"],
                "batch_hit_rate_ge_50": batch_metrics["hit_rate_ge_50"],
            }
        )

    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "iteration_prospective_metrics.csv"),
        index=False,
    )


def write_performance_plot(results: Sequence[Dict[str, object]]):
    """Save a compact chart of stage-level evaluation metrics."""
    stage_results = [
        result
        for result in results
        if result["batch_metrics"]["n_rows"]
    ]
    if not stage_results:
        return

    labels = [result["label"].replace("_", "\n") for result in stage_results]
    rmse = [result["batch_metrics"]["rmse"] for result in stage_results]
    spearman = [result["batch_metrics"]["spearman_rho"] for result in stage_results]
    hit50 = [result["batch_metrics"]["hit_rate_ge_50"] for result in stage_results]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    colors = ["#375E97", "#FB6542", "#3F681C"]

    metric_specs = [
        ("Batch RMSE", rmse, colors[0], "Lower is better"),
        ("Spearman", spearman, colors[1], "Higher is better"),
        ("Hit Rate @ 50%", hit50, colors[2], "Higher is better"),
    ]

    for ax, (title, values, color, subtitle) in zip(axes, metric_specs):
        bars = ax.bar(labels, values, color=color, width=0.65)
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)
        if title == "Spearman":
            ax.axhline(0.0, color="#444444", linewidth=1, linestyle="--", alpha=0.7)
            ax.set_ylim(min(-0.6, min(values) - 0.1), max(0.3, max(values) + 0.1))
        elif title == "Hit Rate @ 50%":
            ax.set_ylim(0.0, 1.05)
        else:
            ax.set_ylim(0.0, max(values) * 1.2)
        ax.tick_params(axis="x", labelrotation=0)
        for bar, value in zip(bars, values):
            offset = 0.02 if title != "Batch RMSE" else max(values) * 0.03
            y = value + offset if value >= 0 else value - 0.08
            va = "bottom" if value >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{value:.2f}",
                ha="center",
                va=va,
                fontsize=9,
            )

    fig.suptitle("CryoMN Stage-Based Model Evaluation", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    """Run stage-based evaluation."""
    validation_df = load_validation_df()
    stage_records = build_stage_records()
    if not stage_records:
        raise FileNotFoundError(f"No iteration metadata found under {MODELS_DIR}")

    results = [evaluate_stage(record, validation_df) for record in stage_records]
    print_summary(results)
    write_outputs(results)
    write_performance_plot(results)


if __name__ == "__main__":
    main()
