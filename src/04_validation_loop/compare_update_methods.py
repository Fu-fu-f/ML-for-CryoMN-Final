#!/usr/bin/env python3
"""
Shadow comparison of available model-update methods using rolling retrospective stages.

This script never activates a model. It trains candidate methods only in-memory,
scores them on later completed stages, and writes recommendation artifacts under
results/model_comparison/.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "cryomn-matplotlib"))
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "model_comparison"
VALIDATION_PATH = PROJECT_ROOT / "data" / "validation" / "validation_results.csv"
PARSED_PATH = PROJECT_ROOT / "data" / "processed" / "parsed_formulations.csv"
HELPER_DIR = PROJECT_ROOT / "src" / "helper"

for path in [PROJECT_ROOT, SCRIPT_DIR, HELPER_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from iteration_metadata import PRIOR_MEAN_METHOD, STANDARD_METHOD, WEIGHTED_SIMPLE_METHOD  # noqa: E402
from update_model import train_standard_model  # noqa: E402
from update_model_weighted_prior import train_prior_mean_model  # noqa: E402
from update_model_weighted_simple import train_weighted_model  # noqa: E402


EXPERIMENT_ID_PATTERN = re.compile(r"(\d+)")
INCUMBENT_LABEL = "prior_mean_correction_alpha_wetlab_0.02"


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    method: str
    weight_multiplier: Optional[int] = None
    alpha_literature: Optional[float] = None
    alpha_wetlab: Optional[float] = None


def round_or_none(value: float, digits: int = 4) -> Optional[float]:
    """Return a JSON-safe rounded value."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def stage_from_experiment_id(experiment_id: str) -> Optional[int]:
    """Map EXP IDs to experiment stages used in this project."""
    match = EXPERIMENT_ID_PATTERN.search(str(experiment_id))
    if not match:
        return None
    value = int(match.group(1))
    if value < 1000:
        return 0
    return value // 1000


def parse_validation_dates(series: pd.Series) -> pd.Series:
    """Parse wet-lab dates using the repository's month/day/year format."""
    parsed = pd.to_datetime(series, format="%m/%d/%y", errors="coerce")
    if parsed.isna().any():
        fallback_mask = parsed.isna()
        parsed.loc[fallback_mask] = pd.to_datetime(series.loc[fallback_mask], errors="coerce")
    return parsed


def load_feature_names() -> List[str]:
    """Load active feature names from model metadata."""
    metadata_path = MODELS_DIR / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    return list(metadata["feature_names"])


def load_literature_data(feature_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Load literature training inputs aligned to the active feature set."""
    df = pd.read_csv(PARSED_PATH)
    df = df[df["viability_percent"] <= 100].copy()
    X = df.reindex(columns=feature_names, fill_value=0.0).fillna(0.0).to_numpy(float)
    y = df["viability_percent"].to_numpy(float)
    return X, y


def load_validation_df(feature_names: Sequence[str]) -> pd.DataFrame:
    """Load measured wet-lab rows with stage labels."""
    df = pd.read_csv(VALIDATION_PATH)
    df = df[df["viability_measured"].notna()].copy()
    df["parsed_date"] = parse_validation_dates(df["experiment_date"])
    df = df[df["parsed_date"].notna()].copy()
    df["parsed_date"] = df["parsed_date"].dt.normalize()
    df["stage"] = df["experiment_id"].map(stage_from_experiment_id)
    df = df[df["stage"].notna()].copy()
    for feature_name in feature_names:
        if feature_name not in df.columns:
            df[feature_name] = 0.0
        df[feature_name] = pd.to_numeric(df[feature_name], errors="coerce").fillna(0.0)
    return df


def predict(model, scaler, X: np.ndarray, is_composite_model: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Predict mean and standard deviation for one feature matrix."""
    if is_composite_model:
        return model.predict(X, return_std=True)
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled, return_std=True)


def compute_metrics(y: np.ndarray, pred_mean: np.ndarray, pred_std: np.ndarray) -> Dict[str, Optional[float]]:
    """Compute model-comparison metrics for one held-out stage."""
    residual = y - pred_mean
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = None if ss_tot == 0 else float(1 - ss_res / ss_tot)
    spearman_value = spearmanr(y, pred_mean).statistic if len(y) > 1 else np.nan
    kendall_value = kendalltau(y, pred_mean).statistic if len(y) > 1 else np.nan

    return {
        "n_rows": int(len(y)),
        "rmse": round_or_none(rmse),
        "mae": round_or_none(mae),
        "r2": round_or_none(r2),
        "spearman_rho": round_or_none(spearman_value),
        "kendall_tau": round_or_none(kendall_value),
        "mean_uncertainty": round_or_none(np.mean(pred_std)),
        "coverage_1sigma": round_or_none(np.mean(np.abs(residual) <= pred_std)),
        "coverage_2sigma": round_or_none(np.mean(np.abs(residual) <= 2 * pred_std)),
        "mean_signed_residual": round_or_none(np.mean(residual)),
        "mean_abs_residual": round_or_none(np.mean(np.abs(residual))),
    }


def candidate_specs() -> List[CandidateSpec]:
    """Return the fixed model-comparison grid."""
    specs = [CandidateSpec(label="standard", method=STANDARD_METHOD)]
    specs.extend(
        CandidateSpec(
            label=f"weighted_simple_x{multiplier}",
            method=WEIGHTED_SIMPLE_METHOD,
            weight_multiplier=multiplier,
        )
        for multiplier in (5, 10, 20)
    )
    specs.extend(
        CandidateSpec(
            label=f"prior_mean_correction_alpha_wetlab_{alpha:.2f}",
            method=PRIOR_MEAN_METHOD,
            alpha_literature=1.0,
            alpha_wetlab=alpha,
        )
        for alpha in (0.02, 0.05, 0.10)
    )
    return specs


def train_candidate(
    spec: CandidateSpec,
    X_orig: np.ndarray,
    y_orig: np.ndarray,
    X_train_wetlab: np.ndarray,
    y_train_wetlab: np.ndarray,
) -> Dict:
    """Train one candidate spec using the production training path."""
    if spec.method == STANDARD_METHOD:
        return train_standard_model(X_orig, y_orig, X_train_wetlab, y_train_wetlab)
    if spec.method == WEIGHTED_SIMPLE_METHOD:
        assert spec.weight_multiplier is not None
        return train_weighted_model(X_orig, y_orig, X_train_wetlab, y_train_wetlab, spec.weight_multiplier)
    assert spec.alpha_literature is not None and spec.alpha_wetlab is not None
    return train_prior_mean_model(
        X_orig,
        y_orig,
        X_train_wetlab,
        y_train_wetlab,
        alpha_literature=spec.alpha_literature,
        alpha_wetlab=spec.alpha_wetlab,
    )


def minmax_inverse(values: pd.Series) -> pd.Series:
    """Inverse min-max normalize a metric where lower is better."""
    minimum = float(values.min())
    maximum = float(values.max())
    if math.isclose(minimum, maximum):
        return pd.Series(np.full(len(values), 0.5), index=values.index)
    return 1.0 - ((values - minimum) / (maximum - minimum))


def minmax_forward(values: pd.Series) -> pd.Series:
    """Min-max normalize a metric where higher is better."""
    minimum = float(values.min())
    maximum = float(values.max())
    if math.isclose(minimum, maximum):
        return pd.Series(np.full(len(values), 0.5), index=values.index)
    return (values - minimum) / (maximum - minimum)


def calibration_component(coverage_1sigma: pd.Series, coverage_2sigma: pd.Series) -> pd.Series:
    """Score closeness to ideal 1σ and 2σ coverage."""
    cov1 = (1.0 - (coverage_1sigma - 0.68).abs() / 0.32).clip(lower=0.0, upper=1.0)
    cov2 = (1.0 - (coverage_2sigma - 0.95).abs() / 0.30).clip(lower=0.0, upper=1.0)
    return 0.5 * cov1 + 0.5 * cov2


def bias_component(mean_signed_residual: pd.Series) -> pd.Series:
    """Score residual bias toward zero."""
    return (1.0 - (mean_signed_residual.abs() / 20.0)).clip(lower=0.0, upper=1.0)


def add_balanced_scores(stage_df: pd.DataFrame) -> pd.DataFrame:
    """Attach balanced-score components within one held-out stage."""
    scored = stage_df.copy()
    rmse_values = scored["rmse"].fillna(scored["rmse"].max())
    spearman_values = scored["spearman_rho"].fillna(-1.0)
    coverage_1sigma = scored["coverage_1sigma"].fillna(0.0)
    coverage_2sigma = scored["coverage_2sigma"].fillna(0.0)
    mean_signed_residual = scored["mean_signed_residual"].fillna(20.0)

    scored["rmse_component"] = minmax_inverse(rmse_values)
    scored["spearman_component"] = minmax_forward(spearman_values)
    scored["calibration_component"] = calibration_component(
        coverage_1sigma, coverage_2sigma
    )
    scored["bias_component"] = bias_component(mean_signed_residual)
    scored["balanced_score"] = (
        0.30 * scored["rmse_component"]
        + 0.30 * scored["spearman_component"]
        + 0.20 * scored["calibration_component"]
        + 0.20 * scored["bias_component"]
    )
    return scored


def aggregate_candidate_summary(stage_results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate candidate metrics across held-out stages."""
    summary = (
        stage_results_df.groupby(["label", "method"], as_index=False)
        .agg(
            stages_evaluated=("stage", "nunique"),
            mean_balanced_score=("balanced_score", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            mean_spearman=("spearman_rho", "mean"),
            mean_kendall=("kendall_tau", "mean"),
            mean_coverage_1sigma=("coverage_1sigma", "mean"),
            mean_coverage_2sigma=("coverage_2sigma", "mean"),
            mean_signed_residual=("mean_signed_residual", "mean"),
            mean_abs_residual=("mean_abs_residual", "mean"),
            mean_uncertainty=("mean_uncertainty", "mean"),
        )
        .sort_values(["mean_balanced_score", "mean_rmse"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary


def choose_recommended_method(summary_df: pd.DataFrame) -> Dict[str, object]:
    """Select the best candidate using balanced score plus guardrails."""
    incumbent_row = summary_df[summary_df["label"] == INCUMBENT_LABEL]
    incumbent_mean_rmse = (
        float(incumbent_row.iloc[0]["mean_rmse"]) if not incumbent_row.empty else None
    )

    eligible = summary_df.copy()
    eligible = eligible[eligible["mean_spearman"] >= 0.10]
    if incumbent_mean_rmse is not None:
        eligible = eligible[eligible["mean_rmse"] <= 1.10 * incumbent_mean_rmse]
    eligible = eligible[
        (eligible["mean_coverage_1sigma"] >= 0.55)
        & (eligible["mean_coverage_1sigma"] <= 0.90)
    ]

    recommendation: Dict[str, object] = {
        "incumbent_label": INCUMBENT_LABEL,
        "incumbent_mean_rmse": round_or_none(incumbent_mean_rmse),
        "decision": "no_switch",
        "recommended_label": None,
        "reason": "No candidate satisfied the comparison guardrails.",
        "guardrail_pass_count": int(len(eligible)),
    }

    if eligible.empty:
        return recommendation

    best_score = float(eligible["mean_balanced_score"].max())
    finalists = eligible[eligible["mean_balanced_score"] >= best_score - 0.02].copy()
    finalists = finalists.sort_values(["mean_rmse", "mean_balanced_score"], ascending=[True, False])
    winner = finalists.iloc[0]

    recommendation.update(
        {
            "decision": "switch" if winner["label"] != INCUMBENT_LABEL else "keep_incumbent",
            "recommended_label": str(winner["label"]),
            "recommended_method": str(winner["method"]),
            "reason": "Highest balanced score among candidates that passed all guardrails.",
            "mean_balanced_score": round_or_none(winner["mean_balanced_score"]),
            "mean_rmse": round_or_none(winner["mean_rmse"]),
            "mean_spearman": round_or_none(winner["mean_spearman"]),
            "mean_coverage_1sigma": round_or_none(winner["mean_coverage_1sigma"]),
            "mean_signed_residual": round_or_none(winner["mean_signed_residual"]),
        }
    )
    return recommendation


def write_outputs(stage_results_df: pd.DataFrame, summary_df: pd.DataFrame, recommendation: Dict[str, object]) -> None:
    """Persist comparison artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rolling_records = stage_results_df.to_dict(orient="records")
    summary_records = summary_df.to_dict(orient="records")

    payload = {
        "stage_results": rolling_records,
        "candidate_summary": summary_records,
        "recommendation": recommendation,
    }
    (OUTPUT_DIR / "rolling_method_comparison.json").write_text(json.dumps(payload, indent=2))
    stage_results_df.to_csv(OUTPUT_DIR / "rolling_method_comparison.csv", index=False)
    (OUTPUT_DIR / "recommended_method.json").write_text(json.dumps(recommendation, indent=2))


def write_plot(summary_df: pd.DataFrame) -> None:
    """Render a compact plot of comparison results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ordered = summary_df.sort_values(["mean_balanced_score", "mean_rmse"], ascending=[False, True]).reset_index(drop=True)
    labels = [label.replace("prior_mean_correction_", "prior_").replace("weighted_simple_", "weighted_") for label in ordered["label"]]
    colors = [
        "#d66a1f" if label == INCUMBENT_LABEL else "#1f6aa5"
        for label in ordered["label"]
    ]

    axes[0].bar(labels, ordered["mean_balanced_score"], color=colors)
    axes[0].set_title("Mean Balanced Score", fontsize=11)
    axes[0].tick_params(axis="x", labelrotation=45)
    axes[0].set_ylim(0.0, max(float(ordered["mean_balanced_score"].max()) * 1.15, 1.0))
    for idx, value in enumerate(ordered["mean_balanced_score"]):
        axes[0].text(idx, value + 0.015, f"{value:.2f}", ha="center", va="bottom", fontsize=8.5)

    axes[1].scatter(
        ordered["mean_rmse"],
        ordered["mean_spearman"],
        s=90,
        c=colors,
        edgecolor="#333333",
        linewidth=0.8,
    )
    axes[1].set_title("Mean RMSE vs Mean Spearman", fontsize=11)
    axes[1].set_xlabel("Mean RMSE")
    axes[1].set_ylabel("Mean Spearman")
    axes[1].axhline(0.10, color="#444444", linestyle="--", linewidth=1.0, alpha=0.7)
    for _, row in ordered.iterrows():
        axes[1].text(
            row["mean_rmse"] + 0.15,
            row["mean_spearman"] + 0.01,
            row["label"].replace("prior_mean_correction_", "prior_").replace("weighted_simple_", "weighted_"),
            fontsize=8,
        )

    fig.suptitle("CryoMN Shadow Update-Method Comparison", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "rolling_method_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the rolling retrospective method comparison."""
    feature_names = load_feature_names()
    X_orig, y_orig = load_literature_data(feature_names)
    validation_df = load_validation_df(feature_names)
    completed_stages = sorted(int(stage) for stage in validation_df["stage"].unique() if int(stage) >= 1)
    specs = candidate_specs()

    stage_results: List[Dict[str, object]] = []
    for target_stage in completed_stages:
        train_df = validation_df[validation_df["stage"] < target_stage].copy()
        eval_df = validation_df[validation_df["stage"] == target_stage].copy()
        if train_df.empty or eval_df.empty:
            continue

        X_train_wetlab = train_df.reindex(columns=feature_names, fill_value=0.0).fillna(0.0).to_numpy(float)
        y_train_wetlab = train_df["viability_measured"].to_numpy(float)
        X_eval = eval_df.reindex(columns=feature_names, fill_value=0.0).fillna(0.0).to_numpy(float)
        y_eval = eval_df["viability_measured"].to_numpy(float)

        stage_rows: List[Dict[str, object]] = []
        for spec in specs:
            training = train_candidate(spec, X_orig, y_orig, X_train_wetlab, y_train_wetlab)
            pred_mean, pred_std = predict(
                training["model"],
                training["scaler"],
                X_eval,
                bool(training["is_composite_model"]),
            )
            metrics = compute_metrics(y_eval, pred_mean, pred_std)
            stage_rows.append(
                {
                    "stage": target_stage,
                    "label": spec.label,
                    "method": spec.method,
                    "weight_multiplier": spec.weight_multiplier,
                    "alpha_literature": spec.alpha_literature,
                    "alpha_wetlab": spec.alpha_wetlab,
                    **metrics,
                }
            )

        scored_stage_rows = add_balanced_scores(pd.DataFrame(stage_rows))
        stage_results.extend(scored_stage_rows.to_dict(orient="records"))

    stage_results_df = pd.DataFrame(stage_results)
    summary_df = aggregate_candidate_summary(stage_results_df) if not stage_results_df.empty else pd.DataFrame()
    recommendation = choose_recommended_method(summary_df) if not summary_df.empty else {
        "decision": "no_switch",
        "recommended_label": None,
        "reason": "No completed stage results were available for comparison.",
    }

    write_outputs(stage_results_df, summary_df, recommendation)
    write_plot(summary_df)

    print("=" * 80)
    print("CryoMN Shadow Update-Method Comparison")
    print("=" * 80)
    if summary_df.empty:
        print("No completed stage results were available for comparison.")
        return

    print(summary_df[["label", "mean_balanced_score", "mean_rmse", "mean_spearman"]].to_string(index=False))
    print("")
    print(f"Decision: {recommendation['decision']}")
    print(f"Recommended label: {recommendation.get('recommended_label')}")
    print(f"Reason: {recommendation['reason']}")


if __name__ == "__main__":
    main()
