from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for relative in [
    ("src", "helper"),
    ("src", "03_optimization"),
    ("src", "05_bo_optimization"),
    ("src", "07_next_formulations"),
]:
    path = str(PROJECT_ROOT.joinpath(*relative))
    if path not in sys.path:
        sys.path.insert(0, path)

from formulation_formatting import (  # noqa: E402
    explicit_percentage_cap_excess_from_matrix,
    explicit_percentage_total_from_mapping,
    exceeds_explicit_percentage_cap_mapping,
    exceeds_explicit_percentage_cap_vector,
)
from optimize_formulation import FormulationOptimizer, OptimizationConfig  # noqa: E402
from bo_optimizer import BOConfig, BayesianOptimizer  # noqa: E402
import next_formulations  # noqa: E402
from next_formulations import StageArtifacts, ValidationError  # noqa: E402


class IdentityScaler:
    def transform(self, X):
        return np.asarray(X, dtype=float)


class DummyModel:
    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        mean = np.sum(np.nan_to_num(X), axis=1)
        if return_std:
            return mean, np.full(len(X), 0.25, dtype=float)
        return mean


def test_explicit_percentage_helpers_handle_thresholds_and_legacy_invalid_pattern():
    feature_names = ["dmso_M", "fbs_pct", "human_serum_pct", "hsa_pct", "pvp_pct"]

    row = {
        "fbs_pct": 99.9,
        "human_serum_pct": np.nan,
        "hsa_pct": 0.05,
        "pvp_pct": 0.0,
    }
    assert explicit_percentage_total_from_mapping(row, feature_names) == pytest.approx(99.9)
    assert not exceeds_explicit_percentage_cap_mapping(row, feature_names)

    at_cap = np.array([0.0, 99.9, 0.0, 0.1, 0.0], dtype=float)
    above_cap = np.array([0.0, 100.00001, 0.0, 0.0, 0.0], dtype=float)
    legacy_invalid = np.array(
        [0.0, 58.235598634414444, 64.52989722748069, 0.0, 4.760511939304846],
        dtype=float,
    )

    assert not exceeds_explicit_percentage_cap_vector(at_cap, feature_names)
    assert exceeds_explicit_percentage_cap_vector(above_cap, feature_names)
    assert exceeds_explicit_percentage_cap_vector(legacy_invalid, feature_names)

    excess = explicit_percentage_cap_excess_from_matrix(
        np.vstack([at_cap, legacy_invalid]),
        feature_names,
    )
    assert excess[0] == pytest.approx(0.0)
    assert excess[1] == pytest.approx(27.526007801199978)


def test_random_candidate_generator_filters_over_cap_percentages(monkeypatch):
    feature_names = ["dmso_M", "fbs_pct", "human_serum_pct", "hsa_pct"]
    invalid = np.array([0.0, 58.235598634414444, 64.52989722748069, 4.760511939304846], dtype=float)
    valid = np.array([0.0, 36.8, 0.0, 6.0], dtype=float)

    optimizer = FormulationOptimizer(
        DummyModel(),
        IdentityScaler(),
        feature_names,
        OptimizationConfig(max_ingredients=4, n_candidates=3, random_seed=0),
    )

    state = {"calls": 0}

    def fake_generate_random_candidate():
        state["calls"] += 1
        return invalid.copy() if state["calls"] % 2 else valid.copy()

    monkeypatch.setattr(optimizer, "_generate_random_candidate", fake_generate_random_candidate)

    observed_X = np.array([[0.0, 10.0, 0.0, 0.0]], dtype=float)
    observed_y = np.array([50.0], dtype=float)

    candidates = optimizer.optimize(observed_X, observed_y, n_candidates=3)

    pct_cols = [col for col in candidates.columns if col.endswith("_pct")]
    assert len(candidates) == 3
    assert np.all(candidates[pct_cols].fillna(0.0).sum(axis=1).to_numpy() <= 100.000001)
    assert candidates.get("human_serum_pct", pd.Series(dtype=float)).fillna(0.0).eq(0.0).all()


def test_bo_optimizer_penalizes_and_discards_over_cap_percentages(monkeypatch):
    feature_names = ["dmso_M", "fbs_pct", "human_serum_pct", "hsa_pct"]
    observed_df = pd.DataFrame(
        {
            "dmso_M": [0.0],
            "fbs_pct": [10.0],
            "human_serum_pct": [0.0],
            "hsa_pct": [0.0],
            "viability_percent": [50.0],
        }
    )
    invalid = np.array([0.0, 58.235598634414444, 64.52989722748069, 4.760511939304846], dtype=float)
    valid = np.array([0.0, 40.0, 0.0, 5.0], dtype=float)

    optimizer = BayesianOptimizer(
        DummyModel(),
        None,
        feature_names,
        BOConfig(max_ingredients=4, n_candidates=2, random_seed=0),
        is_composite=True,
    )

    def fake_fit_search_context(frame):
        optimizer.seed_context = frame.copy()
        optimizer.effective_max_ingredients = 4
        optimizer.reference_ingredient_count = 1
        optimizer.observed_support_scaled = None
        optimizer.support_radius = np.inf

    monkeypatch.setattr(optimizer, "_fit_search_context", fake_fit_search_context)

    attempts = {"count": 0}

    def fake_run_de_single(y_best, seed, found_candidates=None, seed_points=None):
        attempts["count"] += 1
        vector = invalid if attempts["count"] == 1 else valid
        return vector.copy(), 1.0

    monkeypatch.setattr(optimizer, "_run_de_single", fake_run_de_single)

    penalty = optimizer._objective_batch(np.vstack([valid, invalid]), y_best=50.0)
    assert penalty[1] > penalty[0] + 1000.0

    candidates = optimizer.optimize(observed_df, n_candidates=2)
    pct_cols = [col for col in candidates.columns if col.endswith("_pct")]
    assert len(candidates) == 2
    assert np.all(candidates[pct_cols].fillna(0.0).sum(axis=1).to_numpy() <= 100.000001)


def test_next_formulations_validation_rejects_rows_above_percentage_cap(monkeypatch):
    feature_names = ["dmso_M", "fbs_pct", "human_serum_pct", "hsa_pct"]

    def make_row(index: int) -> dict:
        fbs_pct = 10.0 + index
        row = {
            "recommendation_type": "exploit" if index < next_formulations.EXPLOIT_COUNT else "explore",
            "formulation": f"row-{index}",
            "rationale": "test rationale",
            "origin": "bo_candidate",
            "predicted_viability": fbs_pct,
            "uncertainty": 0.25,
            "n_ingredients": 1,
            "dmso_percent": 0.0,
            "dmso_M": 0.0,
            "fbs_pct": fbs_pct,
            "human_serum_pct": 0.0,
            "hsa_pct": 0.0,
        }
        return row

    rows = [make_row(index) for index in range(next_formulations.TOTAL_COUNT)]
    rows[0]["fbs_pct"] = 58.235598634414444
    rows[0]["human_serum_pct"] = 64.52989722748069
    rows[0]["hsa_pct"] = 4.760511939304846
    rows[0]["predicted_viability"] = (
        rows[0]["fbs_pct"] + rows[0]["human_serum_pct"] + rows[0]["hsa_pct"]
    )
    rows[0]["n_ingredients"] = 3

    monkeypatch.setattr(
        next_formulations,
        "predict",
        lambda model, scaler, X, is_composite_model: (
            np.sum(X, axis=1),
            np.full(len(X), 0.25, dtype=float),
        ),
    )

    active_stage = StageArtifacts(
        stage=5,
        iteration_dir="iteration_5_prior_mean",
        metadata={},
        feature_names=feature_names,
        is_composite_model=True,
        model=object(),
        scaler=None,
        observed_context=pd.DataFrame(),
    )
    optimizer = SimpleNamespace(
        effective_max_ingredients=4,
        config=SimpleNamespace(max_dmso_percent=5.0),
        bounds=[(0.0, 0.704), (0.0, 90.0), (0.0, 90.0), (0.0, 10.0)],
    )

    with pytest.raises(ValidationError, match="explicit percentage cap"):
        next_formulations.validate_output_rows(rows, active_stage, optimizer, tested_signatures=set())
