import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


REPO_ROOT = Path("/Users/doggonebastard/Antigravity/ML for CryoMN-observed-context")
VALIDATION_DIR = REPO_ROOT / "src/04_validation_loop"
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


observed_context = load_module("observed_context_module", "src/04_validation_loop/observed_context.py")
opt03 = load_module("opt03_module", "src/03_optimization/optimize_formulation.py")
bo05 = load_module("bo05_module", "src/05_bo_optimization/bo_optimizer.py")
exp06 = load_module("exp06_module", "src/06_explainability/explainability.py")


class IdentityScaler:
    def transform(self, X):
        return np.asarray(X, dtype=float)


class FakeCompositeModel:
    def __init__(self, constant=None):
        self.constant = constant
        self.scaler_literature = IdentityScaler()

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        if self.constant is None:
            mean = 15.0 + np.sum(X, axis=1) * 10.0
        else:
            mean = np.full(X.shape[0], float(self.constant))
        std = np.full(X.shape[0], 0.5)
        if return_std:
            return mean, std
        return mean


def make_project(tmp_path: Path, feature_names):
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "data/validation").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)

    literature_df = pd.DataFrame(
        [
            {"formulation_id": 1, "viability_percent": 40.0, feature_names[0]: 0.1, feature_names[1]: 0.0},
            {"formulation_id": 2, "viability_percent": 55.0, feature_names[0]: 0.0, feature_names[1]: 0.2},
        ]
    )
    literature_df.to_csv(tmp_path / "data/processed/parsed_formulations.csv", index=False)

    wetlab_df = pd.DataFrame(
        [
            {
                "experiment_id": "EXP001",
                "experiment_date": "2026-03-10",
                "viability_measured": 80.0,
                "notes": "test",
                feature_names[0]: 0.3,
                feature_names[1]: 0.4,
            }
        ]
    )
    wetlab_df.to_csv(tmp_path / "data/validation/validation_results.csv", index=False)

    return literature_df, wetlab_df


def make_resolution(
    feature_names,
    model_method="prior_mean_correction",
    iteration=2,
    iteration_dir="iteration_2_prior_mean",
    **metadata,
):
    full_metadata = {
        "feature_names": feature_names,
        "model_method": model_method,
        "iteration": iteration,
        "iteration_dir": iteration_dir,
        "is_composite_model": model_method == "prior_mean_correction",
    }
    full_metadata.update(metadata)
    return SimpleNamespace(
        gp=FakeCompositeModel(),
        scaler=None,
        metadata=full_metadata,
        is_composite=full_metadata["is_composite_model"],
        iteration=iteration,
        iteration_dir=iteration_dir,
        model_method=model_method,
    )


def test_observed_context_builder_and_reconstruction_weights(tmp_path):
    feature_names = ["dmso_M", "trehalose_M"]
    literature_df, wetlab_df = make_project(tmp_path, feature_names)

    standard_df = observed_context.reconstruct_observed_context(
        str(tmp_path), feature_names, "standard", 1, "iteration_1", {}
    )
    assert list(standard_df["context_weight"]) == [1.0, 1.0, 1.0]
    assert set(standard_df["source"]) == {"literature", "wetlab"}

    weighted_df = observed_context.reconstruct_observed_context(
        str(tmp_path),
        feature_names,
        "weighted_simple",
        2,
        "iteration_2_weighted_simple",
        {"weight_multiplier": 10},
    )
    assert weighted_df.loc[weighted_df["source"] == "wetlab", "context_weight"].tolist() == [10.0]

    prior_df = observed_context.reconstruct_observed_context(
        str(tmp_path),
        feature_names,
        "prior_mean_correction",
        3,
        "iteration_3_prior_mean",
        {"noise_ratio": 50.0, "is_composite_model": True},
    )
    assert prior_df.loc[prior_df["source"] == "wetlab", "context_weight"].tolist() == [50.0]
    assert len(prior_df) == len(literature_df) + len(wetlab_df)


def test_03_loads_combined_context_and_remains_deterministic(tmp_path):
    feature_names = ["dmso_M", "trehalose_M"]
    make_project(tmp_path, feature_names)
    resolution = make_resolution(feature_names, noise_ratio=50.0)

    observed_df = opt03.load_observed_data(str(tmp_path), resolution)
    assert len(observed_df) == 3
    assert set(observed_df["source"]) == {"literature", "wetlab"}

    optimizer_a = opt03.FormulationOptimizer(
        gp=FakeCompositeModel(),
        scaler=None,
        feature_names=feature_names,
        config=opt03.OptimizationConfig(n_candidates=5, max_ingredients=2),
        is_composite=True,
    )
    optimizer_b = opt03.FormulationOptimizer(
        gp=FakeCompositeModel(),
        scaler=None,
        feature_names=feature_names,
        config=opt03.OptimizationConfig(n_candidates=5, max_ingredients=2),
        is_composite=True,
    )

    np.random.seed(42)
    result_a = optimizer_a.optimize(
        observed_df[feature_names].values,
        observed_df["viability_percent"].values,
        n_candidates=5,
    )
    np.random.seed(42)
    result_b = optimizer_b.optimize(
        observed_df[feature_names].values,
        observed_df["viability_percent"].values,
        n_candidates=5,
    )
    pd.testing.assert_frame_equal(result_a, result_b)


def test_05_collapses_duplicate_weights_and_keeps_support_radius_nonzero():
    feature_names = ["dmso_M", "trehalose_M"]
    observed_df = pd.DataFrame(
        [
            {"dmso_M": 0.0, "trehalose_M": 0.2, "viability_percent": 70.0, "source": "wetlab", "context_weight": 10.0},
            {"dmso_M": 0.0, "trehalose_M": 0.2, "viability_percent": 70.0, "source": "wetlab", "context_weight": 10.0},
            {"dmso_M": 0.1, "trehalose_M": 0.0, "viability_percent": 45.0, "source": "literature", "context_weight": 1.0},
            {"dmso_M": 0.2, "trehalose_M": 0.3, "viability_percent": 50.0, "source": "literature", "context_weight": 1.0},
        ]
    )

    collapsed = observed_context.collapse_observed_context_for_bo(observed_df, feature_names)
    assert len(collapsed) == 3
    assert collapsed.loc[
        (collapsed["dmso_M"] == 0.0) & (collapsed["trehalose_M"] == 0.2),
        "context_weight",
    ].iloc[0] == 20.0

    optimizer = bo05.BayesianOptimizer(
        gp=FakeCompositeModel(),
        scaler=None,
        feature_names=feature_names,
        config=bo05.BOConfig(max_ingredients=None),
        is_composite=True,
    )
    optimizer._fit_search_context(observed_df)
    assert np.isfinite(optimizer.support_radius)
    assert optimizer.support_radius > 0


def test_05_seed_order_breaks_prediction_ties_by_context_weight():
    feature_names = ["dmso_M", "trehalose_M"]
    observed_df = pd.DataFrame(
        [
            {"dmso_M": 0.1, "trehalose_M": 0.0, "viability_percent": 40.0, "source": "literature", "context_weight": 1.0},
            {"dmso_M": 0.0, "trehalose_M": 0.3, "viability_percent": 40.0, "source": "wetlab", "context_weight": 50.0},
        ]
    )

    optimizer = bo05.BayesianOptimizer(
        gp=FakeCompositeModel(constant=10.0),
        scaler=None,
        feature_names=feature_names,
        config=bo05.BOConfig(n_candidates=2),
        is_composite=True,
    )
    result = optimizer.optimize(observed_df, n_candidates=2)
    top_row = result.iloc[0]
    assert top_row["trehalose_M"] == 0.3


def test_06_loads_observed_context_artifact(monkeypatch, tmp_path):
    feature_names = ["dmso_M", "trehalose_M"]
    make_project(tmp_path, feature_names)
    iteration_dir = "iteration_2_prior_mean"
    (tmp_path / "models" / iteration_dir).mkdir(parents=True)

    artifact_df = pd.DataFrame(
        [
            {
                "dmso_M": 0.0,
                "trehalose_M": 0.2,
                "viability_percent": 70.0,
                "source": "wetlab",
                "context_weight": 50.0,
                "model_method": "prior_mean_correction",
                "iteration": 2,
                "iteration_dir": iteration_dir,
            }
        ]
    )
    artifact_df.to_csv(tmp_path / "models" / iteration_dir / "observed_context.csv", index=False)
    resolution = make_resolution(feature_names)
    monkeypatch.setattr(exp06, "resolve_active_model", lambda project_root: resolution)

    _, _, _, df, _, _, _ = exp06.load_model_and_data(str(tmp_path))
    assert len(df) == 1
    assert df["context_weight"].iloc[0] == 50.0


def test_06_reconstructs_context_when_artifact_is_missing(monkeypatch, tmp_path):
    feature_names = ["dmso_M", "trehalose_M"]
    make_project(tmp_path, feature_names)
    resolution = make_resolution(feature_names, noise_ratio=50.0)
    monkeypatch.setattr(exp06, "resolve_active_model", lambda project_root: resolution)

    _, _, _, df, _, _, _ = exp06.load_model_and_data(str(tmp_path))
    assert len(df) == 3
    assert df.loc[df["source"] == "wetlab", "context_weight"].tolist() == [50.0]
