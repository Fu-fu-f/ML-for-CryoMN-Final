# Step 6: Evaluation and Explainability

## Overview

This module now groups the post-update analysis tools:

| Script | Purpose | Main outputs |
|--------|---------|--------------|
| `evaluate_iterations.py` | Score each frozen stage against the wet-lab batch it actually produced | `results/evaluation/` |
| `explainability.py` | Visualize what the active model is learning | `results/explainability/<iteration_tag>/` |

Both scripts use the same iteration-aware model resolution used by
`03_optimization` and `05_bo_optimization`.

## Usage

```bash
cd "/path/to/project"

# Stage-based evaluation
python src/06_evaluation_explainability/evaluate_iterations.py

# Model explainability
python src/06_evaluation_explainability/explainability.py
```

## Shared Inputs

- `models/model_metadata.json`
- `data/validation/iteration_history.json`
- `models/iteration_*`

For the active iteration, `06_evaluation_explainability` resolves the exact
checkpoint from metadata plus iteration history. If the active metadata is
missing or inconsistent, `explainability.py` can prompt for an iteration number
and repair `models/model_metadata.json` with an explicit overwrite notice.

When the canonical observed-context artifact is missing, `explainability.py`
reconstructs it from literature plus wet-lab inputs. For older prior-mean
iterations it can also read the legacy `data/processed/evaluation_data.csv`
mirror.

## Stage Evaluation

`evaluate_iterations.py` compares each frozen stage against the later wet-lab
rows that stage actually generated:

- result files without an iteration suffix map to the literature-only stage
- `iteration_1_*` outputs map to the first post-validation wet-lab batch
- `iteration_2_*`, `iteration_3_*`, and later outputs follow the same stage ID rule
- when available, the literature baseline is loaded from `models/literature_only/`

Outputs:

- `results/evaluation/iteration_prospective_summary.json`
- `results/evaluation/iteration_prospective_metrics.csv`
- `results/evaluation/stage_performance.png`

The evaluator reports:

- batch-level predictive metrics such as RMSE, MAE, Spearman, Kendall, coverage, and hit rates
- candidate-rank cross references showing which frozen candidate rows were later tested in wet lab

Candidate-hit matching uses the same practical concentration floor as `05` and
`07`:

- `_pct` values `<0.1%` are treated as absent
- `_M` values `<0.001 M` (`<1.0 mM`) are treated as absent

This means a frozen candidate row can still count as a later wet-lab hit when
the only difference is a trace ingredient that should effectively be zero.

## Explainability

`explainability.py` generates iteration-specific artifacts under:

- `results/explainability/<iteration_tag>/`

`<iteration_tag>` comes from the resolved active model identity, for example:

- `iteration_1`
- `iteration_3_weighted_simple`
- `iteration_5_prior_mean`

If no explicit iteration metadata exists, the fallback directory is:

- `results/explainability/active_model/`

Artifacts:

| File | Description |
|------|-------------|
| `feature_importance.csv` | Recomputed permutation importance (weighted for composite model) |
| `feature_importance.png` | Horizontal bar chart of permutation-based feature importance |
| `shap_summary.png` | SHAP beeswarm plot showing individual feature impacts |
| `shap_importance.png` | SHAP-based feature importance ranking |
| `partial_dependence_plots.png` | PDPs showing how viability changes with each ingredient |
| `interaction_contours.png` | 2D contour plots of ingredient pair interactions |
| `acquisition_landscape.png` | Acquisition visualization (UCB by default; EI configurable) |
| `uncertainty_analysis.png` | GP uncertainty calibration and distribution analysis |

Feature importance is always recomputed at runtime against the resolved active
model. When using the composite model, weighting follows `context_weight` so
wet-lab rows influence importance consistently with the active iteration.

> `shap` is optional. If it is not installed, the SHAP plots are skipped while
> the other explainability outputs still run.

## Feature Name Handling

Display labels are cleaned automatically:

- `dmso_M` -> `Dmso`
- `fbs_pct` -> `Fbs`
- `hyaluronic_acid_pct` -> `Hyaluronic Acid`
