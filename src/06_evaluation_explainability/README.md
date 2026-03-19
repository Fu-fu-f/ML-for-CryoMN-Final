# Step 6: Evaluation and Explainability

## Overview

This module groups the post-update analysis tools:

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
- recommendation-slate evaluation for `results/next_formulations/<iteration_tag>/next_formulations.csv`, including exploit/explore and origin-level summaries when those files exist

`stage_performance.png` is a categorized small-multiples dashboard rather than
a 3-metric summary chart. It groups stage metrics into:

- error: RMSE and MAE
- ranking: Spearman rho and Kendall tau
- calibration: mean uncertainty and coverage @ 1σ
- threshold decision: hit rate @ 50% and hit rate @ 70%

Each metric gets its own raw-value bar-chart subplot, with the category
groupings preserved in the overall layout. Missing stages render as `N/A`
annotations instead of bars.

Candidate-hit matching uses the same practical concentration floor as `05` and
`07`:

- `_pct` values `<0.1%` are treated as absent
- `_M` values `<0.001 M` (`<1.0 mM`) are treated as absent

This means a frozen candidate row can still count as a later wet-lab hit when
the only difference is a trace ingredient that should effectively be zero.

Additional evaluation outputs:

- `results/evaluation/next_formulations_performance.png`

The recommendation-slate audit rescales the saved `07` rows with the frozen
stage model inside `06`, then compares them with later wet-lab measurements.
It reports:

- overall `07` slate performance
- `exploit` versus `explore` summaries
- origin-level summaries such as `bo_candidate`, `local_rank_probe`, `blindspot_probe`, and `explore_fallback`

## Explainability

`explainability.py` generates iteration-specific artifacts under:

- `results/explainability/<iteration_tag>/`

`<iteration_tag>` comes from the resolved active model identity, for example:

- `iteration_1`
- `iteration_3_weighted_simple`
- `iteration_5_prior_mean`

If no explicit iteration metadata exists, the fallback directory is:

- `results/explainability/active_model/`

The explainability suite is now intentionally support-aware:

- slice and contour axes default to observed quantile bounds instead of raw extrema
- observed literature and wet-lab rows are overlaid wherever support matters
- stronger-support regions are marked with dashed boundaries or line-style changes instead of masking the surface
- the BO landscape keeps the contour aesthetic, but documents which production penalties are included

Support cues:

- In `partial_dependence_plots.png`, dashed curve segments indicate the same empirical slice continued outside stronger local 1D support.
- In `interaction_contours.png` and `acquisition_landscape.png`, the dashed white boundary marks the stronger pairwise support envelope estimated from observed formulations.
- Inside that dashed boundary, the surface is better grounded in observed data. Outside it, the surface is still shown for continuity, but should be interpreted as more extrapolative.
- In `feature_importance.png`, the vertical dashed line is only a visual dominance cutoff to separate the strongest features from the long tail; it is not a hard statistical threshold.
- In `shap_summary.png`, only the top features are shown. Color encodes feature value and horizontal spread shows directional contribution magnitude across observed formulations.

Artifacts:

| File | Description |
|------|-------------|
| `feature_importance.csv` | Recomputed permutation importance (weighted for composite model) |
| `feature_importance.png` | Publication-style overview of weighted permutation importance with dominant-feature emphasis |
| `shap_summary.png` | SHAP beeswarm focused on the top features and their directional impact on observed rows |
| `shap_importance.png` | SHAP-based feature importance ranking |
| `partial_dependence_plots.png` | Support-aware empirical marginal slices over observed rows, with dashed segments outside stronger local support |
| `interaction_contours.png` | Support-aware pairwise contour maps with observed-point overlays and dashed support boundaries |
| `acquisition_landscape.png` | Static BO score landscape using the `05` visual language, with support and sparsity penalties but no sequential batch-diversity term |
| `uncertainty_analysis.png` | Decision-focused uncertainty dashboard covering calibration, residual growth, and uncertainty by viability band |
| `support_diagnostics.png` | Compact support-envelope view for the top features and top pair, split by literature vs wet lab |

Feature importance is always recomputed at runtime against the resolved active
model. When using the composite model, weighting follows `context_weight` so
wet-lab rows influence importance consistently with the active iteration.

`acquisition_landscape.png` should be interpreted as a static approximation of
the production BO objective. It reuses the `05_bo_optimization` acquisition
settings and static penalties, but it does not include sequential
batch-diversity effects that depend on already-selected candidates.

> `shap` is optional. If it is not installed, the SHAP plots are skipped while
> the other explainability outputs still run.

## Feature Name Handling

Display labels are cleaned automatically:

- `dmso_M` -> `Dmso`
- `fbs_pct` -> `Fbs`
- `hyaluronic_acid_pct` -> `Hyaluronic Acid`
