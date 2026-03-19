# CryoMN ML-Based Cryoprotective Solution Optimization

Machine learning pipeline for optimizing cryoprotective formulations for cryomicroneedle (CryoMN) technology.

Repository checkpoint artifacts referenced below use stage-tagged iteration directories such as `iteration_5_prior_mean`.

## Goals

1. **Minimize DMSO usage** (reduce toxicity)
2. **Maximize cell viability** (maintain therapeutic efficacy)
3. **Limit ingredients** (≤10 components per formulation)

---

## Workflow Overview

The project was developed using a multi-agent AI workflow, combining planning and implementation phases with human oversight:

![Project Workflow Schematic](workflow_schematic_final.png)

---

## Approach

**Gaussian Process Regression + Bayesian Optimization**
- Works well with limited data (~200 samples)
- Provides uncertainty quantification
- Supports iterative refinement with wet lab validation

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Parse formulation data
python src/01_data_parsing/parse_formulations.py

# 2. Train GP model
python src/02_model_training/train_gp_model.py

# 3. Generate candidates
python src/03_optimization/optimize_formulation.py      # Fast random sampling, iteration-aware
python src/05_bo_optimization/bo_optimizer.py          # Proper BO with DE

# 4. Integrate wet lab results (after experiments)
python src/04_validation_loop/update_model.py
python src/04_validation_loop/update_model_weighted_simple.py
python src/04_validation_loop/update_model_weighted_prior.py

# 5. Explain model predictions (auto-detects composite model)
python src/06_evaluation_explainability/explainability.py

# 6. Evaluate frozen stages against their wet-lab batches
python src/06_evaluation_explainability/evaluate_iterations.py

# 7. Generate the next 20-formulation wet-lab batch
python src/07_next_formulations/next_formulations.py
```

> [!CAUTION]
> `python src/05_bo_optimization/bo_optimizer.py` is a long-running optimization step. It evaluates repeated Differential Evolution searches for both general and low-DMSO candidate batches, and it prints live DE-search status while the script is running.

## Repository Snapshot

The snapshot dated 2026-03-16 uses the composite prior-mean correction checkpoint `iteration_4_prior_mean`.

| Metric | Snapshot value |
|--------|---------------|
| Wet-lab validation rows | 43 |
| Wet-lab batch date in snapshot | 2026-03-12 |
| Best validated viability | 79.09% |
| Best validated formulation | 304.7mM ectoin + 1.55M ethylene glycol |
| Mean wet-lab viability | 37.75% |
| Median wet-lab viability | 35.08% |
| Wet-lab runs at or above 50% viability | 14 |

The snapshot still highlights the ectoin + ethylene glycol ridge, while the
residual-driven `07_next_formulations` step adapts to the blind spots exposed
by completed wet-lab stages.

## Active Model and Iterations

- `src/04_validation_loop/*` saves each retrained checkpoint under `models/iteration_*`, updates `data/validation/iteration_history.json`, and then replaces the active root metadata in `models/model_metadata.json` with an explicit overwrite notice.
- `src/03_optimization/optimize_formulation.py`, `src/05_bo_optimization/bo_optimizer.py`, and `src/06_evaluation_explainability/explainability.py` share the same iteration-aware resolver.
- `src/04_validation_loop/*` also writes a canonical observed-context artifact to `models/<iteration_dir>/observed_context.csv` and mirrors the active copy to `models/observed_context.csv`.
- If root metadata is missing or inconsistent, these entry points prompt for an iteration number, reject nonsensical choices, and repair `models/model_metadata.json` only after telling you it is overwriting the metadata.
- Composite iterations are strict: if metadata says composite, the shared resolver will not fall back to a standard GP automatically.
- `03_optimization`, `05_bo_optimization`, and `06_evaluation_explainability` all load the same iteration-aware observed context, and reconstruct it from literature + validation inputs on demand if the artifact is missing.
- `05_bo_optimization` uses analytic wet-lab weights from the observed context when calibrating BO support geometry, instead of relying on literal duplicate rows.
- `03`, `05`, `06`, and `07` share a practical concentration floor for formulation identity: values below `0.1%` or below `1.0 mM` are treated as absent when generating candidates, matching hits, and rendering formulation strings.
- The repository does not rely on one permanent static train/test split. The update scripts estimate wet-lab generalization with K-fold cross-validation over wet-lab rows only, while retaining all literature rows in training for every fold.
- In code, the wet-lab fold count is `min(5, len(X_val))` with `shuffle=True` and `random_state=42`. In the saved iterations in this repo that behaves as 5-fold CV, because every completed wet-lab stage has at least 5 measured rows.
- For the standard update, each fold trains on `literature + wetlab_train_fold` and predicts the held-out wet-lab fold. The weighted-simple update uses the same split but duplicates the training-fold wet-lab rows, and the prior-mean update keeps the literature GP fixed while cross-validating only the wet-lab residual correction GP.

## Results Snapshot

### Wet-Lab Validation Signal

The best measured wet-lab result in this snapshot is:

- `79.09%` viability for `304.7mM ectoin + 1.55M ethylene glycol`

That same region remains the model's top BO target, which is a useful consistency check between prediction and validation.

### DE-Based Bayesian Optimization (`05_bo_optimization`)

General BO summary for this snapshot: `results/bo_candidates_general_iteration_4_prior_mean_summary.txt`

| Rank | Formulation | Predicted viability |
|------|-------------|---------------------|
| 1 | 304.7mM ectoin + 1.55M ethylene glycol | 77.9% ± 14.2% |
| 2 | 310.5mM ectoin + 1.57M ethylene glycol + 0.1% HSA + 142.8mM raffinose | 75.9% ± 15.3% |
| 3 | 314.1mM ectoin + 1.43M ethylene glycol | 75.6% ± 15.6% |
| 4 | 304.7mM ectoin + 1.55M ethylene glycol + 6.5% FBS | 72.6% ± 16.9% |
| 5 | 48.6mM betaine + 32.5mM DMSO + 305.6mM ectoin + 1.65M ethylene glycol + 0.2% HSA + 24.6mM sucrose | 72.6% ± 17.0% |

### Next Formulations (`07_next_formulations`)

The recommended next wet-lab batch comes from:

```bash
python src/07_next_formulations/next_formulations.py
```

This script:
- resolves the active iteration automatically
- requires validation coverage through stage `N-1` when targeting stage `N`
- uses `05` BO outputs for 8 exploitation picks
- normalizes existing and newly generated candidates so trace ingredients below `0.1%` or `1.0 mM` are treated as absent
- adaptively relaxes the positive-residual anchor threshold when stronger anchors are unavailable
- builds 12 exploration/calibration rows as 8 local-rank probes plus 4 blind-spot probes, then uses BO fallback only if needed
- allows exploration probes to anchor from any historical positive-residual wet-lab stage
- writes recommended batch subsets for wet-lab capacities from 6 to 12 formulations
- validates inputs before generation and validates all 20 outputs again before writing

Outputs are written under `results/next_formulations/<iteration_tag>/`, for example:
- `results/next_formulations/iteration_5_prior_mean/next_formulations.csv`
- `results/next_formulations/iteration_5_prior_mean/next_formulations_summary.txt`
- `results/next_formulations/iteration_5_prior_mean/next_formulations_metadata.json`
- `results/next_formulations/iteration_5_prior_mean/input_validation.json`
- `results/next_formulations/iteration_5_prior_mean/batch_recommendations.json`
- `results/next_formulations/iteration_5_prior_mean/batch_recommendations.csv`

The summary and metadata artifacts record which positive-residual thresholds
were tried, which threshold was selected, how many exploration rows came
from local-rank probes, blind-spot probes, and BO fallback, and which
historical anchor stages fed the generated probes. The text summary also
includes a human-readable version of each recommended batch subset for wet-lab
capacities from 6 through 12 formulations.

The batch recommendation `score` is a heuristic subset-selection score. It is
used to rank candidate subsets built from the 20-row slate, not to represent
predicted viability or expected improvement directly. Higher scores reflect a
better tradeoff among row utility, chemistry-family diversity, local-anchor
diversity, and closeness to the intended exploit / local-rank / blind-spot mix.

The per-row `utility` values shown in the text summary are the row-level inputs
to that subset score. They are also heuristic. A row's utility depends on its
role: exploitation rows emphasize predicted viability and confidence, while
exploration rows place more weight on uncertainty, blind-spot value, and
novelty.

### Stage-Based Evaluation

The repository includes a stage-based evaluator that scores each frozen
model output against the wet-lab batch it actually generated:

- standalone literature-only checkpoint in `models/literature_only/` plus outputs in `results/*` without an iteration suffix → `EXP101` to `EXP306`
- `iteration_1_*` outputs → `EXP1101` to `EXP1206`
- `iteration_2_*` outputs → `EXP2101` to `EXP2106`
- `iteration_3_*` outputs → `EXP3101` to `EXP3108`
- `iteration_4_*` outputs → `EXP4101` to `EXP4106`
- `iteration_5_*` outputs → pending wet-lab results

Run:

```bash
python src/06_evaluation_explainability/evaluate_iterations.py
```

Outputs:

- `results/evaluation/iteration_prospective_summary.json`
- `results/evaluation/iteration_prospective_metrics.csv`
- `results/evaluation/stage_performance.png`
- `results/evaluation/next_formulations_performance.png`

Candidate-hit matching in `06_evaluation_explainability` uses the same
practical concentration floor, so frozen candidate rows still count as later
hits when the only difference is a trace ingredient below `0.1%` or `1.0 mM`.

Stage-level metrics from the saved evaluation artifacts:

| Stage | Validation batch | Rows | RMSE | Spearman | Hit Rate @ 50% |
|------|-------------------|------|------|----------|----------------|
| Literature only | `EXP101-EXP306` | 18 | 41.21 | -0.327 | 0.444 |
| Iteration 1 | `EXP1101-EXP1206` | 11 | 21.67 | -0.518 | 0.909 |
| Iteration 2 | `EXP2101-EXP2106` | 6 | 14.74 | 0.086 | 0.667 |
| Iteration 3 | `EXP3101-EXP3108` | 8 | 21.05 | 0.476 | 0.625 |
| Iteration 4 | `EXP4101-EXP4106` | 6 | 9.24 | -0.600 | 1.000 |
| Iteration 5 | pending wet-lab results | 0 | N/A | N/A | N/A |

Interpretation:

- absolute error improved substantially from literature-only to iteration 2
- rank ordering is still weak, especially for literature-only and iteration 1
- iteration 3 is a better ranker than iteration 2, but still a weak calibrated predictor
- `07_next_formulations` uses stage residuals plus BO outputs to choose a mixed exploit/explore wet-lab batch

![Stage Performance](results/evaluation/stage_performance.png)

---

## Model Explainability

Understanding which ingredients drive cell viability predictions is crucial for guiding wet lab experiments. The explainability module generates comprehensive visualizations:

### Explainability Outputs (`iteration_3_prior_mean`)

The explainability outputs shown here live in `results/explainability/iteration_3_prior_mean/`. The top-ranked features in this checkpoint are `ethylene_glycol`, `glycerol`, `ectoin`, `dmso`, and `hsa`.

#### SHAP Summary

![SHAP Summary](results/explainability/iteration_3_prior_mean/shap_summary.png)

#### Feature Importance

![Feature Importance](results/explainability/iteration_3_prior_mean/feature_importance.png)

#### Acquisition Landscape

The acquisition landscape defaults to **Upper Confidence Bound (UCB)**, highlighting the model's exploitation-exploration tradeoff:

![Acquisition Landscape](results/explainability/iteration_3_prior_mean/acquisition_landscape.png)

#### Interaction Contours

Visualizing how pairs of top ingredients interact to affect viability:

![Interaction Contours](results/explainability/iteration_3_prior_mean/interaction_contours.png)

#### Uncertainty Analysis

![Uncertainty Analysis](results/explainability/iteration_3_prior_mean/uncertainty_analysis.png)

For detailed interpretation and additional visualizations, see [`src/06_evaluation_explainability/README.md`](src/06_evaluation_explainability/README.md).

---

## Project Structure

```
├── data/
│   ├── raw/                    # Original literature data
│   ├── processed/              # Parsed formulations + legacy prior-mean evaluation mirror
│   └── validation/             # Wet lab results template
├── models/                     # Active model mirror + per-iteration checkpoints + observed context
├── results/                    # Optimized candidates + explainability + evaluation graphs
└── src/
    ├── 01_data_parsing/        # Parse CSV, normalize units, merge synonyms
    ├── 02_model_training/      # Train GP regression model (Matérn kernel)
    ├── 03_optimization/        # Random sampling + GP prediction (fast)
    ├── 04_validation_loop/     # Integrate wet lab feedback, retrain model
    ├── 05_bo_optimization/     # Proper BO with Differential Evolution
    ├── 06_evaluation_explainability/  # Stage evaluation + explainability plots
    └── 07_next_formulations/   # Build the next 20-formulation wet-lab batch
```

## Module Descriptions

| Module | Method | Best For |
|--------|--------|----------|
| `01_data_parsing` | Data Parsing & Normalization | Preparing clean, structured training data from raw literature |
| `02_model_training` | Gaussian Process Regression (Matérn Kernel) | Learning the viability landscape from limited data |
| `03_optimization` | Random sampling, iteration-aware model loading | Quick generation, metadata repair when active model state is inconsistent |
| `04_validation_loop` | Three update strategies + iteration checkpointing + shadow method comparison helpers | Closing the active learning loop with wet lab feedback and comparing candidate update methods without activation |
| `05_bo_optimization` | Differential Evolution with batched population scoring, wet-lab-aware BO context, shared iteration-aware model loading | Exploiting validated winners while proposing nearby informative variants |
| `06_evaluation_explainability` | Stage-based evaluation, recommendation-slate auditing, SHAP, PDPs, Interaction Contours, shared iteration-aware model loading | Measuring frozen-stage performance, auditing `07` outputs, and understanding model drivers |
| `07_next_formulations` | Strict next-batch generation from BO outputs + residual blind spots + smaller-batch subset recommendation | Selecting exactly 20 future wet-lab formulations with an 8 exploit / 12 explore split and recommending subsets for batch sizes 6 through 12 |

## Key Features

- **34 ingredients** tracked (DMSO, trehalose, glycerol, FBS, PEG by MW, etc.)
- **PEG molecular weight handling**: Individual tracking of PEG 400, 600, 1K, 3350, 5K, 10K, 20K, etc.
- **Dual unit handling**: Molar (`_M`) for CPAs, Percentage (`_pct`) for sera/polymers
- **Synonym merging** (e.g., FBS = FCS = fetal bovine serum)
- **Unit normalization** (concentrations converted to molar or kept as percentage)
- **Uncertainty quantification** (GP provides confidence intervals)
- **Iterative refinement** (model improves with each wet lab validation)
- **Iteration-aware recovery** (`03` can repair missing/conflicting active metadata interactively)
- **Explainable AI** (SHAP and partial dependence plots to interpret Black Box GP)
- **Two optimization modes**: Fast random sampling OR proper Bayesian optimization
- **Canonical observed context** (`04` writes `observed_context.csv`; `03`, `05`, and `06` all consume the same active iteration view)
- **Wet-lab-aware BO** (`05` uses weighted observed context and seeds from top observed formulations)
- **Vectorized DE scoring** (`05` evaluates each DE population in batches so GP prediction and penalty calculations are not repeated point-by-point)
- **Strict next-batch planning** (`07` validates inputs, generates calibration probes from residual blind spots, and writes traceable next-batch artifacts)
- **Subset recommendation for limited wet-lab capacity** (`07` writes exact best-subset recommendations for batch sizes 6 through 12)
