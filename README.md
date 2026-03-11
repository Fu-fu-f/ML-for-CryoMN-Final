# CryoMN ML-Based Cryoprotective Solution Optimization

Machine learning pipeline for optimizing cryoprotective formulations for cryomicroneedle (CryoMN) technology.

Repository checkpoint: `iteration_3_prior_mean` (snapshot dated 2026-03-11).

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

# 3. Generate candidates (choose one)
python src/03_optimization/optimize_formulation.py      # Fast random sampling, iteration-aware
python src/05_bo_optimization/bo_optimizer.py          # Proper BO with DE

# 4. Integrate wet lab results (after experiments)
python src/04_validation_loop/update_model_weighted_prior.py

# 5. Explain model predictions (auto-detects composite model)
python src/06_explainability/explainability.py

# 6. Filter out already tested candidates
python filter_tested_candidates.py
```

## Repository Snapshot

The snapshot dated 2026-03-11 uses the composite prior-mean correction checkpoint `iteration_3_prior_mean`.

| Metric | Snapshot value |
|--------|---------------|
| Wet-lab validation rows | 35 |
| Wet-lab batch date in snapshot | 2026-03-11 |
| Best validated viability | 79.09% |
| Best validated formulation | 304.7mM ectoin + 1.55M ethylene glycol |
| Mean wet-lab viability | 35.52% |
| Median wet-lab viability | 31.90% |
| Wet-lab runs at or above 50% viability | 11 |

This snapshot highlights convergence around an ectoin + ethylene glycol region, with glycerol and serum/protein additives acting as secondary modifiers rather than replacing that core pair.

## Active Model and Iterations

- `src/04_validation_loop/*` saves each retrained checkpoint under `models/iteration_*`, updates `data/validation/iteration_history.json`, and then replaces the active root metadata in `models/model_metadata.json` with an explicit overwrite notice.
- `src/03_optimization/optimize_formulation.py`, `src/05_bo_optimization/bo_optimizer.py`, and `src/06_explainability/explainability.py` share the same iteration-aware resolver.
- `src/04_validation_loop/*` also writes a canonical observed-context artifact to `models/<iteration_dir>/observed_context.csv` and mirrors the active copy to `models/observed_context.csv`.
- If root metadata is missing or inconsistent, these entry points prompt for an iteration number, reject nonsensical choices, and repair `models/model_metadata.json` only after telling you it is overwriting the metadata.
- Composite iterations are strict: if metadata says composite, the shared resolver will not fall back to a standard GP automatically.
- `03_optimization`, `05_bo_optimization`, and `06_explainability` all load the same iteration-aware observed context, and reconstruct it from literature + validation inputs on demand if the artifact is missing.
- `05_bo_optimization` uses analytic wet-lab weights from the observed context when calibrating BO support geometry, instead of relying on literal duplicate rows.

## Results Snapshot

### Wet-Lab Validation Signal

The best measured wet-lab result in this snapshot is:

- `79.09%` viability for `304.7mM ectoin + 1.55M ethylene glycol`

That same region remains the model's top BO target, which is a useful consistency check between prediction and validation.

### DE-Based Bayesian Optimization (`05_bo_optimization`)

General BO summary for this snapshot: `results/bo_candidates_general_iteration_3_prior_mean_summary.txt`

| Rank | Formulation | Predicted viability |
|------|-------------|---------------------|
| 1 | 304.7mM ectoin + 1.55M ethylene glycol | 78.2% ± 14.2% |
| 2 | 295.3mM ectoin + 1.67M ethylene glycol | 77.0% ± 15.2% |
| 3 | 314.5mM ectoin + 1.55M ethylene glycol | 77.0% ± 14.5% |
| 4 | 280.0mM ectoin + 1.48M ethylene glycol + 0.2% hyaluronic acid + 20.4µM sucrose | 75.5% ± 16.7% |
| 5 | 272.6mM ectoin + 1.46M ethylene glycol + 3.7% FBS + 0.2% methylcellulose | 75.1% ± 17.4% |

### Untested BO Candidates

After filtering out already validated formulations with `python filter_tested_candidates.py`, the leading untested BO candidates for iteration 3 are:

| Rank | Untested formulation | Predicted viability |
|------|----------------------|---------------------|
| 2 | 295.3mM ectoin + 1.67M ethylene glycol | 77.0% ± 15.2% |
| 3 | 314.5mM ectoin + 1.55M ethylene glycol | 77.0% ± 14.5% |
| 4 | 280.0mM ectoin + 1.48M ethylene glycol + 0.2% hyaluronic acid + 20.4µM sucrose | 75.5% ± 16.7% |

The filtered outputs are written to `Untested/Iteration X/`, where `X` is the iteration you choose or the highest-numbered available candidate iteration if you press Enter.

See `results/` for full candidate lists.

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

For detailed interpretation and additional visualizations, see [`src/06_explainability/README.md`](src/06_explainability/README.md).

---

## Project Structure

```
├── data/
│   ├── raw/                    # Original literature data
│   ├── processed/              # Parsed formulations + legacy prior-mean evaluation mirror
│   └── validation/             # Wet lab results template
├── models/                     # Active model mirror + per-iteration checkpoints + observed context
├── results/                    # Optimized candidates + explainability graphs
└── src/
    ├── 01_data_parsing/        # Parse CSV, normalize units, merge synonyms
    ├── 02_model_training/      # Train GP regression model (Matérn kernel)
    ├── 03_optimization/        # Random sampling + GP prediction (fast)
    ├── 04_validation_loop/     # Integrate wet lab feedback, retrain model
    ├── 05_bo_optimization/     # Proper BO with Differential Evolution
    └── 06_explainability/      # Generate SHAP and explainability plots
```

## Module Descriptions

| Module | Method | Best For |
|--------|--------|----------|
| `01_data_parsing` | Data Parsing & Normalization | Preparing clean, structured training data from raw literature |
| `02_model_training` | Gaussian Process Regression (Matérn Kernel) | Learning the viability landscape from limited data |
| `03_optimization` | Random sampling, iteration-aware model loading | Quick generation, metadata repair when active model state is inconsistent |
| `04_validation_loop` | Three update strategies + iteration checkpointing | Closing the active learning loop with wet lab feedback |
| `05_bo_optimization` | Differential Evolution, wet-lab-aware BO context, shared iteration-aware model loading | Exploiting validated winners while proposing nearby informative variants |
| `06_explainability` | SHAP, PDPs, Interaction Contours, shared iteration-aware model loading | Understanding model drivers and ensuring trust |

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
