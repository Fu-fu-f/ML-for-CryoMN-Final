# CryoMN ML-Based Cryoprotective Solution Optimization

Machine learning pipeline for optimizing cryoprotective formulations for cryomicroneedle (CryoMN) technology.

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
```

## Active Model and Iterations

- `src/04_validation_loop/*` saves each retrained checkpoint under `models/iteration_*`, updates `data/validation/iteration_history.json`, and then replaces the active root metadata in `models/model_metadata.json` with an explicit overwrite notice.
- `src/03_optimization/optimize_formulation.py`, `src/05_bo_optimization/bo_optimizer.py`, and `src/06_explainability/explainability.py` now share the same iteration-aware resolver.
- If root metadata is missing or inconsistent, these entry points prompt for an iteration number, reject nonsensical choices, and repair `models/model_metadata.json` only after telling you it is overwriting the metadata.
- Composite iterations are strict: if metadata says composite, the shared resolver will not fall back to a standard GP automatically.
- `06_explainability` is also strict about composite data: if a composite model is active, `data/processed/evaluation_data.csv` must exist.

## Results

### Random Sampling (`03_optimization`)

| Category | Best Candidate | Predicted Viability |
|----------|----------------|---------------------|
| General (≤5% DMSO) | 1.83M EG + 52% FBS + 0.6% HES | 72.4% ± 23.6% |
| Low-DMSO (<0.5%) | 2.07M ethylene glycol | 81.6% ± 20.8% |

> **Key Finding**: The model predicts high ethylene glycol concentrations (~2M) are highly effective at very low or zero DMSO.

### DE-based BO (`05_bo_optimization`)

| Category | Best Candidate | Acquisition Score (UCB default) |
|----------|----------------|---------------------------------|
| General (≤5% DMSO) | 10-ingredient formulation | UCB = 0.842 |
| Low-DMSO (<0.5%) | 10-ingredient formulation | UCB = 0.840 |

> **Note**: DE-based BO prioritizes *informative* experiments (high uncertainty) over highest predicted mean.

See `results/` for full candidate lists.

---

## Model Explainability

Understanding which ingredients drive cell viability predictions is crucial for guiding wet lab experiments. The explainability module generates comprehensive visualizations:

### SHAP Importance

SHAP values reveal how each ingredient impacts individual predictions. High DMSO concentrations (pink dots) can have both positive and negative effects:

![SHAP Summary](results/explainability/shap_summary.png)

### Acquisition Landscape

The acquisition landscape defaults to **Upper Confidence Bound (UCB)**, highlighting the model's exploitation-exploration tradeoff:

![Acquisition Landscape](results/explainability/acquisition_landscape.png)

### Interaction Contours

Visualizing how pairs of top ingredients interact to affect viability:

![Interaction Contours](results/explainability/interaction_contours.png)

For detailed interpretation and additional visualizations, see [`src/06_explainability/README.md`](src/06_explainability/README.md).

---

## Project Structure

```
├── data/
│   ├── raw/                    # Original literature data
│   ├── processed/              # Parsed formulations + evaluation data (with weights)
│   └── validation/             # Wet lab results template
├── models/                     # Active model mirror + per-iteration checkpoints
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
| `05_bo_optimization` | Differential Evolution, shared iteration-aware model loading | Most informative experiments, exploration-exploitation balance |
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
