# Step 4: Validation Loop

## Overview

This module integrates wet lab validation results to iteratively refine the GP model. It includes three scripts with different approaches for incorporating validation data.

## Scripts

| Script | Method | Best For |
|--------|--------|----------|
| `update_model.py` | Simple concatenation | Baseline (no weighting) |
| `update_model_weighted_simple.py` | Sample duplication (10x) | Quick experiments, first iterations |
| `update_model_weighted_prior.py` | Prior mean + correction | When literature has systematic bias |

## Workflow

```
┌─────────────────┐
│  Train Model    │  ← Literature data
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Optimize      │  → Candidate formulations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Wet Lab       │  → Validation results
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Update Model   │  ← Combined data (WEIGHTED)
└────────┬────────┘
         │
         └──────────→ (Repeat)
```

## Usage

### First Time Setup

```bash
cd "/path/to/project"
python src/04_validation_loop/update_model.py
```

This creates a validation template at `data/validation/validation_template.csv`.

### After Wet Lab Experiments

1. Copy template to `data/validation/validation_results.csv`
2. Fill in experimental viability values
3. Choose and run a script:

```bash
# Option 1: No weighting (original)
python src/04_validation_loop/update_model.py

# Option 2: Simple weighting (10x duplication)
python src/04_validation_loop/update_model_weighted_simple.py

# Option 3: Prior mean + correction
python src/04_validation_loop/update_model_weighted_prior.py
```

### ⚠️ Before Running Any Update Script

> **These scripts overwrite the active model in `models/`.** Run them on a branch or commit your current state first so you have a clean rollback point.

## Validation CSV Format

The CSV uses **full feature names** with `_M` (molar) or `_pct` (percentage) suffixes to match the model's feature names:

```csv
experiment_id,experiment_date,viability_measured,notes,acetamide_M,betaine_M,...,dmso_M,...,ethylene_glycol_M,fbs_pct,...,glycerol_M,...,hsa_pct,...,trehalose_M
EXP101,2026-02-04,21.01,"33.0mM DMSO + 2.07M ethylene glycol",0,0,...,0.033,...,2.07,0,...,0,...,0,...,0
EXP205,2026-02-11,63.31,"34.7% FBS + 2.35M glycerol + 6.0% HSA",0,0,...,0,...,0,34.7,...,2.35,...,6,...,0
```

**Notes**:
- Columns include all 34 ingredient features — set unused ingredients to `0`
- Molar concentrations (`_M`) are in **mol/L** (e.g., 33 mM DMSO → `0.033`)
- Percentage ingredients (`_pct`) are in **%** (e.g., 34.7% FBS → `34.7`)
- Use the `validation_template.csv` as a starting point to ensure all columns are present

## Weighting Approaches

### Option A: Sample Duplication (`update_model_weighted_simple.py`)

Each wet lab sample is duplicated 10x before combining with literature data.

**Configuration** (edit at top of script):
```python
VALIDATION_WEIGHT_MULTIPLIER = 10  # Increase for more wet lab influence
```

**Pros:**
- Simple and intuitive
- Works with standard GP
- Easy to tune

### Option B: Prior Mean + Correction (`update_model_weighted_prior.py`)

Uses literature GP as prior mean, wet lab GP models corrections.

**Configuration:**
```python
ALPHA_LITERATURE = 1.0   # Higher noise = less trusted
ALPHA_WETLAB = 0.02      # Lower noise = more trusted  (50x trust ratio)
```

**Pros:**
- Corrects systematic biases
- Meaningful uncertainty
- Works with very few samples

**Output:** Creates a `CompositeGP` model with both components.

## Model Selection Behavior

Each update script now stamps the trained iteration with explicit identity fields:
- `iteration`
- `iteration_dir`
- `model_method`
- `is_composite_model`

The same identity is also appended to `data/validation/iteration_history.json`.

Downstream scripts use the active metadata in different ways:

- `update_model.py` and `update_model_weighted_simple.py` mark the active model as **standard GP**
- `update_model_weighted_prior.py` marks the active model as **composite GP**
- `03_optimization`, `05_bo_optimization`, and `06_explainability` now share the same iteration-aware resolver
- the shared resolver validates metadata against iteration history and does **not** fall back automatically across model types
- each update script now writes `models/<iteration_dir>/observed_context.csv` and mirrors the active copy to `models/observed_context.csv`
- `03`, `05`, and `06` all load that observed context first and reconstruct it on demand if it is missing

Whenever an update script activates a newly trained iteration by replacing `models/model_metadata.json`, it prints a notice that the active metadata is being overwritten and identifies the target iteration/method.

## Output

- Updated model in `models/iteration_N_<method>/`
- Main model updated in `models/`
- **Observed context** in `models/iteration_N_<method>/observed_context.csv` and `models/observed_context.csv`
- **Legacy evaluation data** in `data/processed/evaluation_data.csv` (compatibility mirror written by `update_model_weighted_prior.py`)
- Iteration history in `data/validation/iteration_history.json`

The canonical observed context CSV includes a `context_weight` column (1.0 for literature, weighted values for wet lab), a `source` column, and iteration identity fields. `03`, `05`, and `06` use this file as the source of truth for the active iteration. The legacy evaluation data CSV keeps the older `weight` column for prior-mean compatibility only.

## Iteration Tracking

Each iteration is logged with:
- Timestamp
- Iteration number and iteration directory
- Model method and whether the model is composite
- Number of validation samples
- Wet-lab cross-validated RMSE (`validation_rmse`)
- Wet-lab in-sample RMSE (`wetlab_train_rmse` in model metadata)
- Weighting method and parameters
