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

> **These scripts will overwrite the current model in `models/`.** To ensure you can roll back if needed, **always stage and commit your changes before running an update.**

```bash
# 1. Stage and commit BEFORE updating the model
cd "/path/to/project"
git add -A
git commit -m "pre-update: save state before iteration N"

# 2. Now run the update script
python src/04_validation_loop/update_model_weighted_prior.py
```

**To rollback** if the updated model is unsatisfactory:

```bash
# Undo all changes since the last commit
git checkout -- models/
# Remove any new files created by the update
git clean -fd models/
# Reset the iteration history
git checkout -- data/validation/iteration_history.json
```

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

## Output

- Updated model in `models/iteration_N_<method>/`
- Main model updated in `models/`
- **Evaluation data** in `data/processed/evaluation_data.csv` (literature + wet lab with weights)
- Iteration history in `data/validation/iteration_history.json`

The evaluation data CSV includes a `weight` column (1.0 for literature, 50.0 for wet lab) and a `source` column. This file is used by the explainability script to compute weighted feature importance.

## Iteration Tracking

Each iteration is logged with:
- Timestamp
- Number of validation samples
- Validation RMSE
- Weighting method and parameters
