# Step 3: Candidate Generation via Random Sampling

## Overview

This module generates candidate cryoprotective formulations using **random sampling + GP prediction**. It provides a fast way to explore the formulation space, though it uses pure exploitation (no exploration-exploitation balance).

> **Note**: For proper Bayesian optimization with acquisition-guided search, see [`05_bo_optimization`](../05_bo_optimization/README.md).

## Usage

```bash
cd "/path/to/project"
python src/03_optimization/optimize_formulation.py
```

## Input

- **Model**: `models/composite_model.pkl` (preferred) or `models/gp_model.pkl` (fallback)
- **Data**: `data/processed/parsed_formulations.csv`

The script reads `models/model_metadata.json` to decide whether the active model is composite or standard, then prints its selection:
- `>>> Using COMPOSITE model (literature prior + wet lab correction)` — if `composite_model.pkl` is found. This model is specifically created by running `04_validation_loop/update_model_weighted_prior.py`.
- `>>> Using STANDARD GP model (literature-only)` — if metadata marks the active model as standard. Stale composite artifacts are ignored automatically.

## Output

- `results/candidates_general.csv` - Candidates with ≤5% DMSO
- `results/candidates_dmso_free.csv` - Low-DMSO candidates (`<0.5%` DMSO)
- `*_summary.txt` - Human-readable summaries

## Algorithm

1. Load the active model selected by `models/model_metadata.json`
2. Generate large pool of random formulations (50× target count)
3. Filter by constraints (max DMSO, max ingredients)
4. Use model to predict viability for each candidate
5. Rank by predicted viability (highest mean)
6. Select top-N candidates

### Constraints

| Constraint | Value |
|------------|-------|
| Max DMSO | 5% (general), 0.5% (low-DMSO) |
| Max ingredients | 10 |
| Min viability | 70% (target) |

## Comparison with Proper BO

| Aspect | This Module (03) | Proper BO (05) |
|--------|------------------|----------------|
| **Method** | Random sampling | Differential Evolution |
| **Selection** | Highest predicted mean | Highest UCB by default |
| **Exploration** | None (pure exploitation) | Balanced via uncertainty |
| **Diversity** | Naturally diverse (random) | Batch-mode penalization |
| **Speed** | Fast (~seconds) | Slower (~minutes) |
| **Best for** | Quick generation | Most informative experiments |

### Why the difference matters

- **This module** always suggests what the model thinks will work best *right now*
- **Proper BO** suggests what would be most *informative* to test, including uncertain regions that might reveal better formulations

## Output Format

The output CSV includes both molar and percentage-based features:

```csv
rank,predicted_viability,uncertainty,dmso_percent,n_ingredients,dmso_M,trehalose_M,fbs_pct,hsa_pct,...
1,85.2,12.3,0.0,5,0.0,0.5,20.0,0.0,...
```

**Column naming convention:**
- `{ingredient}_M` - Molar concentration
- `{ingredient}_pct` - Percentage concentration

## Programmatic Usage

Like the training module, this script is CLI-first. The numbered source layout means examples like `from src.03_optimization...` are not valid Python imports, so use `importlib.util` by file path or refactor reusable pieces into a conventional package if you need library-style access.
