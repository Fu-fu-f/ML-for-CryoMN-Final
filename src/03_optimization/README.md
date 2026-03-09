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

- **Model registry**: `models/model_metadata.json` + `data/validation/iteration_history.json`
- **Iteration artifacts**: `models/iteration_*`
- **Data**: `data/processed/parsed_formulations.csv`

The script now uses the shared active-model resolver that is also used by `05_bo_optimization` and `06_explainability`. It validates the active model against both root metadata and the recorded iteration history:
- If the latest recorded iteration and `models/model_metadata.json` agree, `03` loads that iteration's artifacts directly.
- If metadata is missing, malformed, or points at the wrong iteration, `03` prompts for an iteration number.
- If you choose a valid iteration during conflict recovery, `03` overwrites `models/model_metadata.json` to repair the conflict and explicitly notifies you before and after doing so.
- If metadata says the model is composite but the composite artifacts are missing, the script stops. It does **not** fall back to the standard GP automatically.

## Output

- `results/candidates_general.csv` - Candidates with ≤5% DMSO
- `results/candidates_dmso_free.csv` - Low-DMSO candidates (`<0.5%` DMSO)
- `*_summary.txt` - Human-readable summaries

## Algorithm

1. Validate the active iteration using `models/model_metadata.json`, `iteration_history.json`, and `models/iteration_*`
2. Load the exact artifacts for the selected iteration
3. Generate large pool of random formulations (50× target count)
4. Filter by constraints (max DMSO, max ingredients)
5. Use model to predict viability for each candidate
6. Rank by predicted viability (highest mean)
7. Select top-N candidates

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
