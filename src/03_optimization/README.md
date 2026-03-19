# Step 3: Random-Sampling Candidate Generation

## Overview

This module generates candidate cryoprotective formulations using **random sampling + GP prediction**. It is a fast baseline that samples broadly, filters by practical constraints, and ranks survivors by predicted viability.

> **Note**: `03` does not perform acquisition-guided exploration. For Bayesian optimization with acquisition-guided search, see [`05_bo_optimization`](../05_bo_optimization/README.md).

## Usage

```bash
cd "/path/to/project"
python src/03_optimization/optimize_formulation.py
```

## Input

- **Model registry**: `models/model_metadata.json` + `data/validation/iteration_history.json`
- **Iteration artifacts**: `models/iteration_*`
- **Observed context**: `models/<iteration_dir>/observed_context.csv` when available
- **Fallback inputs**: `data/processed/parsed_formulations.csv` + `data/validation/validation_results.csv`

The script uses the shared active-model resolver that is also used by `05_bo_optimization` and `06_evaluation_explainability`. It validates the active model against both root metadata and the recorded iteration history:
- If `models/model_metadata.json` matches a recorded iteration, `03` loads that iteration's artifacts directly.
- If metadata is missing, malformed, or points at the wrong iteration, `03` prompts for an iteration number.
- If you choose a valid iteration during conflict recovery, `03` overwrites `models/model_metadata.json` to repair the conflict and explicitly notifies you before and after doing so.
- If metadata says the model is composite but the composite artifacts are missing, the script stops. It does **not** fall back to the standard GP automatically.
- For observed data, `03` loads the same iteration-aware observed context used by `05` and `06`. If the artifact is missing, it reconstructs the context from literature + measured wet-lab rows without requiring any other downstream module to run first.

## Output

- `results/candidates_general_<iteration_tag>.csv` - Candidates with ≤5% DMSO
- `results/candidates_dmso_free_<iteration_tag>.csv` - Low-DMSO candidates (`<0.5%` DMSO)
- `*_summary.txt` - Human-readable summaries saved alongside the CSVs

Before candidates are scored and exported, `03` applies the same practical
concentration floor used elsewhere in the pipeline:

- `_pct` values `<0.1%` are zeroed
- `_M` values `<0.001 M` (`<1.0 mM`) are zeroed

This keeps trace ingredients from inflating `n_ingredients` or surviving into
candidate identity/output text.

`<iteration_tag>` comes from the resolved active model identity, for example:
- `iteration_1`
- `iteration_3_weighted_simple`
- `iteration_5_prior_mean`

## Algorithm

1. Validate the active iteration using `models/model_metadata.json`, `iteration_history.json`, and `models/iteration_*`
2. Load the exact artifacts for the selected iteration
3. Load the active observed context (literature + wet lab rows)
4. Generate large pool of random formulations (50× target count)
5. Apply the practical concentration floor to zero trace ingredients
6. Filter by constraints (max DMSO, max ingredients)
7. Use model to predict viability for each candidate
8. Rank by predicted viability (highest mean)
9. Select top-N candidates

### Constraints

| Constraint | Value |
|------------|-------|
| Max DMSO | 5% (general), 0.5% (low-DMSO) |
| Max ingredients | 10 |
| Min viability | 70% (target) |

## Comparison with BO Search

| Aspect | This Module (03) | Proper BO (05) |
|--------|------------------|----------------|
| **Method** | Random sampling | Differential Evolution |
| **Selection** | Highest predicted mean | UCB-guided search, then exploit-oriented export |
| **Exploration control** | None | Search uses uncertainty; `07` owns explicit wet-lab exploration policy |
| **Diversity** | Naturally diverse (random) | Batch-mode penalization |
| **Speed** | Fast (~seconds) | Slower (~minutes) |
| **Best for** | Quick baseline generation | Strong BO candidate pool for downstream batch design |

### Why the difference matters

- **This module** is a quick baseline: it samples candidates randomly and keeps the highest predicted means
- **`05`** performs acquisition-guided BO search to build a stronger candidate pool
- **`07`** is the explicit experiment-policy layer that turns the BO pool into an exploitation/exploration wet-lab slate

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
