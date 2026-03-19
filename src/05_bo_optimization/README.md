# Step 5: Bayesian Optimization with Differential Evolution

## Overview

This module performs **proper Bayesian optimization** using Differential Evolution (DE) to maximize the configured acquisition function. By default it uses **Upper Confidence Bound (UCB)**, seeds the search from the best observed formulations, and then uses batch-mode local penalization to generate diverse nearby candidates. DE population scoring is evaluated in batches so GP prediction, acquisition, and penalty calculations run over each generation as a vectorized pass rather than point-by-point.

The key distinction in this pipeline is that `05` owns the **BO search**, while `07` owns the explicit **wet-lab exploit/explore policy**. `05` uses UCB during search, but the exported candidate files remain exploit-oriented and are intended to feed downstream selection rather than serve as the final exploration policy by themselves.

## Usage

```bash
cd "/path/to/project"
python src/05_bo_optimization/bo_optimizer.py
```

> [!CAUTION]
> `05_bo_optimization` can take a while to finish because it runs repeated Differential Evolution searches for both the general and low-DMSO candidate sets. The script prints live DE-search status while it is running, including candidate count, DE attempt count, and elapsed time.

## Input

- **Model registry**: `models/model_metadata.json` + `data/validation/iteration_history.json`
- **Iteration artifacts**: `models/iteration_*`
- **Observed context**: `models/<iteration_dir>/observed_context.csv` when available
- **Fallback inputs**: `data/processed/parsed_formulations.csv` + `data/validation/validation_results.csv`

This script uses the same active-model resolver as `03_optimization`:
- If `models/model_metadata.json` matches a recorded iteration, `05` loads that iteration's artifacts directly.
- If metadata is missing, malformed, or points at the wrong iteration, `05` prompts for an iteration number.
- If you choose a valid iteration during conflict recovery, `05` overwrites `models/model_metadata.json` to repair the conflict and explicitly notifies you before and after doing so.
- If metadata says the model is composite but the composite artifacts are missing, the script stops. It does **not** fall back to the standard GP automatically.
- For BO context, `05` loads the same observed context used by `03` and `06`, and reconstructs it on demand if the artifact is missing.

## Output

- `results/bo_candidates_general_<iteration_tag>.csv` - Candidates with ≤5% DMSO
- `results/bo_candidates_dmso_free_<iteration_tag>.csv` - Low-DMSO candidates (`<0.5%` DMSO)
- `*_summary.txt` - Human-readable summaries saved alongside the CSVs

Before candidates are scored and exported, `05` applies a practical
concentration floor to every feature:

- `_pct` values `<0.1%` are zeroed
- `_M` values `<0.001 M` (`<1.0 mM`) are zeroed

This keeps trace ingredients from surviving into candidate identity, ingredient
counts, and rendered formulation text.

These BO outputs are also the BO source pool consumed by
`src/07_next_formulations/next_formulations.py`, which builds the final
20-formulation wet-lab batch by combining:

- 8 exploitation picks from these `05` BO candidate files
- 12 exploration/calibration probes assembled from local-rank probes, blind-spot probes, and BO fallback if needed

`<iteration_tag>` comes from the resolved active model identity, for example:
- `iteration_1`
- `iteration_3_weighted_simple`
- `iteration_5_prior_mean`

## How It Works

### Algorithm

1. Validate the active iteration using `models/model_metadata.json`, `iteration_history.json`, and `models/iteration_*`
2. Load the exact artifacts for the selected iteration
3. Build the BO context from the active observed context (literature + wet-lab rows with `context_weight`)
4. Compute `y_best` from model predictions on the combined observed set
5. Seed the candidate pool with the best observed formulations under the active model
6. For each remaining candidate (sequentially):
   - Run Differential Evolution to find `x* = argmax(UCB(x) - penalty(x))`
   - DE starts from warm starts around top observed formulations instead of a blind search only
   - Each DE generation is scored as a batch, so model inference and penalty evaluation are applied to the full population together
   - the practical concentration floor is applied before sparsification and prediction
   - **Batch diversity**: Gaussian penalty repels DE away from previously found candidates
   - Constraint violations (DMSO, ingredient count, distance from observed support) are penalized
   - Exact duplicates are skipped
7. Recalculate pure UCB (without penalty) for accurate reporting
8. Rank candidates by predicted viability for exploit-oriented export
9. Export with predictions, uncertainty estimates, and acquisition values for downstream policy use

### Batch Diversity (Local Penalization)

To prevent all candidates from converging to the same optimum, each DE run adds a Gaussian repulsion centered on previously found candidates:

```
penalty(x) = Σ_i  strength · exp(-0.5 · ||x - x_i||² / r²)
```

Where `strength` and `r` (radius) control how strongly candidates repel each other. This ensures each new candidate explores a different region of formulation space.

### Upper Confidence Bound (UCB)

This optimizer uses the **Upper Confidence Bound (UCB)** acquisition function rather than Expected Improvement (EI). The BO part is the search itself: DE repeatedly optimizes UCB over the constrained formulation space to discover the candidate set before the final export ordering is applied.

```
UCB(x) = μ(x) + κ · σ(x)
```

Where:
- `μ(x)` = GP predicted mean
- `σ(x)` = GP predicted uncertainty
- `κ` (kappa) = Exploration weight

**Why UCB instead of EI?**
In high-dimensional spaces (e.g., 21 ingredients) with limited data, almost the entire formulation space is "out-of-distribution" (the void). In the void, the model reverts to its prior mean (~27.5%) with maximum uncertainty (~24%). 

Because EI mathematically rewards pure uncertainty, an EI-driven optimizer will actively dive into the flat void rather than staying near known good recipes. UCB (with a tuned `kappa`) places proportional weight on the *predicted mean*, anchoring the optimizer to known high-performing peaks while still exploring slightly uncertain edges.

### Wet-Lab Exploitation

When wet-lab results exist, `05` keeps them in the observed context with explicit weights instead of treating them as unrelated downstream artifacts:

- `y_best` can come from a validated wet-lab winner
- DE is warm-started from the best observed formulations, breaking prediction ties in favor of heavier weighted rows
- support radius and ingredient-count priors are computed on unique formulations with analytic `context_weight`, so duplicate weighting from `04` does not collapse BO geometry

This matters for narrow peaks such as the validated ectoin + ethylene glycol region, which can be missed by a blind global search in a sparse 21-dimensional space.

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `acquisition` | `ucb` | Acquisition function to use (`ucb` or `ei`) |
| `max_ingredients` | `None` | Infer max non-zero ingredients from observed support; optional manual override |
| `max_dmso_percent` | 5.0 | Max DMSO (general), 0.5% (low-DMSO) |
| `n_candidates` | 20 | Number of diverse candidates to generate |
| `kappa` | 0.5 | UCB exploration parameter |
| `de_maxiter` | 100 | DE iterations per candidate |
| `de_popsize` | 15 | DE population size |
| `diversity_penalty` | 5.0 | Strength of batch diversity repulsion |
| `diversity_radius` | 0.05 | Narrow radius; forces variations *around* the peak instead of pushing candidates completely into the void |
| `sparsity_penalty` | 0.35 | Mild preference for simpler formulations on flat plateaus |
| `support_penalty` | 4.0 | Penalizes candidates that move too far from observed support |
| `support_radius_scale` | 1.25 | Slack around the observed nearest-neighbor radius |

## Comparison: Random Sampling vs DE-based BO

| Aspect | `03_optimization` | `05_bo_optimization` |
|--------|-------------------|----------------------|
| **Search** | Random sampling | Differential Evolution |
| **Acquisition** | None | Maximizes UCB by default (EI optional) |
| **Exploration ownership** | None | BO search uses uncertainty; `07` owns explicit wet-lab exploration policy |
| **Diversity** | Naturally diverse (random) | Batch-mode penalization |
| **Observed context** | Combined literature + wet-lab rows | Combined literature + wet-lab rows with analytic BO weights |
| **Speed** | Fast | Slower overall, but DE population scoring is batched/vectorized |
| **Quality** | May miss optima | Finds acquisition maxima while staying closer to validated support |

## When to Use Which?

- **`03_optimization`**: Quick candidate generation, initial exploration, when speed matters
- **`05_bo_optimization`**: Serious optimization, when you want a BO-generated candidate pool that preserves the best validated recipes and explores high-value local variants around them
- **`07_next_formulations`**: After `05`, when you need the actual wet-lab batch recommendation with a strict 8 exploit / 12 explore split, smaller-batch subset recommendations, and full input/output validation
