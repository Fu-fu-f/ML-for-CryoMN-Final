# Step 5: Bayesian Optimization with Differential Evolution

## Overview

This module performs **proper Bayesian optimization** using Differential Evolution (DE) to maximize the configured acquisition function. By default it uses **Upper Confidence Bound (UCB)** and batch-mode local penalization to generate diverse candidates.

## Usage

```bash
cd "/path/to/project"
python src/05_bo_optimization/bo_optimizer.py
```

## Input

- **Model**: `models/composite_model.pkl` (preferred) or `models/gp_model.pkl` (fallback)
- **Data**: `data/processed/parsed_formulations.csv`

The script reads `models/model_metadata.json` to decide whether the active model is composite or standard, and prints its selection:
- `>>> Using COMPOSITE model (literature prior + wet lab correction)` — if `composite_model.pkl` is found. This model is specifically created by running `04_validation_loop/update_model_weighted_prior.py`.
- `>>> Using STANDARD GP model (literature-only)` — if metadata marks the active model as standard. Stale composite artifacts are ignored automatically.

## Output

- `results/bo_candidates_general.csv` - Candidates with ≤5% DMSO
- `results/bo_candidates_dmso_free.csv` - Low-DMSO candidates (`<0.5%` DMSO)
- `*_summary.txt` - Human-readable summaries

## How It Works

### Algorithm

1. Load the active model selected by `models/model_metadata.json`
2. Compute `y_best` from model predictions on observed data
3. For each candidate (sequentially):
   - Run Differential Evolution to find `x* = argmax(UCB(x) - penalty(x))`
   - DE explores the entire search space globally
   - **Batch diversity**: Gaussian penalty repels DE away from previously found candidates
   - Constraint violations (DMSO, ingredient count) are penalized
4. Recalculate pure UCB (without penalty) for accurate reporting
5. Rank candidates by predicted viability
6. Export with predictions and uncertainty estimates

### Batch Diversity (Local Penalization)

To prevent all candidates from converging to the same optimum, each DE run adds a Gaussian repulsion centered on previously found candidates:

```
penalty(x) = Σ_i  strength · exp(-0.5 · ||x - x_i||² / r²)
```

Where `strength` and `r` (radius) control how strongly candidates repel each other. This ensures each new candidate explores a different region of formulation space.

### Upper Confidence Bound (UCB)

This optimizer uses the **Upper Confidence Bound (UCB)** acquisition function rather than Expected Improvement (EI). 

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

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `acquisition` | `ucb` | Acquisition function to use (`ucb` or `ei`) |
| `max_ingredients` | 10 | Max non-zero ingredients per formulation |
| `max_dmso_percent` | 5.0 | Max DMSO (general), 0.5% (low-DMSO) |
| `n_candidates` | 20 | Number of diverse candidates to generate |
| `kappa` | 0.5 | UCB exploration parameter |
| `de_maxiter` | 100 | DE iterations per candidate |
| `de_popsize` | 15 | DE population size |
| `diversity_penalty` | 5.0 | Strength of batch diversity repulsion |
| `diversity_radius` | 0.05 | Narrow radius; forces variations *around* the peak instead of pushing candidates completely into the void |

## Comparison: Random Sampling vs DE-based BO

| Aspect | `03_optimization` | `05_bo_optimization` |
|--------|-------------------|----------------------|
| **Search** | Random sampling | Differential Evolution |
| **Acquisition** | Sorts by mean only | Maximizes UCB by default (EI optional) |
| **Exploration** | Pure exploitation | Balanced |
| **Diversity** | Naturally diverse (random) | Batch-mode penalization |
| **Speed** | Fast (~seconds) | Slower (~minutes) |
| **Quality** | May miss optima | Finds acquisition maxima |

## When to Use Which?

- **`03_optimization`**: Quick candidate generation, initial exploration, when speed matters
- **`05_bo_optimization`**: Serious optimization, when you want the most diverse and informative experiments
