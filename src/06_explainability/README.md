# Step 6: Model Explainability

## Overview

This module generates comprehensive visualizations to explain how the Gaussian Process model makes predictions for cryoprotective formulations. Understanding model behavior helps guide experimental design and builds trust in the optimization recommendations.

## Usage

```bash
cd "/path/to/project"
python src/06_explainability/explainability.py
```

## Input

**Model resolution:**
- `models/model_metadata.json`
- `data/validation/iteration_history.json`
- `models/iteration_*`

`06_explainability` uses the same iteration-aware resolver as `03_optimization` and `05_bo_optimization`. If the active metadata is missing or inconsistent, it prompts for an iteration number and repairs `models/model_metadata.json` with an explicit overwrite notice.

**Standard GP mode (literature-only):**
- `models/gp_model.pkl`, `models/scaler.pkl`
- `data/processed/parsed_formulations.csv`

**Composite GP mode (after running validation loop):**
- `models/composite_model.pkl` (used when the resolved active iteration is composite)
- `models/<iteration_dir>/observed_context.csv` (literature + wet lab rows with `context_weight`)

If the canonical observed-context artifact is missing, `06` reconstructs it from literature + validation inputs. For older prior-mean iterations, it can also read the legacy `data/processed/evaluation_data.csv` mirror.

Common:
- `models/feature_importance.csv` (optional seed file; regenerated at runtime if missing)

## Output

Artifacts are written to an iteration-specific directory:
- `results/explainability/<iteration_tag>/`

`<iteration_tag>` comes from the resolved active model identity, for example:
- `iteration_1`
- `iteration_3_weighted_simple`
- `iteration_5_prior_mean`

If no explicit iteration metadata exists, the fallback directory is:
- `results/explainability/active_model/`

| File | Description |
|------|-------------|
| `feature_importance.csv` | Recomputed permutation importance (weighted for composite model) |
| `feature_importance.png` | Horizontal bar chart of permutation-based feature importance |
| `shap_summary.png` | SHAP beeswarm plot showing individual feature impacts |
| `shap_importance.png` | SHAP-based feature importance ranking |
| `partial_dependence_plots.png` | PDPs showing how viability changes with each ingredient |
| `interaction_contours.png` | 2D contour plots of ingredient pair interactions |
| `acquisition_landscape.png` | Acquisition visualization (UCB by default; EI configurable) |
| `uncertainty_analysis.png` | GP uncertainty calibration and distribution analysis |

## Visualization Details

### 0. Feature Importance (recomputed)
Feature importance is always **recomputed at runtime** using permutation importance against the active model and observed-context dataset. When using the composite model, importance uses **weighted R²** where each wet lab data point counts according to `context_weight`, matching the active iteration's trust ratio. If `models/feature_importance.csv` is missing, the script rebuilds it instead of failing.

### 2. SHAP Analysis
Uses [SHAP (SHapley Additive exPlanations)](https://shap.readthedocs.io/) to explain individual predictions:
- **Summary plot**: Shows direction and magnitude of feature effects
- **Importance plot**: Ranks features by mean absolute SHAP value

> **Note**: Requires `shap>=0.42.0`. If not installed, this analysis is skipped.

### 3. Partial Dependence Plots (PDPs)
Shows the marginal effect of each ingredient on predicted viability:
- X-axis: Ingredient concentration (M or %)
- Y-axis: Predicted viability (%)
- Blue line: Mean prediction
- Shaded area: 95% confidence interval from GP uncertainty

### 4. 2D Interaction Contours
Visualizes how pairs of top ingredients interact:
- Color gradient: Predicted viability
- Contour lines: Viability iso-lines
- Helps identify synergistic or antagonistic ingredient combinations

### 5. Acquisition Function Landscape
Three-panel visualization for Bayesian optimization insight:
1. **GP Mean**: Where the model predicts high viability
2. **GP Uncertainty**: Where the model is uncertain (exploration opportunity)
3. **Acquisition Score**: UCB by default, with EI available as a configuration option

Red star marks the best observed formulation.

### 6. Uncertainty Analysis
Four-panel analysis of model confidence:
1. **Predicted vs Actual**: Scatter plot colored by uncertainty
2. **Uncertainty Distribution**: Histogram of prediction uncertainties
3. **Error vs Uncertainty**: Calibration check (should correlate positively)
4. **Uncertainty by Viability Range**: Where is the model most/least confident?

## Dependencies

```
matplotlib>=3.7.0
seaborn>=0.12.0
shap>=0.42.0  # Optional but recommended
scipy>=1.10.0
scikit-learn>=1.3.0
```

## Example Output

After running, you'll see:

```
================================================================================
CryoMN Model Explainability Analysis
================================================================================

📊 Loading model and data...
  Model loaded with 21 features
  Data loaded with 191 formulations

📈 Generating visualizations...

1️⃣ Feature Importance Bar Chart
  ✓ Feature importance chart saved: results/explainability/feature_importance.png

2️⃣ SHAP Values Analysis
  ✓ SHAP summary plot saved: results/explainability/shap_summary.png
  ✓ SHAP importance plot saved: results/explainability/shap_importance.png

...

================================================================================
✅ Explainability Analysis Complete!
================================================================================
```

## Interpretation Guide

### High Importance Features
- **DMSO**: Highest importance (0.29) - key cryoprotectant but toxic at high concentrations
- **HES, Trehalose, Sucrose**: Important sugars for cell membrane protection
- **Glycerol**: Classic CPA with high importance
- **FBS**: Percentage-based serum with protective effects

### Reading the PDPs
- Upward slope: Ingredient increases viability
- Downward slope: Ingredient decreases viability (or becomes toxic)
- Wide confidence interval: High uncertainty in that concentration range

### Using Acquisition Landscape
- High acquisition-score regions: Most informative next experiments
- High uncertainty + moderate mean: Exploration opportunities
- High mean + low uncertainty: Exploitation (known good regions)

## Feature Name Handling

The module automatically cleans feature names for display:
- `dmso_M` → `Dmso`
- `fbs_pct` → `Fbs`
- `hyaluronic_acid_pct` → `Hyaluronic Acid`

Both `_M` (molar) and `_pct` (percentage) suffixes are stripped for readability.
