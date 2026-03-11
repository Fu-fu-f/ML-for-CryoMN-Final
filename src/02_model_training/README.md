# Step 2: Model Training

## Overview

This module trains a Gaussian Process (GP) regression model to predict cell viability from cryoprotective formulation ingredients.

## Usage

```bash
cd "/path/to/project"
python src/02_model_training/train_gp_model.py
```

## Input

- **File**: `data/processed/parsed_formulations.csv`
- **Features**: Ingredient concentrations (both `_M` and `_pct` columns)
- **Target**: `viability_percent`

## Output

- **Model**: `models/gp_model.pkl`
- **Scaler**: `models/scaler.pkl`  
- **Metadata**: `models/model_metadata.json`
- **Feature Importance**: `models/feature_importance.csv`

## Model Details

### Kernel

Matérn kernel (ν=2.5) with:
- Constant kernel (amplitude scaling)
- White kernel (noise modeling)

### Training

- 5-fold cross-validation for model assessment
- StandardScaler for feature normalization
- Target normalization enabled

### Feature Selection

Features are automatically selected based on:
- Column suffix: `_M` (molar) or `_pct` (percentage)
- Minimum non-zero count: ≥3 formulations must contain the ingredient

### Metrics

| Metric | Description | Value |
|--------|-------------|--------------|
| CV RMSE | Cross-validation RMSE | 19.97 ± 2.85 |
| CV R² | Cross-validation R² | 0.279 ± 0.088 |
| Training RMSE | Final model RMSE | 14.31 |
| Training R² | Final model R² | 0.638 |

### Top Features (by importance)

| Feature | Importance |
|---------|------------|
| DMSO | 0.290 |
| HES | 0.135 |
| Trehalose | 0.125 |
| Glycerol | 0.098 |
| Sucrose | 0.091 |
| FBS | 0.086 |

## Programmatic Usage

This module is primarily intended to be run as a CLI script. Because the source folders are numbered (`src/02_model_training`, etc.), direct imports like `from src.02_model_training...` are not valid Python syntax. If you need programmatic access, load the file by path with `importlib.util` or move the reusable code into a conventional package layout.

## Model Stats

- **Active features**: 21 ingredients
- **Training samples**: 198 formulations
- **Feature types**: 14 molar + 7 percentage-based
