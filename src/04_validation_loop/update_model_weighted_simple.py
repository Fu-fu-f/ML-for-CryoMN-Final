#!/usr/bin/env python3
"""
CryoMN Weighted Validation Loop - Simple Sample Duplication

Integrates wet lab validation results with higher weight than literature data.
Uses sample duplication to emphasize validation data during retraining.

Approach: Each wet lab sample is duplicated N times (configurable multiplier)
before combining with literature data for retraining.

Author: CryoMN ML Project
Date: 2026-01-27
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from typing import Dict, List, Tuple
from datetime import datetime

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from iteration_metadata import (
    WEIGHTED_SIMPLE_METHOD,
    activate_iteration_artifacts,
    append_iteration_history,
    derive_iteration_dir,
    load_iteration_history,
    stamp_model_metadata,
)
from observed_context import (
    build_observed_context_df,
    save_observed_context,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# How many times each validation sample counts relative to literature samples
# Higher values = more weight on wet lab data
# Recommended range: 5-20
VALIDATION_WEIGHT_MULTIPLIER = 10


# =============================================================================
# VALIDATION DATA HANDLING
# =============================================================================

def create_validation_template(feature_names: List[str], output_path: str):
    """
    Create a template CSV for entering wet lab validation results.
    
    Args:
        feature_names: List of ingredient feature names
        output_path: Path to save template
    """
    columns = ['experiment_id', 'experiment_date', 'viability_measured', 'notes']
    columns.extend(feature_names)
    
    template_df = pd.DataFrame(columns=columns)
    
    # Add example row
    template_df.loc[0] = ['EXP001', '2026-01-25', 85.5, 'Example entry'] + [0.0] * len(feature_names)
    
    template_df.to_csv(output_path, index=False)
    print(f"Validation template created: {output_path}")
    return template_df


def load_validation_results(validation_path: str, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load validated formulation results.
    
    Args:
        validation_path: Path to validation CSV
        feature_names: List of feature names (with _M suffix)
        
    Returns:
        Tuple of (X features, y viability)
    """
    df = pd.read_csv(validation_path)
    
    X_data = []
    y_data = []
    
    for _, row in df.iterrows():
        if pd.isna(row['viability_measured']):
            continue
        
        x = np.zeros(len(feature_names))
        for i, name in enumerate(feature_names):
            if name in row and pd.notna(row[name]):
                x[i] = float(row[name])
        
        X_data.append(x)
        y_data.append(float(row['viability_measured']))
    
    if len(X_data) == 0:
        return np.array([]).reshape(0, len(feature_names)), np.array([])
    
    return np.array(X_data), np.array(y_data)


# =============================================================================
# WEIGHTED MODEL UPDATE
# =============================================================================

def create_gp_model() -> GaussianProcessRegressor:
    """Create the standard GP used across update workflows."""
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(nu=2.5, length_scale_bounds=(1e-5, 1e5))
        + WhiteKernel(noise_level=1.0)
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        random_state=42,
        alpha=1e-6,
        normalize_y=True,
    )


def fit_weighted_gp(
    X_orig: np.ndarray,
    y_orig: np.ndarray,
    X_val_train: np.ndarray,
    y_val_train: np.ndarray,
    weight_multiplier: int,
) -> Tuple[GaussianProcessRegressor, StandardScaler, np.ndarray, np.ndarray]:
    """Fit the duplicated-sample GP on a wet-lab training subset."""
    X_val_weighted = np.repeat(X_val_train, weight_multiplier, axis=0)
    y_val_weighted = np.repeat(y_val_train, weight_multiplier)
    X_combined = np.vstack([X_orig, X_val_weighted])
    y_combined = np.concatenate([y_orig, y_val_weighted])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    gp = create_gp_model()
    gp.fit(X_scaled, y_combined)

    return gp, scaler, X_val_weighted, y_val_weighted


def compute_wetlab_cv_rmse(
    X_orig: np.ndarray,
    y_orig: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    weight_multiplier: int,
) -> float:
    """Estimate wet-lab generalization error by cross-validating over wet-lab rows."""
    if len(X_val) < 2:
        return float('nan')

    splitter = KFold(
        n_splits=min(5, len(X_val)), shuffle=True, random_state=42
    )
    y_pred = np.zeros(len(y_val))

    for train_idx, test_idx in splitter.split(X_val):
        gp_fold, scaler_fold, _, _ = fit_weighted_gp(
            X_orig, y_orig, X_val[train_idx], y_val[train_idx], weight_multiplier
        )
        y_pred[test_idx] = gp_fold.predict(scaler_fold.transform(X_val[test_idx]))

    return float(np.sqrt(np.mean((y_val - y_pred) ** 2)))

def update_model_weighted(
    original_model_dir: str, 
    validation_data: Tuple[np.ndarray, np.ndarray],
    original_data: Tuple[np.ndarray, np.ndarray], 
    output_dir: str,
    weight_multiplier: int = VALIDATION_WEIGHT_MULTIPLIER,
    iteration: int = 0,
    iteration_dir_name: str = '',
    model_method: str = WEIGHTED_SIMPLE_METHOD,
) -> Dict:
    """
    Update GP model with weighted validation data using sample duplication.
    
    Each validation sample is duplicated `weight_multiplier` times to give
    wet lab data more influence on the model.
    
    Args:
        original_model_dir: Path to original model directory
        validation_data: Tuple of (X_val, y_val) from wet lab
        original_data: Tuple of (X_orig, y_orig) from literature
        output_dir: Directory to save updated model
        weight_multiplier: How many times to duplicate each validation sample
        
    Returns:
        Dictionary with update statistics
    """
    X_val, y_val = validation_data
    X_orig, y_orig = original_data
    
    gp, scaler, X_val_weighted, y_val_weighted = fit_weighted_gp(
        X_orig, y_orig, X_val, y_val, weight_multiplier
    )
    X_combined = np.vstack([X_orig, X_val_weighted])
    y_combined = np.concatenate([y_orig, y_val_weighted])
    
    print(f"Original literature data: {len(X_orig)} samples")
    print(f"Validation data: {len(X_val)} samples")
    print(f"Validation data (after {weight_multiplier}x weighting): {len(X_val_weighted)} samples")
    print(f"Effective combined data: {len(X_combined)} samples")
    print(f"Effective validation weight: {len(X_val_weighted) / len(X_combined) * 100:.1f}% of training data")
    
    print("\nTraining weighted model...")
    
    # Measure both in-sample fit and a wet-lab cross-validated RMSE.
    X_val_scaled = scaler.transform(X_val)
    y_val_pred = gp.predict(X_val_scaled)
    train_rmse = float(np.sqrt(np.mean((y_val - y_val_pred) ** 2)))
    val_rmse = compute_wetlab_cv_rmse(
        X_orig, y_orig, X_val, y_val, weight_multiplier
    )
    
    print(f"Optimized kernel: {gp.kernel_}")
    print(f"Wet-lab train RMSE: {train_rmse:.2f}")
    if np.isnan(val_rmse):
        print("Wet-lab CV RMSE: N/A (need at least 2 wet-lab samples)")
    else:
        print(f"Wet-lab CV RMSE: {val_rmse:.2f}")
    
    # Save updated model
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'gp_model.pkl'), 'wb') as f:
        pickle.dump(gp, f)
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Load and update metadata
    metadata_path = os.path.join(original_model_dir, 'model_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    metadata['updated_at'] = datetime.now().isoformat()
    metadata['n_validation_samples'] = len(X_val)
    metadata['n_total_samples'] = len(X_combined)
    metadata['validation_rmse'] = val_rmse
    metadata['wetlab_train_rmse'] = train_rmse
    metadata['weighting_method'] = 'sample_duplication'
    metadata['weight_multiplier'] = weight_multiplier
    metadata = stamp_model_metadata(
        metadata,
        iteration=iteration,
        model_method=model_method,
        iteration_dir=iteration_dir_name,
        is_composite_model=False,
    )
    
    with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nWeighted model saved to: {output_dir}")
    
    return {
        'n_original': len(X_orig),
        'n_validation': len(X_val),
        'n_validation_weighted': len(X_val_weighted),
        'n_total': len(X_combined),
        'wetlab_train_rmse': train_rmse,
        'validation_rmse': val_rmse,
        'weight_multiplier': weight_multiplier,
    }


# =============================================================================
# ITERATION TRACKING
# =============================================================================

def get_iteration_number(project_dir: str) -> int:
    """Get current iteration number from history."""
    return len(load_iteration_history(project_dir))


def save_iteration(project_dir: str, iteration_data: Dict):
    """Save iteration information to history."""
    append_iteration_history(project_dir, iteration_data)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for weighted validation loop (sample duplication)."""
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    model_dir = os.path.join(project_root, 'models')
    data_dir = os.path.join(project_root, 'data')
    validation_dir = os.path.join(data_dir, 'validation')
    os.makedirs(validation_dir, exist_ok=True)
    
    print("=" * 80)
    print("CryoMN Weighted Validation Loop (Sample Duplication)")
    print(f"Weight Multiplier: {VALIDATION_WEIGHT_MULTIPLIER}x")
    print("=" * 80)
    
    # Load model metadata
    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    feature_names = metadata['feature_names']
    print(f"\nModel has {len(feature_names)} features")
    
    # Check for validation data
    validation_path = os.path.join(validation_dir, 'validation_results.csv')
    template_path = os.path.join(validation_dir, 'validation_template.csv')
    
    if not os.path.exists(template_path):
        print("\nCreating validation template...")
        create_validation_template(feature_names, template_path)
    
    if not os.path.exists(validation_path):
        print("\n" + "=" * 80)
        print("No validation results found.")
        print(f"\nTo add wet lab results:")
        print(f"  1. Copy: {template_path}")
        print(f"  2. To:   {validation_path}")
        print(f"  3. Fill in your experimental results")
        print(f"  4. Run this script again")
        print("=" * 80)
        return
    
    # Load validation data
    print(f"\nLoading validation results from: {validation_path}")
    X_val, y_val = load_validation_results(validation_path, feature_names)
    
    if len(X_val) == 0:
        print("No valid validation entries found. Please add results to the CSV.")
        return
    
    print(f"Found {len(X_val)} validation experiments")
    print(f"Viability range: {y_val.min():.1f}% - {y_val.max():.1f}%")
    
    # Load original training data
    parsed_path = os.path.join(data_dir, 'processed', 'parsed_formulations.csv')
    df_orig = pd.read_csv(parsed_path)
    df_orig = df_orig[df_orig['viability_percent'] <= 100].copy()
    
    X_orig = df_orig[feature_names].values
    y_orig = df_orig['viability_percent'].values
    
    # Get iteration number
    iteration = get_iteration_number(project_root) + 1
    print(f"\n--- Iteration {iteration} (Weighted: {VALIDATION_WEIGHT_MULTIPLIER}x) ---")
    
    # Update model with weighting
    model_method = WEIGHTED_SIMPLE_METHOD
    iteration_dir_name = derive_iteration_dir(iteration, model_method)
    updated_model_dir = os.path.join(model_dir, iteration_dir_name)
    stats = update_model_weighted(
        model_dir,
        (X_val, y_val),
        (X_orig, y_orig),
        updated_model_dir,
        weight_multiplier=VALIDATION_WEIGHT_MULTIPLIER,
        iteration=iteration,
        iteration_dir_name=iteration_dir_name,
        model_method=model_method,
    )

    observed_context_df = build_observed_context_df(
        feature_names=feature_names,
        X_literature=X_orig,
        y_literature=y_orig,
        X_wetlab=X_val,
        y_wetlab=y_val,
        model_method=model_method,
        iteration=iteration,
        iteration_dir=iteration_dir_name,
        wetlab_context_weight=float(VALIDATION_WEIGHT_MULTIPLIER),
    )
    save_observed_context(updated_model_dir, observed_context_df)
    
    # Also update main model directory
    print("\nUpdating main model...")
    activate_iteration_artifacts(
        updated_model_dir,
        model_dir,
        ['gp_model.pkl', 'scaler.pkl', 'model_metadata.json', 'observed_context.csv'],
        iteration=iteration,
        model_method=model_method,
        reason='activating a newly trained iteration',
    )
    
    # Save iteration history
    save_iteration(project_root, {
        'iteration': iteration,
        'method': model_method,
        'model_method': model_method,
        'iteration_dir': iteration_dir_name,
        'is_composite_model': False,
        'weight_multiplier': stats['weight_multiplier'],
        'n_validation_samples': stats['n_validation'],
        'n_validation_weighted': stats['n_validation_weighted'],
        'validation_rmse': stats['validation_rmse'],
    })
    
    print("\n" + "=" * 80)
    print("Weighted Validation Loop Complete!")
    print("=" * 80)
    print(f"\nMethod: Sample duplication ({VALIDATION_WEIGHT_MULTIPLIER}x weight)")
    print(f"Effective validation influence: {stats['n_validation_weighted']} / {stats['n_total']} samples")
    print(f"\nNext steps:")
    print(f"  1. Run optimization: python src/05_bo_optimization/bo_optimizer.py")
    print(f"  2. Test top candidates in wet lab")
    print(f"  3. Add results to: {validation_path}")
    print(f"  4. Run this script again for next iteration")


if __name__ == '__main__':
    main()
