#!/usr/bin/env python3
"""
CryoMN Weighted Validation Loop - Prior Mean with Tiered Noise

A sophisticated approach that uses the literature-trained GP as a prior mean
function, with wet lab data modeling corrections/deviations from literature.

Approach:
1. Literature GP provides baseline predictions (prior mean)
2. Wet lab GP models residuals (corrections to literature)
3. Final prediction = Literature baseline + Wet lab correction
4. Tiered noise: lower noise for wet lab data (higher trust)

This approach is ideal when:
- Literature data is directionally correct but has systematic bias
- You have limited wet lab samples (even 1-3 can be effective)
- You want meaningful uncertainty quantification

Author: CryoMN ML Project
Date: 2026-01-27
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import shutil
import sys
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONFIGURATION
# =============================================================================

# Noise levels for tiered uncertainty
# Lower noise = higher trust in data source
ALPHA_LITERATURE = 1.0     # Higher noise: literature data has more uncertainty
ALPHA_WETLAB = 0.02        # Lower noise: wet lab data is more trusted

# Noise ratio in the combined approach
NOISE_RATIO = ALPHA_LITERATURE / ALPHA_WETLAB  # 50x

# Make the module importable under a stable name even when executed as a script.
sys.modules.setdefault('update_model_weighted_prior', sys.modules[__name__])


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
# COMPOSITE GP MODEL (PRIOR MEAN + CORRECTION)
# =============================================================================

class CompositeGP:
    """
    A composite GP model that uses literature GP as prior mean
    and wet lab GP for corrections.
    
    Final prediction = literature_prediction + wetlab_correction
    
    This allows wet lab data to correct systematic biases in literature
    while preserving the structure learned from literature.
    """
    
    def __init__(self, gp_literature: GaussianProcessRegressor, 
                 gp_correction: GaussianProcessRegressor,
                 scaler_literature: StandardScaler,
                 scaler_correction: StandardScaler):
        """
        Initialize composite model.
        
        Args:
            gp_literature: GP trained on literature data
            gp_correction: GP trained on wet lab residuals
            scaler_literature: Scaler used for literature GP
            scaler_correction: Scaler used for correction GP
        """
        self.gp_literature = gp_literature
        self.gp_correction = gp_correction
        self.scaler_literature = scaler_literature
        self.scaler_correction = scaler_correction
    
    def predict(self, X: np.ndarray, return_std: bool = False):
        """
        Make predictions using composite model.
        
        Args:
            X: Feature matrix (unscaled)
            return_std: Whether to return uncertainty
            
        Returns:
            Predictions (and optionally std)
        """
        # Scale for each model
        X_lit_scaled = self.scaler_literature.transform(X)
        X_corr_scaled = self.scaler_correction.transform(X)
        
        if return_std:
            y_lit, std_lit = self.gp_literature.predict(X_lit_scaled, return_std=True)
            y_corr, std_corr = self.gp_correction.predict(X_corr_scaled, return_std=True)
            
            # Combined prediction
            y_pred = y_lit + y_corr
            
            # Combined uncertainty (sum of variances, then sqrt)
            std_combined = np.sqrt(std_lit**2 + std_corr**2)
            
            return y_pred, std_combined
        else:
            y_lit = self.gp_literature.predict(X_lit_scaled)
            y_corr = self.gp_correction.predict(X_corr_scaled)
            
            return y_lit + y_corr
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score on data."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


CompositeGP.__module__ = 'update_model_weighted_prior'


def create_gp_model(alpha: float, noise_level: float = 1.0) -> GaussianProcessRegressor:
    """Create a GP configured for either literature or correction fitting."""
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(nu=2.5, length_scale_bounds=(1e-5, 1e5))
        + WhiteKernel(noise_level=noise_level)
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        random_state=42,
        alpha=alpha,
        normalize_y=True,
    )


def fit_literature_model(
    X_orig: np.ndarray, y_orig: np.ndarray, alpha_literature: float
) -> Tuple[GaussianProcessRegressor, StandardScaler]:
    """Fit the literature-only GP once."""
    scaler_lit = StandardScaler()
    X_orig_scaled = scaler_lit.fit_transform(X_orig)
    gp_literature = create_gp_model(alpha_literature, noise_level=1.0)
    gp_literature.fit(X_orig_scaled, y_orig)
    return gp_literature, scaler_lit


def fit_correction_model(
    X_val_train: np.ndarray, residuals_train: np.ndarray, alpha_wetlab: float
) -> Tuple[GaussianProcessRegressor, StandardScaler]:
    """Fit the wet-lab residual model."""
    scaler_corr = StandardScaler()
    X_val_scaled = scaler_corr.fit_transform(X_val_train)
    gp_correction = create_gp_model(alpha_wetlab, noise_level=0.1)
    gp_correction.fit(X_val_scaled, residuals_train)
    return gp_correction, scaler_corr


def compute_wetlab_cv_rmse(
    gp_literature: GaussianProcessRegressor,
    scaler_lit: StandardScaler,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alpha_wetlab: float,
) -> float:
    """Estimate wet-lab generalization error by cross-validating the correction GP."""
    if len(X_val) < 2:
        return float('nan')

    y_lit_pred = gp_literature.predict(scaler_lit.transform(X_val))
    residuals = y_val - y_lit_pred
    splitter = KFold(
        n_splits=min(5, len(X_val)), shuffle=True, random_state=42
    )
    y_pred = np.zeros(len(y_val))

    for train_idx, test_idx in splitter.split(X_val):
        gp_corr_fold, scaler_corr_fold = fit_correction_model(
            X_val[train_idx], residuals[train_idx], alpha_wetlab
        )
        composite_fold = CompositeGP(
            gp_literature=gp_literature,
            gp_correction=gp_corr_fold,
            scaler_literature=scaler_lit,
            scaler_correction=scaler_corr_fold,
        )
        y_pred[test_idx] = composite_fold.predict(X_val[test_idx])

    return float(np.sqrt(np.mean((y_val - y_pred) ** 2)))


def save_composite_model(composite_model: CompositeGP, output_dir: str, metadata: Dict):
    """
    Save composite model components.
    
    Args:
        composite_model: The CompositeGP instance
        output_dir: Directory to save model
        metadata: Model metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save literature model components
    with open(os.path.join(output_dir, 'gp_literature.pkl'), 'wb') as f:
        pickle.dump(composite_model.gp_literature, f)
    with open(os.path.join(output_dir, 'scaler_literature.pkl'), 'wb') as f:
        pickle.dump(composite_model.scaler_literature, f)
    
    # Save correction model components
    with open(os.path.join(output_dir, 'gp_correction.pkl'), 'wb') as f:
        pickle.dump(composite_model.gp_correction, f)
    with open(os.path.join(output_dir, 'scaler_correction.pkl'), 'wb') as f:
        pickle.dump(composite_model.scaler_correction, f)
    
    # Save composite model wrapper
    with open(os.path.join(output_dir, 'composite_model.pkl'), 'wb') as f:
        pickle.dump(composite_model, f)
    
    # Save a standard gp_model.pkl for compatibility with other modules
    # (This is the literature model, which modules can fall back to)
    with open(os.path.join(output_dir, 'gp_model.pkl'), 'wb') as f:
        pickle.dump(composite_model.gp_literature, f)
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(composite_model.scaler_literature, f)
    
    # Save metadata
    with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Composite model saved to: {output_dir}")


def load_composite_model(model_dir: str) -> CompositeGP:
    """
    Load composite model from directory.
    
    Args:
        model_dir: Directory containing saved model
        
    Returns:
        CompositeGP instance
    """
    with open(os.path.join(model_dir, 'composite_model.pkl'), 'rb') as f:
        return pickle.load(f)


# =============================================================================
# WEIGHTED MODEL UPDATE
# =============================================================================

def update_model_with_prior_mean(
    original_model_dir: str, 
    validation_data: Tuple[np.ndarray, np.ndarray],
    original_data: Tuple[np.ndarray, np.ndarray], 
    output_dir: str,
    alpha_literature: float = ALPHA_LITERATURE,
    alpha_wetlab: float = ALPHA_WETLAB
) -> Dict:
    """
    Update model using literature as prior mean + wet lab correction.
    
    Args:
        original_model_dir: Path to original (literature-trained) model
        validation_data: Tuple of (X_val, y_val) from wet lab
        original_data: Tuple of (X_orig, y_orig) from literature
        output_dir: Directory to save updated model
        alpha_literature: Noise level for literature data
        alpha_wetlab: Noise level for wet lab data
        
    Returns:
        Dictionary with update statistics
    """
    X_val, y_val = validation_data
    X_orig, y_orig = original_data
    
    print(f"Literature data: {len(X_orig)} samples (α={alpha_literature})")
    print(f"Wet lab data: {len(X_val)} samples (α={alpha_wetlab})")
    print(f"Noise ratio (literature/wetlab): {alpha_literature/alpha_wetlab:.1f}x")
    
    # =========================================================================
    # Stage 1: Literature model (already trained, or retrain for consistency)
    # =========================================================================
    print("\n--- Stage 1: Literature Model (Prior Mean) ---")
    
    print("Training literature model...")
    gp_literature, scaler_lit = fit_literature_model(
        X_orig, y_orig, alpha_literature
    )
    print(f"Literature kernel: {gp_literature.kernel_}")
    
    # Get literature predictions at validation points
    X_val_lit_scaled = scaler_lit.transform(X_val)
    y_lit_pred = gp_literature.predict(X_val_lit_scaled)
    
    # Calculate residuals (what literature got wrong)
    residuals = y_val - y_lit_pred
    print(f"\nLiterature predictions at wet lab points:")
    print(f"  Mean residual: {np.mean(residuals):+.2f}%")
    print(f"  Std residual: {np.std(residuals):.2f}%")
    
    # =========================================================================
    # Stage 2: Correction model (learns wet lab deviations from literature)
    # =========================================================================
    print("\n--- Stage 2: Correction Model (Wet Lab Residuals) ---")
    
    print("Training correction model on residuals...")
    gp_correction, scaler_corr = fit_correction_model(
        X_val, residuals, alpha_wetlab
    )
    print(f"Correction kernel: {gp_correction.kernel_}")
    
    # =========================================================================
    # Create Composite Model
    # =========================================================================
    print("\n--- Creating Composite Model ---")
    
    composite_model = CompositeGP(
        gp_literature=gp_literature,
        gp_correction=gp_correction,
        scaler_literature=scaler_lit,
        scaler_correction=scaler_corr
    )
    
    # Evaluate composite model on wet-lab data and estimate held-out RMSE.
    y_composite_pred = composite_model.predict(X_val)
    train_rmse = float(np.sqrt(np.mean((y_val - y_composite_pred) ** 2)))
    val_rmse = compute_wetlab_cv_rmse(
        gp_literature, scaler_lit, X_val, y_val, alpha_wetlab
    )
    
    # Also check literature-only RMSE for comparison
    lit_rmse = np.sqrt(np.mean((y_val - y_lit_pred) ** 2))
    
    print(f"\nValidation Performance:")
    print(f"  Literature-only RMSE: {lit_rmse:.2f}")
    print(f"  Composite train RMSE: {train_rmse:.2f}")
    if np.isnan(val_rmse):
        print("  Wet-lab CV RMSE: N/A (need at least 2 wet-lab samples)")
    else:
        print(f"  Wet-lab CV RMSE: {val_rmse:.2f}")
        print(f"  Improvement: {lit_rmse - val_rmse:.2f} ({(1 - val_rmse/lit_rmse)*100:.1f}%)")
    
    # =========================================================================
    # Save Model
    # =========================================================================
    
    # Load original metadata and update
    metadata_path = os.path.join(original_model_dir, 'model_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    metadata['updated_at'] = datetime.now().isoformat()
    metadata['n_validation_samples'] = len(X_val)
    metadata['n_literature_samples'] = len(X_orig)
    metadata['validation_rmse'] = val_rmse
    metadata['wetlab_train_rmse'] = train_rmse
    metadata['literature_only_rmse'] = lit_rmse
    metadata['weighting_method'] = 'prior_mean_correction'
    metadata['alpha_literature'] = alpha_literature
    metadata['alpha_wetlab'] = alpha_wetlab
    metadata['noise_ratio'] = alpha_literature / alpha_wetlab
    metadata['is_composite_model'] = True
    
    save_composite_model(composite_model, output_dir, metadata)
    
    return {
        'n_literature': len(X_orig),
        'n_validation': len(X_val),
        'literature_rmse': lit_rmse,
        'wetlab_train_rmse': train_rmse,
        'validation_rmse': val_rmse,
        'improvement': lit_rmse - val_rmse,
        'noise_ratio': alpha_literature / alpha_wetlab,
        'mean_residual': np.mean(residuals),
    }


# =============================================================================
# ITERATION TRACKING
# =============================================================================

def get_iteration_number(project_dir: str) -> int:
    """Get current iteration number from history."""
    history_path = os.path.join(project_dir, 'data', 'validation', 'iteration_history.json')
    
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        return len(history.get('iterations', []))
    return 0


def save_iteration(project_dir: str, iteration_data: Dict):
    """Save iteration information to history."""
    history_path = os.path.join(project_dir, 'data', 'validation', 'iteration_history.json')
    
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
    else:
        history = {'iterations': []}
    
    iteration_data['timestamp'] = datetime.now().isoformat()
    history['iterations'].append(iteration_data)
    
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Iteration {len(history['iterations'])} logged")


# =============================================================================
# EVALUATION DATA EXPORT
# =============================================================================

def save_evaluation_data(project_root: str, feature_names: List[str],
                         X_orig: np.ndarray, y_orig: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         noise_ratio: float):
    """
    Save combined evaluation data (literature + wet lab) with weights.
    
    This file is used by the explainability script to compute
    weighted feature importance that reflects both data sources.
    
    Args:
        project_root: Project root directory
        feature_names: List of feature names
        X_orig: Literature feature matrix
        y_orig: Literature target values
        X_val: Wet lab feature matrix
        y_val: Wet lab target values
        noise_ratio: Trust multiplier for wet lab data
    """
    # Build literature rows
    df_lit = pd.DataFrame(X_orig, columns=feature_names)
    df_lit['viability_percent'] = y_orig
    df_lit['weight'] = 1.0
    df_lit['source'] = 'literature'
    
    # Build wet lab rows
    df_wet = pd.DataFrame(X_val, columns=feature_names)
    df_wet['viability_percent'] = y_val
    df_wet['weight'] = noise_ratio
    df_wet['source'] = 'wetlab'
    
    # Concatenate
    df_eval = pd.concat([df_lit, df_wet], ignore_index=True)
    
    # Save
    eval_path = os.path.join(project_root, 'data', 'processed', 'evaluation_data.csv')
    df_eval.to_csv(eval_path, index=False)
    
    print(f"\n📊 Evaluation data saved: {eval_path}")
    print(f"  Literature rows: {len(df_lit)} (weight=1.0)")
    print(f"  Wet lab rows:    {len(df_wet)} (weight={noise_ratio:.0f})")
    print(f"  Total:           {len(df_eval)}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for prior mean + correction validation loop."""
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    model_dir = os.path.join(project_root, 'models')
    data_dir = os.path.join(project_root, 'data')
    validation_dir = os.path.join(data_dir, 'validation')
    os.makedirs(validation_dir, exist_ok=True)
    
    print("=" * 80)
    print("CryoMN Weighted Validation Loop (Prior Mean + Correction)")
    print(f"Literature α: {ALPHA_LITERATURE} | Wet Lab α: {ALPHA_WETLAB}")
    print(f"Trust Ratio: Wet lab is {NOISE_RATIO:.0f}x more trusted than literature")
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
    print(f"\n--- Iteration {iteration} (Prior Mean + Correction) ---")
    
    # Update model with prior mean approach
    updated_model_dir = os.path.join(model_dir, f'iteration_{iteration}_prior_mean')
    stats = update_model_with_prior_mean(
        model_dir,
        (X_val, y_val),
        (X_orig, y_orig),
        updated_model_dir,
        alpha_literature=ALPHA_LITERATURE,
        alpha_wetlab=ALPHA_WETLAB
    )
    
    # Update main model directory
    # Note: We copy the composite model but also keep backward-compatible files
    print("\nUpdating main model directory...")
    for filename in ['gp_model.pkl', 'scaler.pkl', 'model_metadata.json', 
                     'composite_model.pkl', 'gp_literature.pkl', 'scaler_literature.pkl',
                     'gp_correction.pkl', 'scaler_correction.pkl']:
        src = os.path.join(updated_model_dir, filename)
        dst = os.path.join(model_dir, filename)
        if os.path.exists(src):
            shutil.copy(src, dst)
    
    # Save evaluation data for explainability
    save_evaluation_data(
        project_root, feature_names,
        X_orig, y_orig,
        X_val, y_val,
        NOISE_RATIO
    )
    
    # Save iteration history
    save_iteration(project_root, {
        'iteration': iteration,
        'method': 'prior_mean_correction',
        'n_validation_samples': stats['n_validation'],
        'validation_rmse': stats['validation_rmse'],
        'literature_rmse': stats['literature_rmse'],
        'improvement': stats['improvement'],
        'noise_ratio': stats['noise_ratio'],
    })
    
    print("\n" + "=" * 80)
    print("Prior Mean + Correction Validation Loop Complete!")
    print("=" * 80)
    print(f"\nMethod: Literature as prior + wet lab correction")
    print(f"Improvement over literature-only (CV RMSE): {stats['improvement']:.2f}")
    print(f"Mean systematic bias found: {stats['mean_residual']:+.2f}%")
    print(f"\nNext steps:")
    print(f"  1. Run optimization: python src/05_bo_optimization/bo_optimizer.py")
    print(f"  2. Test top candidates in wet lab")
    print(f"  3. Add results to: {validation_path}")
    print(f"  4. Run this script again for next iteration")
    
    print("\n" + "-" * 40)
    print("NOTE: This model uses a composite architecture.")
    print("For full functionality, use the composite_model.pkl file.")
    print("Standard gp_model.pkl contains the literature model only")
    print("(for backward compatibility with other modules).")
    print("-" * 40)


if __name__ == '__main__':
    main()
