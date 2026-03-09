#!/usr/bin/env python3
"""
CryoMN Gaussian Process Model Training

Trains a Gaussian Process regression model to predict cell viability
from cryoprotective formulation ingredients.

Author: CryoMN ML Project
Date: 2026-01-24
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from typing import Tuple, Dict, List, Optional
from datetime import datetime

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'kernel_type': 'matern',  # 'matern' or 'rbf'
    'matern_nu': 2.5,  # Smoothness parameter (0.5, 1.5, 2.5)
    'n_restarts': 10,  # Number of optimizer restarts
    'cv_folds': 5,  # Cross-validation folds
    'random_state': 42,
    'alpha': 1e-6,  # Noise level (regularization)
}


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_parsed_data(data_path: str) -> pd.DataFrame:
    """
    Load parsed formulation data.
    
    Args:
        data_path: Path to parsed_formulations.csv
        
    Returns:
        DataFrame with parsed data
    """
    df = pd.read_csv(data_path)
    
    # Filter out invalid viability values (>100% are relative, not absolute)
    original_count = len(df)
    df = df[df['viability_percent'] <= 100].copy()
    filtered_count = len(df)
    
    if original_count != filtered_count:
        print(f"Filtered {original_count - filtered_count} rows with viability > 100%")
    
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare feature matrix and target vector.
    
    Args:
        df: Parsed DataFrame
        
    Returns:
        Tuple of (X features, y target, feature_names)
    """
    # Get ingredient columns (both molar _M and percentage _pct)
    ingredient_cols = [c for c in df.columns if c.endswith('_M') or c.endswith('_pct')]
    
    # Filter to only include ingredients with at least 3 non-zero values
    active_ingredients = []
    for col in ingredient_cols:
        if (df[col] > 0).sum() >= 3:
            active_ingredients.append(col)
    
    print(f"Using {len(active_ingredients)} active ingredients as features")
    
    # Extract features and target
    X = df[active_ingredients].values
    y = df['viability_percent'].values
    
    return X, y, active_ingredients


def scale_features(X_train: np.ndarray, X_test: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale features using StandardScaler.
    
    Args:
        X_train: Training features
        X_test: Optional test features
        
    Returns:
        Tuple of (scaled X_train, scaled X_test or None, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


# =============================================================================
# MODEL TRAINING
# =============================================================================

def create_kernel(config: Dict) -> object:
    """
    Create GP kernel based on configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        sklearn kernel object
    """
    if config['kernel_type'] == 'matern':
        base_kernel = Matern(nu=config['matern_nu'], length_scale_bounds=(1e-5, 1e5))
    else:
        base_kernel = RBF(length_scale_bounds=(1e-5, 1e5))
    
    # Add constant kernel for amplitude and white kernel for noise
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * base_kernel + WhiteKernel(noise_level=1.0)
    
    return kernel


def train_gp_model(X: np.ndarray, y: np.ndarray, config: Dict = None) -> Tuple[GaussianProcessRegressor, StandardScaler]:
    """
    Train Gaussian Process regression model.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        config: Optional configuration dictionary
        
    Returns:
        Tuple of (trained GP model, feature scaler)
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    # Scale features
    X_scaled, _, scaler = scale_features(X)
    
    # Create kernel
    kernel = create_kernel(config)
    
    # Create and train GP
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=config['n_restarts'],
        random_state=config['random_state'],
        alpha=config['alpha'],
        normalize_y=True,  # Normalize target for better numerical stability
    )
    
    print("Training Gaussian Process model...")
    gp.fit(X_scaled, y)
    
    print(f"Optimized kernel: {gp.kernel_}")
    print(f"Log-marginal-likelihood: {gp.log_marginal_likelihood_value_:.3f}")
    
    return gp, scaler


def cross_validate_model(X: np.ndarray, y: np.ndarray, config: Dict = None) -> Dict:
    """
    Perform cross-validation on GP model.
    
    Args:
        X: Feature matrix
        y: Target vector
        config: Optional configuration dictionary
        
    Returns:
        Dictionary with CV results
    """
    if config is None:
        config = DEFAULT_CONFIG

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gp', GaussianProcessRegressor(
            kernel=create_kernel(config),
            n_restarts_optimizer=config['n_restarts'],
            random_state=config['random_state'],
            alpha=config['alpha'],
            normalize_y=True,
        )),
    ])
    
    # Cross-validation
    kfold = KFold(n_splits=config['cv_folds'], shuffle=True, random_state=config['random_state'])
    
    cv_scores = cross_val_score(
        pipeline, X, y, cv=kfold, scoring='neg_mean_squared_error'
    )
    cv_rmse = np.sqrt(-cv_scores)
    
    # Also compute R² scores
    r2_scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='r2')
    
    results = {
        'cv_rmse_mean': cv_rmse.mean(),
        'cv_rmse_std': cv_rmse.std(),
        'cv_r2_mean': r2_scores.mean(),
        'cv_r2_std': r2_scores.std(),
        'n_folds': config['cv_folds'],
    }
    
    return results


def evaluate_model(gp: GaussianProcessRegressor, scaler: StandardScaler, 
                   X: np.ndarray, y: np.ndarray) -> Dict:
    """
    Evaluate trained model on data.
    
    Args:
        gp: Trained GP model
        scaler: Feature scaler
        X: Feature matrix
        y: True target values
        
    Returns:
        Dictionary with evaluation metrics
    """
    X_scaled = scaler.transform(X)
    y_pred, y_std = gp.predict(X_scaled, return_std=True)
    
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    results = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mean_uncertainty': y_std.mean(),
    }
    
    return results


# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================

def analyze_feature_importance(gp: GaussianProcessRegressor, scaler: StandardScaler,
                               feature_names: List[str], X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """
    Analyze feature importance using permutation importance.
    
    Args:
        gp: Trained GP model
        scaler: Feature scaler
        feature_names: List of feature names
        X: Feature matrix
        y: Target values
        
    Returns:
        DataFrame with feature importance scores
    """
    X_scaled = scaler.transform(X)
    baseline_score = gp.score(X_scaled, y)
    
    importance_scores = []
    
    for i, name in enumerate(feature_names):
        # Permute feature
        X_permuted = X_scaled.copy()
        np.random.seed(42)
        X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
        
        # Calculate score drop
        permuted_score = gp.score(X_permuted, y)
        importance = baseline_score - permuted_score
        
        importance_scores.append({
            'feature': name.replace('_M', '').replace('_pct', ''),
            'importance': importance,
        })
    
    importance_df = pd.DataFrame(importance_scores)
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_model(gp: GaussianProcessRegressor, scaler: StandardScaler, 
               feature_names: List[str], output_dir: str, 
               metrics: Dict = None, config: Dict = None):
    """
    Save trained model and associated files.
    
    Args:
        gp: Trained GP model
        scaler: Feature scaler
        feature_names: List of feature names
        output_dir: Output directory
        metrics: Optional evaluation metrics
        config: Optional configuration
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'gp_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(gp, f)
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metadata
    metadata = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'trained_at': datetime.now().isoformat(),
        'config': config or DEFAULT_CONFIG,
        'metrics': metrics,
        'is_composite_model': False,
    }
    
    metadata_path = os.path.join(output_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {output_dir}")


def load_model(model_dir: str) -> Tuple[GaussianProcessRegressor, StandardScaler, Dict]:
    """
    Load trained model and associated files.
    
    Args:
        model_dir: Directory containing saved model
        
    Returns:
        Tuple of (GP model, scaler, metadata)
    """
    # Load model
    model_path = os.path.join(model_dir, 'gp_model.pkl')
    with open(model_path, 'rb') as f:
        gp = pickle.load(f)
    
    # Load scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load metadata
    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return gp, scaler, metadata


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for model training."""
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    data_path = os.path.join(project_root, 'data', 'processed', 'parsed_formulations.csv')
    model_dir = os.path.join(project_root, 'models')
    
    print("=" * 80)
    print("CryoMN Gaussian Process Model Training")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    df = load_parsed_data(data_path)
    print(f"Loaded {len(df)} formulations")
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target range: {y.min():.1f}% - {y.max():.1f}%")
    
    # Cross-validation
    print("\n" + "-" * 40)
    print("Cross-Validation")
    print("-" * 40)
    cv_results = cross_validate_model(X, y)
    print(f"CV RMSE: {cv_results['cv_rmse_mean']:.2f} ± {cv_results['cv_rmse_std']:.2f}")
    print(f"CV R²:   {cv_results['cv_r2_mean']:.3f} ± {cv_results['cv_r2_std']:.3f}")
    
    # Train final model on all data
    print("\n" + "-" * 40)
    print("Training Final Model")
    print("-" * 40)
    gp, scaler = train_gp_model(X, y)
    
    # Evaluate on training data (for reference)
    train_metrics = evaluate_model(gp, scaler, X, y)
    print(f"\nTraining Metrics:")
    print(f"  RMSE: {train_metrics['rmse']:.2f}")
    print(f"  MAE:  {train_metrics['mae']:.2f}")
    print(f"  R²:   {train_metrics['r2']:.3f}")
    print(f"  Mean Uncertainty: {train_metrics['mean_uncertainty']:.2f}")
    
    # Feature importance
    print("\n" + "-" * 40)
    print("Feature Importance (Top 10)")
    print("-" * 40)
    importance_df = analyze_feature_importance(gp, scaler, feature_names, X, y)
    print(importance_df.head(10).to_string(index=False))
    
    # Save model
    print("\n" + "-" * 40)
    print("Saving Model")
    print("-" * 40)
    
    all_metrics = {
        'cv': cv_results,
        'train': train_metrics,
    }
    save_model(gp, scaler, feature_names, model_dir, metrics=all_metrics)
    
    # Save feature importance
    importance_path = os.path.join(model_dir, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"Feature importance saved to: {importance_path}")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
