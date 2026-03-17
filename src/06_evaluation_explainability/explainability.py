#!/usr/bin/env python3
"""
CryoMN Model Explainability Module

Generates comprehensive visualizations to explain the GP model predictions:
- Feature importance bar chart
- SHAP values analysis
- Partial dependence plots (PDPs)
- 2D contour plots (ingredient interactions)
- Acquisition function landscape
- GP uncertainty visualization

Author: CryoMN ML Project
Date: 2026-01-27
"""

import pandas as pd
import numpy as np
import os
import sys
import tempfile
import warnings
from typing import Tuple, Dict, List, Optional
from datetime import datetime

if 'MPLCONFIGDIR' not in os.environ:
    mpl_config_dir = os.path.join(tempfile.gettempdir(), 'cryomn-mpl')
    os.makedirs(mpl_config_dir, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = mpl_config_dir

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Optional imports with graceful fallback
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Note: seaborn not installed. Using matplotlib defaults.")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# Add shared helper modules to path for model resolution and observed-context loading
_script_dir = os.path.dirname(os.path.abspath(__file__))
_helper_dir = os.path.join(os.path.dirname(_script_dir), 'helper')
if _helper_dir not in sys.path:
    sys.path.insert(0, _helper_dir)
from active_model_resolver import ModelResolutionError, resolve_active_model  # noqa: E402
from observed_context import load_observed_context  # noqa: E402

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for all plots
if HAS_SEABORN:
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            pass
    sns.set_palette("husl")
else:
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


# =============================================================================
# CONFIGURATION
# =============================================================================

class ExplainabilityConfig:
    """Configuration for explainability visualizations."""
    # General settings
    figsize_small = (8, 6)
    figsize_medium = (10, 8)
    figsize_large = (14, 10)
    dpi = 150
    
    # PDP settings
    n_pdp_points = 50
    n_top_features_pdp = 8
    pdp_specific_features = None  # Use top features from importance ranking
    
    # Contour settings
    n_contour_points = 30
    n_top_pairs = 3
    interaction_specific_pairs = None  # Use top feature pairs from importance ranking
    
    # SHAP settings
    n_shap_samples = 100  # Background samples for SHAP
    
    # Acquisition settings
    acquisition_specific_pair = None  # Use top 2 features from importance ranking
    acquisition_mode = 'ucb'
    acquisition_kappa = 0.5
    acquisition_xi = 0.01
    
    # Color settings
    cmap_viability = 'RdYlGn'
    cmap_uncertainty = 'YlOrRd'
    cmap_acquisition = 'viridis'


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_model_and_data(project_root: str):
    """
    Load the trained GP model, scaler, and data.
    
    Uses model_metadata.json to decide whether the active model is composite
    (literature + wet lab correction) or standard GP (literature-only).
    
    Loads the active iteration's observed context, reconstructing it on demand
    if the canonical artifact is missing.
    
    Returns:
        Tuple of (model, scaler, feature_names, data_df, importance_df, is_composite, resolution)
    """
    resolution = resolve_active_model(project_root)
    gp = resolution.gp
    scaler = resolution.scaler
    metadata = resolution.metadata
    feature_names = metadata['feature_names']
    is_composite = resolution.is_composite

    df = load_observed_context(
        project_root=project_root,
        feature_names=feature_names,
        model_method=resolution.model_method,
        iteration=resolution.iteration,
        iteration_dir=resolution.iteration_dir,
        metadata=metadata,
    )
    
    model_dir = os.path.join(project_root, 'models')
    importance_path = os.path.join(model_dir, 'feature_importance.csv')
    if os.path.exists(importance_path):
        importance_df = pd.read_csv(importance_path)
    else:
        importance_df = pd.DataFrame({
            'feature': [name.replace('_M', '').replace('_pct', '') for name in feature_names],
            'importance': np.zeros(len(feature_names)),
        })
    
    return gp, scaler, feature_names, df, importance_df, is_composite, resolution


def clean_feature_name(name: str) -> str:
    """Clean feature name for display."""
    return name.replace('_M', '').replace('_pct', '').replace('_', ' ').title()


def get_unit(feature: str) -> str:
    """Get the appropriate unit for a feature."""
    if '_pct' in feature:
        return '%'
    elif '_M' in feature:
        return 'M'
    return ''


def resolve_feature_index(clean_name: str, feature_names: List[str]) -> int:
    """
    Resolve a cleaned feature name (from importance_df) back to its index
    in the full feature_names list.
    
    Handles both _M and _pct suffixes by trying all possibilities.
    
    Args:
        clean_name: Cleaned name like 'ectoin', 'fbs', 'peg_3350'
        feature_names: Full feature names like 'ectoin_M', 'fbs_pct'
    
    Returns:
        Index into feature_names, or -1 if not found
    """
    # Try exact match first
    if clean_name in feature_names:
        return feature_names.index(clean_name)
    # Try with _M suffix
    if clean_name + '_M' in feature_names:
        return feature_names.index(clean_name + '_M')
    # Try with _pct suffix
    if clean_name + '_pct' in feature_names:
        return feature_names.index(clean_name + '_pct')
    return -1


def resolve_feature_full_name(clean_name: str, feature_names: List[str]) -> str:
    """Resolve cleaned name to full feature name (with suffix)."""
    idx = resolve_feature_index(clean_name, feature_names)
    if idx >= 0:
        return feature_names[idx]
    return clean_name


def build_explainability_output_dir(base_output_dir: str,
                                    iteration_dir: Optional[str],
                                    iteration: Optional[int]) -> str:
    """Build an iteration-specific output directory for explainability artifacts."""
    if iteration_dir:
        suffix = iteration_dir
    elif iteration is not None:
        suffix = f'iteration_{iteration}'
    else:
        suffix = 'active_model'
    return os.path.join(base_output_dir, suffix)


def predict_model(model, scaler, X_raw: np.ndarray, is_composite: bool,
                  return_std: bool = False):
    """
    Centralized prediction helper that handles both model types.
    
    Args:
        model: GP model (GaussianProcessRegressor or CompositeGP)
        scaler: Feature scaler (unused if is_composite)
        X_raw: Unscaled feature matrix
        is_composite: If True, model handles scaling internally
        return_std: Whether to return uncertainty
    """
    if is_composite:
        return model.predict(X_raw, return_std=return_std)
    else:
        X_scaled = scaler.transform(X_raw)
        return model.predict(X_scaled, return_std=return_std)


# =============================================================================
# 1. FEATURE IMPORTANCE BAR CHART
# =============================================================================

def plot_feature_importance(importance_df: pd.DataFrame, output_dir: str,
                            config: ExplainabilityConfig = None):
    """
    Create a horizontal bar chart of feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        output_dir: Directory to save the plot
        config: Visualization configuration
    """
    config = config or ExplainabilityConfig()
    
    # Sort by importance
    df = importance_df.sort_values('importance', ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=config.figsize_medium)
    
    # Color gradient based on importance
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df)))
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(df)), df['importance'], color=colors)
    
    # Customize
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([clean_feature_name(f) for f in df['feature']])
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Feature Importance for Cell Viability Prediction', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df['importance'])):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    ax.set_xlim(0, df['importance'].max() * 1.15)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"  ✓ Feature importance chart saved: {output_path}")


# =============================================================================
# 2. SHAP VALUES ANALYSIS
# =============================================================================

def compute_shap_values(model, scaler, X: np.ndarray, feature_names: List[str],
                        is_composite: bool = False,
                        config: ExplainabilityConfig = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SHAP values using KernelExplainer (GP-compatible).
    
    Returns:
        Tuple of (shap_values, background_data)
    """
    config = config or ExplainabilityConfig()
    
    # Use a subset for background
    np.random.seed(42)
    n_samples = min(config.n_shap_samples, len(X))
    bg_idx = np.random.choice(len(X), n_samples, replace=False)
    X_background = X[bg_idx]
    
    # Define prediction function
    def predict_fn(X_raw):
        return predict_model(model, scaler, X_raw, is_composite, return_std=False)
    
    try:
        import shap
        
        # Create explainer
        explainer = shap.KernelExplainer(predict_fn, X_background)
        
        # Compute SHAP values for a subset of samples
        n_explain = min(100, len(X))
        explain_idx = np.random.choice(len(X), n_explain, replace=False)
        X_explain = X[explain_idx]
        
        shap_values = explainer.shap_values(X_explain, silent=True)
        
        return shap_values, X_explain
    
    except ImportError:
        print("  ⚠ SHAP library not installed. Skipping SHAP analysis.")
        return None, None


def plot_shap_summary(shap_values: np.ndarray, X_explain: np.ndarray,
                      feature_names: List[str], output_dir: str,
                      config: ExplainabilityConfig = None):
    """Create SHAP summary beeswarm plot."""
    config = config or ExplainabilityConfig()
    
    try:
        import shap
        
        # Clean feature names for display
        clean_names = [clean_feature_name(f) for f in feature_names]
        
        # Create summary plot
        fig, ax = plt.subplots(figsize=config.figsize_medium)
        shap.summary_plot(shap_values, X_explain, feature_names=clean_names,
                          show=False, plot_size=None)
        plt.title('SHAP Summary: Feature Impact on Viability', fontsize=14, fontweight='bold')
        
        output_path = os.path.join(output_dir, 'shap_summary.png')
        plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', transparent=True)
        plt.close()
        
        print(f"  ✓ SHAP summary plot saved: {output_path}")
        
        # Create SHAP importance bar plot
        fig, ax = plt.subplots(figsize=config.figsize_small)
        shap.summary_plot(shap_values, X_explain, feature_names=clean_names,
                          plot_type="bar", show=False, plot_size=None)
        plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
        
        output_path = os.path.join(output_dir, 'shap_importance.png')
        plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', transparent=True)
        plt.close()
        
        print(f"  ✓ SHAP importance plot saved: {output_path}")
        
    except Exception as e:
        print(f"  ⚠ Error creating SHAP plots: {e}")


# =============================================================================
# 3. PARTIAL DEPENDENCE PLOTS (PDPs)
# =============================================================================

def plot_partial_dependence(model, scaler,
                            X: np.ndarray, feature_names: List[str],
                            importance_df: pd.DataFrame, output_dir: str,
                            is_composite: bool = False,
                            config: ExplainabilityConfig = None):
    """
    Create partial dependence plots for top features.
    Shows how predicted viability changes with each ingredient concentration.
    """
    config = config or ExplainabilityConfig()
    
    # Get top features by importance or use specific features
    if hasattr(config, 'pdp_specific_features') and config.pdp_specific_features:
        top_features = config.pdp_specific_features
    else:
        top_features = importance_df.nlargest(config.n_top_features_pdp, 'importance')['feature'].tolist()
    
    # Get feature indices
    feature_indices = {name: i for i, name in enumerate(feature_names)}
    
    # Create figure with subplots
    n_cols = 2
    n_rows = (len(top_features) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        feat_idx = resolve_feature_index(feature, feature_names)
        full_name = resolve_feature_full_name(feature, feature_names)
        
        if feat_idx < 0:
            continue
        
        # Create feature range
        feat_min = X[:, feat_idx].min()
        feat_max = X[:, feat_idx].max()
        if feat_max - feat_min < 1e-6:
            feat_max = feat_min + 1
        
        feat_values = np.linspace(feat_min, feat_max, config.n_pdp_points)
        
        # Compute predictions for each feature value (averaging over other features)
        pdp_means = []
        pdp_stds = []
        
        X_mean = X.mean(axis=0)
        
        for val in feat_values:
            X_temp = X_mean.copy()
            X_temp[feat_idx] = val
            mean, std = predict_model(model, scaler, X_temp.reshape(1, -1),
                                      is_composite, return_std=True)
            pdp_means.append(mean[0])
            pdp_stds.append(std[0])
        
        pdp_means = np.array(pdp_means)
        pdp_stds = np.array(pdp_stds)
        
        # Plot
        ax.plot(feat_values, pdp_means, 'b-', linewidth=2, label='Mean Prediction')
        ax.fill_between(feat_values, 
                        pdp_means - 1.96 * pdp_stds,
                        pdp_means + 1.96 * pdp_stds,
                        alpha=0.3, color='blue', label='95% CI')
        
        ax.set_xlabel(f'{clean_feature_name(full_name)} ({get_unit(full_name)})', fontsize=10)
        ax.set_ylabel('Predicted Viability (%)', fontsize=10)
        ax.set_title(f'PDP: {clean_feature_name(full_name)}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(top_features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Partial Dependence Plots: Effect of Individual Ingredients', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'partial_dependence_plots.png')
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"  ✓ Partial dependence plots saved: {output_path}")


# =============================================================================
# 4. 2D CONTOUR PLOTS (INGREDIENT INTERACTIONS)
# =============================================================================

def plot_interaction_contours(model, scaler,
                              X: np.ndarray, feature_names: List[str],
                              importance_df: pd.DataFrame, output_dir: str,
                              is_composite: bool = False,
                              config: ExplainabilityConfig = None):
    """
    Create 2D contour plots showing interactions between top ingredient pairs.
    """
    config = config or ExplainabilityConfig()
    
    # Get top features
    top_features = importance_df.nlargest(4, 'importance')['feature'].tolist()
    
    # Resolve full feature names
    resolved = [(f, resolve_feature_index(f, feature_names), resolve_feature_full_name(f, feature_names))
                for f in top_features]
    resolved = [(f, idx, full) for f, idx, full in resolved if idx >= 0]
    
    # Generate pairs or use specific pairs
    if hasattr(config, 'interaction_specific_pairs') and config.interaction_specific_pairs:
        pairs = config.interaction_specific_pairs
    else:
        # Generate pairs from resolved features
        pairs = []
        for i in range(len(resolved)):
            for j in range(i + 1, len(resolved)):
                pairs.append((resolved[i], resolved[j]))
        
        pairs = pairs[:config.n_top_pairs]
    
    # Create figure
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5))
    if len(pairs) == 1:
        axes = [axes]
    
    X_mean = X.mean(axis=0)
    
    for p_idx, ((feat1, idx1, full1), (feat2, idx2, full2)) in enumerate(pairs):
        ax = axes[p_idx]
        

        # Create grid
        x1_range = np.linspace(X[:, idx1].min(), X[:, idx1].max(), config.n_contour_points)
        x2_range = np.linspace(X[:, idx2].min(), X[:, idx2].max(), config.n_contour_points)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        
        # Compute predictions over grid
        Z = np.zeros_like(X1)
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                X_temp = X_mean.copy()
                X_temp[idx1] = X1[i, j]
                X_temp[idx2] = X2[i, j]
                Z[i, j] = predict_model(model, scaler, X_temp.reshape(1, -1),
                                        is_composite)[0]
        
        # Plot contour
        contour = ax.contourf(X1, X2, Z, levels=20, cmap=config.cmap_viability)
        plt.colorbar(contour, ax=ax, label='Predicted Viability (%)')
        
        # Add contour lines
        ax.contour(X1, X2, Z, levels=10, colors='white', alpha=0.3, linewidths=0.5)
        
        ax.set_xlabel(f'{clean_feature_name(full1)} ({get_unit(full1)})', fontsize=10)
        ax.set_ylabel(f'{clean_feature_name(full2)} ({get_unit(full2)})', fontsize=10)
        ax.set_title(f'{clean_feature_name(full1)} × {clean_feature_name(full2)}', 
                     fontsize=11, fontweight='bold')
    
    plt.suptitle('Ingredient Interaction Effects on Cell Viability', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'interaction_contours.png')
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"  ✓ Interaction contour plots saved: {output_path}")


# =============================================================================
# 5. ACQUISITION FUNCTION LANDSCAPE
# =============================================================================

def expected_improvement(mean: np.ndarray, std: np.ndarray, 
                         y_best: float, xi: float = 0.01) -> np.ndarray:
    """Calculate Expected Improvement."""
    with np.errstate(divide='warn'):
        z = (mean - y_best - xi) / std
        ei = (mean - y_best - xi) * norm.cdf(z) + std * norm.pdf(z)
        ei[std < 1e-9] = 0.0
    return ei


def upper_confidence_bound(mean: np.ndarray, std: np.ndarray, kappa: float = 0.5) -> np.ndarray:
    """Calculate Upper Confidence Bound."""
    return mean + kappa * std


def plot_acquisition_landscape(model, scaler,
                               X: np.ndarray, y: np.ndarray, feature_names: List[str],
                               importance_df: pd.DataFrame, output_dir: str,
                               is_composite: bool = False,
                               config: ExplainabilityConfig = None):
    """
    Visualize the configured acquisition function landscape.
    """
    config = config or ExplainabilityConfig()
    
    # Get features for 2D visualization
    if hasattr(config, 'acquisition_specific_pair') and config.acquisition_specific_pair:
        top_features = list(config.acquisition_specific_pair)
    else:
        # Get top 2 features
        top_features = importance_df.nlargest(2, 'importance')['feature'].tolist()
    
    feat1, feat2 = top_features[0], top_features[1]
    idx1 = resolve_feature_index(feat1, feature_names)
    idx2 = resolve_feature_index(feat2, feature_names)
    full1 = resolve_feature_full_name(feat1, feature_names)
    full2 = resolve_feature_full_name(feat2, feature_names)
    
    X_mean = X.mean(axis=0)
    
    # Compute y_best from model predictions (important for composite model)
    y_pred = predict_model(model, scaler, X, is_composite)
    y_best = np.max(y_pred)
    
    # Create grid (use data-driven ranges)
    n_points = config.n_contour_points
    x1_range = np.linspace(X[:, idx1].min(), X[:, idx1].max(), n_points)
    x2_range = np.linspace(X[:, idx2].min(), X[:, idx2].max(), n_points)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # Compute predictions and acquisition score.
    Z_mean = np.zeros_like(X1)
    Z_std = np.zeros_like(X1)
    Z_acq = np.zeros_like(X1)

    acquisition_mode = config.acquisition_mode.lower()
    if acquisition_mode == 'ei':
        acquisition_label = 'Expected Improvement'
        acquisition_title = 'Acquisition Function (EI)'
    else:
        acquisition_label = 'Upper Confidence Bound'
        acquisition_title = 'Acquisition Function (UCB)'
    
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            X_temp = X_mean.copy()
            X_temp[idx1] = X1[i, j]
            X_temp[idx2] = X2[i, j]
            mean, std = predict_model(model, scaler, X_temp.reshape(1, -1),
                                      is_composite, return_std=True)
            Z_mean[i, j] = mean[0]
            Z_std[i, j] = std[0]
            if acquisition_mode == 'ei':
                Z_acq[i, j] = expected_improvement(
                    mean, std, y_best, xi=config.acquisition_xi
                )[0]
            else:
                Z_acq[i, j] = upper_confidence_bound(
                    mean, std, kappa=config.acquisition_kappa
                )[0]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: GP Mean
    contour1 = axes[0].contourf(X1, X2, Z_mean, levels=20, cmap=config.cmap_viability)
    plt.colorbar(contour1, ax=axes[0], label='Predicted Viability (%)')
    axes[0].set_xlabel(f'{clean_feature_name(full1)} ({get_unit(full1)})')
    axes[0].set_ylabel(f'{clean_feature_name(full2)} ({get_unit(full2)})')
    axes[0].set_title('GP Mean Prediction', fontweight='bold')
    
    # Plot 2: GP Uncertainty
    contour2 = axes[1].contourf(X1, X2, Z_std, levels=20, cmap=config.cmap_uncertainty)
    plt.colorbar(contour2, ax=axes[1], label='Uncertainty (std)')
    axes[1].set_xlabel(f'{clean_feature_name(full1)} ({get_unit(full1)})')
    axes[1].set_ylabel(f'{clean_feature_name(full2)} ({get_unit(full2)})')
    axes[1].set_title('GP Uncertainty', fontweight='bold')
    
    # Plot 3: acquisition score
    contour3 = axes[2].contourf(X1, X2, Z_acq, levels=20, cmap=config.cmap_acquisition)
    plt.colorbar(contour3, ax=axes[2], label=acquisition_label)
    axes[2].set_xlabel(f'{clean_feature_name(full1)} ({get_unit(full1)})')
    axes[2].set_ylabel(f'{clean_feature_name(full2)} ({get_unit(full2)})')
    axes[2].set_title(acquisition_title, fontweight='bold')
    
    # Mark best observed point
    best_idx = np.argmax(y)
    for ax in axes:
        ax.scatter(X[best_idx, idx1], X[best_idx, idx2], 
                   c='red', s=100, marker='*', edgecolors='white',
                   linewidths=1.5, zorder=5, label='Best Observed')
        ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Acquisition Function Landscape: Exploration vs Exploitation', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'acquisition_landscape.png')
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"  ✓ Acquisition landscape saved: {output_path}")


# =============================================================================
# 6. GP UNCERTAINTY VISUALIZATION
# =============================================================================

def plot_uncertainty_analysis(model, scaler,
                              X: np.ndarray, y: np.ndarray, feature_names: List[str],
                              output_dir: str, is_composite: bool = False,
                              config: ExplainabilityConfig = None):
    """
    Visualize GP uncertainty across the observed data.
    """
    config = config or ExplainabilityConfig()
    
    # Get predictions with uncertainty for all data points
    y_pred, y_std = predict_model(model, scaler, X, is_composite, return_std=True)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=config.figsize_large)
    
    # Plot 1: Predicted vs Actual with error bars
    ax1 = axes[0, 0]
    scatter = ax1.scatter(y, y_pred, c=y_std, cmap='YlOrRd', 
                          s=50, alpha=0.7, edgecolors='white', linewidths=0.5)
    plt.colorbar(scatter, ax=ax1, label='Uncertainty (std)')
    
    # Add diagonal line
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Viability (%)', fontsize=10)
    ax1.set_ylabel('Predicted Viability (%)', fontsize=10)
    ax1.set_title('Predicted vs Actual (colored by uncertainty)', fontweight='bold')
    ax1.legend()
    
    # Plot 2: Uncertainty distribution
    ax2 = axes[0, 1]
    ax2.hist(y_std, bins=30, color='coral', edgecolor='white', alpha=0.8)
    ax2.axvline(y_std.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {y_std.mean():.2f}')
    ax2.set_xlabel('Prediction Uncertainty (std)', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Distribution of Model Uncertainty', fontweight='bold')
    ax2.legend()
    
    # Plot 3: Residuals vs Uncertainty
    ax3 = axes[1, 0]
    residuals = y - y_pred
    ax3.scatter(y_std, np.abs(residuals), c=y, cmap='viridis', 
                s=50, alpha=0.7, edgecolors='white', linewidths=0.5)
    ax3.set_xlabel('Prediction Uncertainty (std)', fontsize=10)
    ax3.set_ylabel('Absolute Error (%)', fontsize=10)
    ax3.set_title('Error vs Uncertainty (calibration check)', fontweight='bold')
    
    # Add trend line
    z = np.polyfit(y_std, np.abs(residuals), 1)
    p = np.poly1d(z)
    x_line = np.linspace(y_std.min(), y_std.max(), 100)
    ax3.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=2, label='Trend')
    ax3.legend()
    
    # Plot 4: Uncertainty by viability range
    ax4 = axes[1, 1]
    viability_bins = [0, 30, 50, 70, 90, 100]
    bin_labels = ['0-30%', '30-50%', '50-70%', '70-90%', '90-100%']
    bin_uncertainties = []
    
    for i in range(len(viability_bins) - 1):
        mask = (y >= viability_bins[i]) & (y < viability_bins[i+1])
        if mask.sum() > 0:
            bin_uncertainties.append(y_std[mask].mean())
        else:
            bin_uncertainties.append(0)
    
    bars = ax4.bar(bin_labels, bin_uncertainties, color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(bin_labels))))
    ax4.set_xlabel('Viability Range', fontsize=10)
    ax4.set_ylabel('Mean Uncertainty (std)', fontsize=10)
    ax4.set_title('Model Uncertainty by Viability Range', fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, bin_uncertainties):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f'{val:.1f}', ha='center', fontsize=9)
    
    model_label = 'Composite' if is_composite else 'GP'
    plt.suptitle(f'{model_label} Model Uncertainty Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'uncertainty_analysis.png')
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"  ✓ Uncertainty analysis saved: {output_path}")


# =============================================================================
# 7. FEATURE IMPORTANCE (PERMUTATION-BASED)
# =============================================================================

def compute_feature_importance(model, scaler, feature_names: List[str],
                               X: np.ndarray, y: np.ndarray,
                               is_composite: bool = False,
                               weights: np.ndarray = None) -> pd.DataFrame:
    """
    Compute feature importance using permutation importance.
    
    Uses weighted R² when weights are provided, so wet lab data points
    count proportionally more in the importance calculation.
    
    Args:
        model: Trained model (GaussianProcessRegressor or CompositeGP)
        scaler: Feature scaler (unused if is_composite)
        feature_names: List of feature names
        X: Unscaled feature matrix
        y: Target values
        is_composite: Whether model is a CompositeGP
        weights: Per-sample weights (e.g. 1.0 for literature, 50.0 for wet lab)
    
    Returns:
        DataFrame with feature importance scores
    """
    if weights is None:
        weights = np.ones(len(y))
    
    def weighted_r2(y_true, y_pred, w):
        """Compute weighted R² score."""
        ss_res = np.sum(w * (y_true - y_pred) ** 2)
        ss_tot = np.sum(w * (y_true - np.average(y_true, weights=w)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    # Baseline predictions
    y_pred_baseline = predict_model(model, scaler, X, is_composite)
    baseline_score = weighted_r2(y, y_pred_baseline, weights)
    
    importance_scores = []
    
    for i, name in enumerate(feature_names):
        # Permute feature (on unscaled data)
        X_permuted = X.copy()
        np.random.seed(42)
        X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
        
        # Predictions with permuted feature
        y_pred_perm = predict_model(model, scaler, X_permuted, is_composite)
        permuted_score = weighted_r2(y, y_pred_perm, weights)
        
        importance = baseline_score - permuted_score
        
        importance_scores.append({
            'feature': name.replace('_M', '').replace('_pct', ''),
            'importance': importance,
        })
    
    importance_df = pd.DataFrame(importance_scores)
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for explainability visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    base_output_dir = os.path.join(project_root, 'results', 'explainability')
    os.makedirs(base_output_dir, exist_ok=True)
    
    print("=" * 80)
    print("CryoMN Model Explainability Analysis")
    print("=" * 80)
    
    # Load model and data
    print("\n📊 Loading model and data...")
    try:
        gp, scaler, feature_names, df, importance_df, is_composite, resolution = load_model_and_data(project_root)
    except ModelResolutionError as exc:
        print(f"ERROR: {exc}")
        return
    
    X = df[feature_names].values
    y = df['viability_percent'].values
    weights = df['context_weight'].values if 'context_weight' in df.columns else np.ones(len(y))
    
    print(f"  Model loaded with {len(feature_names)} features")
    print(f"  Data loaded with {len(df)} formulations")
    if resolution.iteration_dir:
        print(f"  Resolved active iteration: {resolution.iteration_dir}")
    elif resolution.iteration is not None:
        print(f"  Resolved active iteration: iteration_{resolution.iteration}")
    if is_composite and 'source' in df.columns:
        n_lit = (df['source'] == 'literature').sum()
        n_wet = (df['source'] == 'wetlab').sum()
        wet_weight = df.loc[df['source'] == 'wetlab', 'context_weight'].iloc[0] if n_wet > 0 else 'N/A'
        print(f"  Sources: {n_lit} literature + {n_wet} wet lab (weight={wet_weight})")
    
    output_dir = build_explainability_output_dir(
        base_output_dir,
        resolution.iteration_dir,
        resolution.iteration,
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n  Saving explainability outputs to: {output_dir}")
    
    config = ExplainabilityConfig()
    
    # Recompute feature importance for the active model
    print("\n0️⃣  Computing Feature Importance (permutation-based)")
    importance_df = compute_feature_importance(
        gp, scaler, feature_names, X, y, is_composite, weights
    )
    # Save to the output directory
    importance_csv_path = os.path.join(output_dir, 'feature_importance.csv')
    importance_df.to_csv(importance_csv_path, index=False)
    print(f"  ✓ Feature importance saved: {importance_csv_path}")
    
    # Generate all visualizations
    print("\n📈 Generating visualizations...\n")
    
    # 1. Feature Importance
    print("1️⃣  Feature Importance Bar Chart")
    plot_feature_importance(importance_df, output_dir, config)
    
    # 2. SHAP Analysis
    print("\n2️⃣  SHAP Values Analysis")
    shap_values, X_explain = compute_shap_values(gp, scaler, X, feature_names, is_composite, config)
    if shap_values is not None:
        plot_shap_summary(shap_values, X_explain, feature_names, output_dir, config)
    
    # 3. Partial Dependence Plots
    print("\n3️⃣  Partial Dependence Plots")
    plot_partial_dependence(gp, scaler, X, feature_names, importance_df, output_dir, is_composite, config)
    
    # 4. Interaction Contours
    print("\n4️⃣  2D Interaction Contour Plots")
    plot_interaction_contours(gp, scaler, X, feature_names, importance_df, output_dir, is_composite, config)
    
    # 5. Acquisition Landscape
    print("\n5️⃣  Acquisition Function Landscape")
    plot_acquisition_landscape(gp, scaler, X, y, feature_names, importance_df, output_dir, is_composite, config)
    
    # 6. Uncertainty Analysis
    print("\n6️⃣  GP Uncertainty Visualization")
    plot_uncertainty_analysis(gp, scaler, X, y, feature_names, output_dir, is_composite, config)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ Explainability Analysis Complete!")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.png') or f.endswith('.csv'):
            print(f"  • {f}")


if __name__ == '__main__':
    main()
