#!/usr/bin/env python3
"""
CryoMN Model Explainability Module

Generates support-aware, publication-ready visualizations that explain the
active model while staying faithful to the observed formulation manifold.

Artifacts:
- Feature importance bar chart
- SHAP summary and SHAP importance
- Support-aware empirical marginal plots
- Support-aware interaction contours
- Acquisition / BO-score landscape
- Uncertainty calibration dashboard
- Support diagnostics

Author: CryoMN ML Project
Date: 2026-01-27
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if 'MPLCONFIGDIR' not in os.environ:
    mpl_config_dir = os.path.join(tempfile.gettempdir(), 'cryomn-mpl')
    os.makedirs(mpl_config_dir, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = mpl_config_dir

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Note: seaborn not installed. Using matplotlib defaults.")

from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_script_dir)
_helper_dir = os.path.join(_src_dir, 'helper')
_bo_dir = os.path.join(_src_dir, '05_bo_optimization')
for path in (_helper_dir, _bo_dir):
    if path not in sys.path:
        sys.path.insert(0, path)

from active_model_resolver import ModelResolutionError, resolve_active_model  # noqa: E402
from formulation_formatting import (  # noqa: E402
    normalize_formulation_matrix,
    normalize_formulation_vector,
)
from observed_context import (  # noqa: E402
    collapse_observed_context_for_bo,
    load_observed_context,
    weighted_quantile,
)
from bo_optimizer import BOConfig  # noqa: E402

warnings.filterwarnings('ignore')

FONT_BUMP = 2

if HAS_SEABORN:
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            pass
else:
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.18


@dataclass
class ExplainabilityConfig:
    """Configuration for explainability visualizations."""

    figsize_small: Tuple[int, int] = (9, 6)
    figsize_medium: Tuple[int, int] = (11, 8)
    figsize_large: Tuple[int, int] = (16, 10)
    figsize_wide: Tuple[int, int] = (18, 6)
    figsize_dashboard: Tuple[int, int] = (16, 11)
    dpi: int = 170

    n_top_features_overview: int = 12
    n_top_features_pdp: int = 6
    n_top_features_shap: int = 12
    n_contour_points: int = 60
    n_pdp_points: int = 70
    n_top_pairs: int = 3
    n_shap_samples: int = 100

    support_quantile_low: float = 0.01
    support_quantile_high: float = 0.95
    support_radius_scale_1d: float = 1.35
    support_radius_scale_2d: float = 1.4
    low_support_alpha: float = 0.08
    low_support_facecolor: str = '#dbe2e8'

    acquisition_mode: str = 'ucb'
    acquisition_kappa: float = BOConfig.kappa
    acquisition_xi: float = BOConfig.xi
    bo_support_penalty: float = BOConfig.support_penalty
    bo_support_radius_scale: float = BOConfig.support_radius_scale
    bo_sparsity_penalty: float = BOConfig.sparsity_penalty

    cmap_viability: str = 'RdYlGn'
    cmap_uncertainty: str = 'YlOrRd'
    cmap_acquisition: str = 'viridis'
    line_primary: str = '#0b5d7a'
    line_secondary: str = '#2e8b57'
    color_literature: str = '#6a7f8f'
    color_wetlab: str = '#d55d3e'
    support_fill: str = '#cfd8df'
    contour_line_dark: str = '#22303c'
    contour_line_light: str = '#ffffff'
    support_diagnostic_legend_scale: float = 1.5
    interaction_min_axis_balance: float = 0.10
    interaction_min_occupied_bins: int = 4


def load_model_and_data(project_root: str):
    """Load the active model, scaler, feature names, and observed context."""
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
    """Get the display unit for a feature."""
    if feature.endswith('_pct'):
        return '%'
    if feature.endswith('_M'):
        return 'M'
    return ''


def resolve_feature_index(clean_name: str, feature_names: Sequence[str]) -> int:
    """Resolve a cleaned feature name back to its full feature index."""
    if clean_name in feature_names:
        return feature_names.index(clean_name)
    if clean_name + '_M' in feature_names:
        return feature_names.index(clean_name + '_M')
    if clean_name + '_pct' in feature_names:
        return feature_names.index(clean_name + '_pct')
    return -1


def resolve_feature_full_name(clean_name: str, feature_names: Sequence[str]) -> str:
    """Resolve a cleaned feature name to the full stored feature name."""
    idx = resolve_feature_index(clean_name, feature_names)
    return feature_names[idx] if idx >= 0 else clean_name


def build_explainability_output_dir(base_output_dir: str,
                                    iteration_dir: Optional[str],
                                    iteration: Optional[int]) -> str:
    """Build the iteration-specific output directory for explainability artifacts."""
    if iteration_dir:
        suffix = iteration_dir
    elif iteration is not None:
        suffix = f'iteration_{iteration}'
    else:
        suffix = 'active_model'
    return os.path.join(base_output_dir, suffix)


def predict_model(model, scaler, X_raw: np.ndarray, is_composite: bool,
                  return_std: bool = False):
    """Centralized prediction helper that handles both model types."""
    X_raw = np.asarray(X_raw, dtype=float)
    if is_composite:
        return model.predict(X_raw, return_std=return_std)
    X_scaled = scaler.transform(X_raw)
    return model.predict(X_scaled, return_std=return_std)


def apply_publication_style(config: ExplainabilityConfig):
    """Apply a consistent publication-style theme across the whole suite."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': '#fbfcfd',
        'savefig.facecolor': 'white',
        'axes.edgecolor': '#c4ccd4',
        'axes.linewidth': 1.0,
        'grid.color': '#d8dde3',
        'grid.linewidth': 0.7,
        'grid.alpha': 0.35,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11 + FONT_BUMP,
        'axes.titlesize': 13 + FONT_BUMP,
        'font.size': 11 + FONT_BUMP,
        'legend.frameon': False,
        'legend.fontsize': 9 + FONT_BUMP,
        'xtick.color': '#33414f',
        'ytick.color': '#33414f',
    })
    if HAS_SEABORN:
        sns.set_theme(style='whitegrid', context='talk', font_scale=0.90)


def source_masks(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Return boolean masks for literature and wet-lab rows."""
    source = df.get('source', pd.Series(['unknown'] * len(df))).astype(str).to_numpy()
    return {
        'literature': source == 'literature',
        'wetlab': source == 'wetlab',
        'other': ~np.isin(source, ['literature', 'wetlab']),
    }


def source_legend_handles(config: ExplainabilityConfig, edge_color: str = 'white') -> List[Line2D]:
    """Legend handles for literature / wet-lab point overlays."""
    return [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=config.color_literature,
               markeredgecolor=edge_color, markeredgewidth=0.5, label='Literature', markersize=7, alpha=0.55),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=config.color_wetlab,
               markeredgecolor=edge_color, markeredgewidth=0.6, label='Wet Lab', markersize=8, alpha=0.95),
    ]


def alpha_legend_handles(config: ExplainabilityConfig, base_color: Optional[str] = None,
                         marker_scale: float = 1.0) -> List[Line2D]:
    """Legend handles for plots that distinguish sources by alpha rather than color."""
    color = base_color or config.line_primary
    return [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
               markeredgecolor='white', markeredgewidth=0.5, label='Literature',
               markersize=7 * marker_scale, alpha=0.20),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
               markeredgecolor='white', markeredgewidth=0.6, label='Wet Lab',
               markersize=8 * marker_scale, alpha=0.95),
    ]


def support_diagnostic_density_legend_handles(config: ExplainabilityConfig,
                                              base_color: Optional[str] = None) -> List[Patch]:
    """Legend handles that match the filled support-density bands."""
    color = base_color or config.line_primary
    return [
        Patch(facecolor=color, edgecolor='none', alpha=0.30, label='Literature'),
        Patch(facecolor=color, edgecolor='none', alpha=0.80, label='Wet Lab'),
    ]


def support_diagnostic_legend_kwargs(config: ExplainabilityConfig) -> Dict[str, float]:
    """Shared, enlarged legend styling for support diagnostics."""
    scale = max(config.support_diagnostic_legend_scale, 1.0)
    return {
        'fontsize': (9 + FONT_BUMP) * scale,
        'borderpad': 0.35 * scale,
        'handlelength': 1.8 * scale,
        'handleheight': 0.8 * scale,
        'handletextpad': 0.5 * scale,
        'labelspacing': 0.3 * scale,
    }


def relative_luminance(color: Tuple[float, float, float, float] | Tuple[float, float, float] | str) -> float:
    """Return the WCAG-style relative luminance for a color."""
    rgb = np.array(mcolors.to_rgb(color), dtype=float)
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    return float(np.dot(linear, np.array([0.2126, 0.7152, 0.0722])))


def choose_contour_line_color(surface: np.ndarray, cmap_name: str,
                              config: ExplainabilityConfig) -> str:
    """Choose white or dark contour overlays from the underlying surface brightness."""
    finite = np.asarray(surface, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return config.contour_line_dark

    if np.isclose(finite.min(), finite.max()):
        sample_positions = np.array([0.5], dtype=float)
    else:
        sample_values = np.quantile(finite, [0.2, 0.5, 0.8])
        sample_positions = (sample_values - finite.min()) / (finite.max() - finite.min())

    cmap = plt.get_cmap(cmap_name)
    luminance = np.mean([relative_luminance(cmap(float(pos))) for pos in sample_positions])
    if luminance < 0.42:
        return config.contour_line_light
    return config.contour_line_dark


def choose_foreground_color_for_surface(surface: np.ndarray, cmap_name: str,
                                        config: ExplainabilityConfig,
                                        loc: str = 'upper right') -> str:
    """Choose a contrasting foreground color from the local surface background."""
    array = np.asarray(surface, dtype=float)
    if array.ndim != 2:
        return choose_contour_line_color(surface, cmap_name, config)

    row_extent = max(1, int(np.ceil(array.shape[0] * 0.30)))
    col_extent = max(1, int(np.ceil(array.shape[1] * 0.35)))
    row_slice = slice(0, row_extent) if 'upper' in loc else slice(array.shape[0] - row_extent, array.shape[0])
    col_slice = slice(array.shape[1] - col_extent, array.shape[1]) if 'right' in loc else slice(0, col_extent)
    local_surface = array[row_slice, col_slice]
    return choose_contour_line_color(local_surface, cmap_name, config)


def style_legend_for_surface(legend, surface: np.ndarray, cmap_name: str,
                             config: ExplainabilityConfig, loc: str = 'upper right'):
    """Apply local background-aware legend styling for contour panels."""
    foreground = choose_foreground_color_for_surface(surface, cmap_name, config, loc=loc)
    for text in legend.get_texts():
        text.set_color(foreground)
    for handle in legend.legend_handles:
        if hasattr(handle, 'set_markeredgecolor'):
            handle.set_markeredgecolor(foreground)
        if hasattr(handle, 'set_color') and isinstance(handle, Line2D) and handle.get_linestyle() != 'None':
            handle.set_color(foreground)


def pair_surface_balance(surface: np.ndarray) -> float:
    """Measure whether both axes materially influence the surface."""
    array = np.asarray(surface, dtype=float)
    row_effect = float(np.std(array.mean(axis=1)))
    col_effect = float(np.std(array.mean(axis=0)))
    return min(row_effect, col_effect) / (max(row_effect, col_effect) + 1e-9)


def pair_support_occupancy(x_values: np.ndarray, y_values: np.ndarray) -> int:
    """Estimate how many coarse 2D support bins are occupied by observed points."""
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if not np.any(mask):
        return 0
    points = np.column_stack([x_values[mask], y_values[mask]])
    x_bins = np.unique(np.quantile(points[:, 0], np.linspace(0, 1, 6)))
    y_bins = np.unique(np.quantile(points[:, 1], np.linspace(0, 1, 6)))
    if len(x_bins) < 2 or len(y_bins) < 2:
        return 1
    hist, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=[x_bins, y_bins])
    return int(np.sum(hist > 0))


def select_interaction_pairs(model, scaler, X: np.ndarray, feature_names: Sequence[str],
                             importance_df: pd.DataFrame, df: pd.DataFrame,
                             is_composite: bool, config: ExplainabilityConfig):
    """Choose interaction pairs as rank1×rank2, rank1×rank3, and rank1×rank4."""
    top_features = importance_df.head(4)['feature'].tolist()
    resolved = [
        (feat, resolve_feature_index(feat, feature_names), resolve_feature_full_name(feat, feature_names))
        for feat in top_features
    ]
    resolved = [(feat, idx, full) for feat, idx, full in resolved if idx >= 0]
    if len(resolved) < 2:
        return []
    anchor = resolved[0]
    return [
        (anchor, resolved[idx])
        for idx in range(1, min(len(resolved), config.n_top_pairs + 1))
    ]


def weighted_mean(values: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Compute a weighted mean with graceful fallback."""
    values = np.asarray(values, dtype=float)
    if weights is None:
        return float(np.mean(values))
    weights = np.asarray(weights, dtype=float)
    if not np.any(weights > 0):
        return float(np.mean(values))
    return float(np.average(values, weights=weights))


def weighted_percentile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Compute a weighted percentile with fallback for degenerate weights."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(values) == 0:
        return float('nan')
    if not np.any(weights > 0):
        return float(np.quantile(values, quantile))
    return weighted_quantile(values, weights, quantile)


def quantile_range(values: np.ndarray, config: ExplainabilityConfig,
                   weights: Optional[np.ndarray] = None,
                   pad_fraction: float = 0.05) -> Tuple[float, float]:
    """Return a padded, quantile-bounded plotting range."""
    values = np.asarray(values, dtype=float)
    if weights is not None:
        low = weighted_percentile(values, weights, config.support_quantile_low)
        high = weighted_percentile(values, weights, config.support_quantile_high)
    else:
        low = float(np.quantile(values, config.support_quantile_low))
        high = float(np.quantile(values, config.support_quantile_high))
    if not np.isfinite(low) or not np.isfinite(high):
        low = float(np.nanmin(values))
        high = float(np.nanmax(values))
    if np.isclose(low, high):
        span = max(abs(low), 1.0) * 0.1
        return low - span, high + span
    span = high - low
    return low - pad_fraction * span, high + pad_fraction * span


def infer_support_radius_1d(values: np.ndarray, config: ExplainabilityConfig) -> float:
    """Infer a reasonable one-dimensional support radius from observed spacing."""
    values = np.sort(np.unique(np.asarray(values, dtype=float)))
    if len(values) < 2:
        return 1.0
    diffs = np.diff(values)
    radius = float(np.quantile(diffs, 0.75) * config.support_radius_scale_1d)
    return max(radius, 1e-6)


def infer_support_mask_1d(grid: np.ndarray, values: np.ndarray, config: ExplainabilityConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Return support mask and local support counts for one-dimensional grids."""
    values = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype=float)
    radius = infer_support_radius_1d(values, config)
    distances = np.abs(grid[:, None] - values[None, :])
    local_counts = np.sum(distances <= radius, axis=1)
    min_support = max(3, int(np.ceil(0.04 * len(values))))
    supported = local_counts >= min_support
    return supported, local_counts


def infer_support_mask_2d(grid_x: np.ndarray, grid_y: np.ndarray,
                          obs_x: np.ndarray, obs_y: np.ndarray,
                          config: ExplainabilityConfig) -> np.ndarray:
    """Return a support mask over a 2D grid using normalized nearest-neighbor distance."""
    points = np.column_stack([np.asarray(obs_x, dtype=float), np.asarray(obs_y, dtype=float)])
    if len(points) < 4:
        return np.ones_like(grid_x, dtype=bool)

    spans = np.ptp(points, axis=0)
    spans = np.where(spans < 1e-6, 1.0, spans)
    points_norm = (points - points.min(axis=0)) / spans

    diffs = points_norm[:, None, :] - points_norm[None, :, :]
    dist = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dist, np.inf)
    nearest = np.min(dist, axis=1)
    threshold = float(np.quantile(nearest, 0.85) * config.support_radius_scale_2d)
    threshold = max(threshold, 0.05)

    grid_points = np.column_stack([
        (grid_x.ravel() - points[:, 0].min()) / spans[0],
        (grid_y.ravel() - points[:, 1].min()) / spans[1],
    ])
    dist_grid = np.linalg.norm(points_norm[None, :, :] - grid_points[:, None, :], axis=2)
    supported = np.min(dist_grid, axis=1) <= threshold
    return supported.reshape(grid_x.shape)


def overlay_source_points(ax: plt.Axes, x: np.ndarray, y: np.ndarray, source_df: pd.DataFrame,
                          config: ExplainabilityConfig, alpha_literature: float = 0.45,
                          alpha_wetlab: float = 0.95):
    """Overlay observed literature and wet-lab points on a 2D axis."""
    masks = source_masks(source_df)
    if masks['literature'].any():
        ax.scatter(
            x[masks['literature']],
            y[masks['literature']],
            c=config.color_literature,
            s=56,
            alpha=alpha_literature,
            edgecolors='white',
            linewidths=0.35,
            zorder=6,
        )
    if masks['wetlab'].any():
        ax.scatter(
            x[masks['wetlab']],
            y[masks['wetlab']],
            c=config.color_wetlab,
            s=84,
            alpha=alpha_wetlab,
            edgecolors='white',
            linewidths=0.55,
            zorder=7,
        )


def draw_support_histogram(ax: plt.Axes, values: np.ndarray, grid: np.ndarray,
                           config: ExplainabilityConfig):
    """Draw a low-contrast support histogram behind a 1D marginal plot."""
    ax_hist = ax.twinx()
    bins = min(20, max(8, len(np.unique(values)) // 2))
    counts, edges = np.histogram(values, bins=bins, range=(grid.min(), grid.max()))
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax_hist.fill_between(
        centers,
        np.zeros_like(centers),
        counts,
        color=config.support_fill,
        alpha=0.25,
        zorder=0,
    )
    ax_hist.set_ylim(0, max(counts.max(), 1) * 3.5)
    ax_hist.set_yticks([])
    for spine in ax_hist.spines.values():
        spine.set_visible(False)


def add_panel_subtitle(ax: plt.Axes, text: str):
    """Add a small descriptive subtitle above a panel."""
    ax.text(
        0.0,
        0.995,
        text,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=8.0,
        color='#50606d',
        bbox={
            'facecolor': 'white',
            'alpha': 0.72,
            'edgecolor': 'none',
            'boxstyle': 'round,pad=0.18',
        },
    )


def make_masked_colormap(name: str, bad_color: str = '#dde4ea'):
    """Return a colormap with a soft grey value for masked regions."""
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(bad_color)
    return cmap


def infer_reference_formulation(X: np.ndarray, weights: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    """Return a weighted-average reference formulation for slice visualizations."""
    ref = []
    for idx, _ in enumerate(feature_names):
        ref.append(weighted_mean(X[:, idx], weights))
    return normalize_formulation_vector(np.array(ref, dtype=float), feature_names)


def compute_bo_support_context(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                               feature_names: Sequence[str], support_scaler) -> Dict[str, object]:
    """Build the static support context used for the BO-score landscape."""
    observed_df = pd.DataFrame(X, columns=feature_names)
    observed_df['viability_percent'] = np.asarray(y, dtype=float)
    observed_df['context_weight'] = np.asarray(weights, dtype=float)
    observed_df['source'] = 'context'
    collapsed = collapse_observed_context_for_bo(observed_df, list(feature_names))
    if collapsed.empty:
        return {
            'collapsed_X': normalize_formulation_matrix(X, feature_names),
            'reference_ingredient_count': 1,
            'support_scaled': None,
            'support_radius': np.inf,
        }

    collapsed_X = normalize_formulation_matrix(collapsed[list(feature_names)].values, feature_names)
    counts = np.sum(np.abs(collapsed_X) > 1e-6, axis=1)
    weights_collapsed = collapsed['context_weight'].to_numpy(dtype=float)
    positive_counts = counts[counts > 0]
    positive_weights = weights_collapsed[counts > 0]
    if len(positive_counts):
        reference_count = int(round(weighted_quantile(positive_counts, positive_weights, 0.5)))
    else:
        reference_count = 1

    if support_scaler is None:
        return {
            'collapsed_X': collapsed_X,
            'reference_ingredient_count': reference_count,
            'support_scaled': None,
            'support_radius': np.inf,
        }

    support_scaled = support_scaler.transform(collapsed_X)
    if len(support_scaled) < 2:
        support_radius = np.inf
    else:
        diffs = support_scaled[:, None, :] - support_scaled[None, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(distances, np.inf)
        nearest = np.min(distances, axis=1)
        support_radius = float(
            weighted_quantile(nearest, weights_collapsed, 0.9) * BOConfig.support_radius_scale
        )

    return {
        'collapsed_X': collapsed_X,
        'reference_ingredient_count': reference_count,
        'support_scaled': support_scaled,
        'support_radius': support_radius,
    }


def expected_improvement(mean: np.ndarray, std: np.ndarray,
                         y_best: float, xi: float = 0.01) -> np.ndarray:
    """Calculate Expected Improvement."""
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    safe_std = np.maximum(std, 1e-12)
    z = (mean - y_best - xi) / safe_std
    ei = (mean - y_best - xi) * norm.cdf(z) + safe_std * norm.pdf(z)
    ei[std < 1e-9] = 0.0
    return ei


def upper_confidence_bound(mean: np.ndarray, std: np.ndarray, kappa: float = 0.5) -> np.ndarray:
    """Calculate Upper Confidence Bound."""
    return np.asarray(mean, dtype=float) + kappa * np.asarray(std, dtype=float)


def compute_feature_importance(model, scaler, feature_names: List[str],
                               X: np.ndarray, y: np.ndarray,
                               is_composite: bool = False,
                               weights: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Compute weighted permutation importance on the active model."""
    if weights is None:
        weights = np.ones(len(y))

    def weighted_r2(y_true, y_pred, w):
        ss_res = np.sum(w * (y_true - y_pred) ** 2)
        ss_tot = np.sum(w * (y_true - np.average(y_true, weights=w)) ** 2)
        return 1 - (ss_res / ss_tot)

    baseline = predict_model(model, scaler, X, is_composite)
    baseline_score = weighted_r2(y, baseline, weights)
    importance_scores = []
    for idx, name in enumerate(feature_names):
        X_perm = X.copy()
        np.random.seed(42)
        X_perm[:, idx] = np.random.permutation(X_perm[:, idx])
        permuted = predict_model(model, scaler, X_perm, is_composite)
        perm_score = weighted_r2(y, permuted, weights)
        importance_scores.append({
            'feature': name.replace('_M', '').replace('_pct', ''),
            'importance': baseline_score - perm_score,
        })

    importance_df = pd.DataFrame(importance_scores).sort_values('importance', ascending=False)
    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame, output_dir: str,
                            config: ExplainabilityConfig):
    """Create a polished feature-importance chart with dominant-feature emphasis."""
    df = importance_df.copy().sort_values('importance', ascending=False)
    top_n = min(config.n_top_features_overview, len(df))
    display = df.head(top_n).iloc[::-1].copy()
    values = display['importance'].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=config.figsize_medium)
    colors = plt.cm.RdYlGn(np.linspace(0.25, 0.85, len(display)))
    bars = ax.barh(range(len(display)), values, color=colors, edgecolor='white', linewidth=1.0, zorder=3)

    ax.set_yticks(range(len(display)))
    ax.set_yticklabels([clean_feature_name(name) for name in display['feature']])
    ax.set_xlabel('Weighted Permutation Importance')
    fig.suptitle('Feature Importance for Cell Viability Prediction', fontsize=19 + FONT_BUMP, fontweight='bold', y=0.955)
    ax.grid(axis='x', alpha=0.25)
    ax.grid(axis='y', visible=False)

    cutoff = values.max() * 0.25 if len(values) else 0.0
    if cutoff > 0:
        ax.axvline(cutoff, color='#90a3b1', linestyle='--', linewidth=1.0, alpha=0.6)

    for bar, val in zip(bars, values):
        ax.text(
            val + max(values.max() * 0.012, 0.003),
            bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}',
            va='center',
            fontsize=9 + FONT_BUMP,
            color='#2f3a42',
        )

    ax.set_xlim(0, max(values.max() * 1.18, 0.05))
    plt.tight_layout(rect=(0, 0, 1, 0.965))
    output_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"  ✓ Feature importance chart saved: {output_path}")


def compute_shap_values(model, scaler, X: np.ndarray, feature_names: Sequence[str],
                        is_composite: bool = False,
                        config: Optional[ExplainabilityConfig] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute SHAP values using KernelExplainer when SHAP is available."""
    config = config or ExplainabilityConfig()
    np.random.seed(42)
    n_background = min(config.n_shap_samples, len(X))
    bg_idx = np.random.choice(len(X), n_background, replace=False)
    X_background = X[bg_idx]

    def predict_fn(X_raw):
        return predict_model(model, scaler, X_raw, is_composite, return_std=False)

    try:
        import shap

        explainer = shap.KernelExplainer(predict_fn, X_background)
        n_explain = min(100, len(X))
        explain_idx = np.random.choice(len(X), n_explain, replace=False)
        X_explain = X[explain_idx]
        shap_values = explainer.shap_values(X_explain, silent=True)
        return shap_values, X_explain
    except ImportError:
        print("  ⚠ SHAP library not installed. Skipping SHAP analysis.")
        return None, None


def plot_shap_summary(shap_values: np.ndarray, X_explain: np.ndarray,
                      feature_names: Sequence[str], output_dir: str,
                      config: ExplainabilityConfig):
    """Create polished SHAP summary artifacts focused on the most informative features."""
    try:
        import shap

        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(-1, 1)
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_n = min(config.n_top_features_shap, len(feature_names))
        top_idx = np.argsort(mean_abs)[-top_n:]
        top_idx = top_idx[np.argsort(mean_abs[top_idx])]

        selected_features = [clean_feature_name(feature_names[idx]) for idx in top_idx]
        selected_X = X_explain[:, top_idx]
        selected_shap = shap_values[:, top_idx]

        fig = plt.figure(figsize=(config.figsize_medium[0], config.figsize_medium[1] * 1.15))
        shap.summary_plot(selected_shap, selected_X, feature_names=selected_features, show=False, plot_size=None)
        plt.title('SHAP Summary: Feature Impact on Viability', fontsize=17 + FONT_BUMP, fontweight='bold', pad=12)
        output_path = os.path.join(output_dir, 'shap_summary.png')
        plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', transparent=True)
        plt.close(fig)
        print(f"  ✓ SHAP summary plot saved: {output_path}")

        fig = plt.figure(figsize=config.figsize_small)
        shap.summary_plot(selected_shap, selected_X, feature_names=selected_features,
                          plot_type='bar', show=False, plot_size=None)
        plt.title('SHAP Feature Importance', fontsize=17 + FONT_BUMP, fontweight='bold', pad=12)
        output_path = os.path.join(output_dir, 'shap_importance.png')
        plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', transparent=True)
        plt.close(fig)
        print(f"  ✓ SHAP importance plot saved: {output_path}")
    except Exception as exc:
        print(f"  ⚠ Error creating SHAP plots: {exc}")


def plot_partial_dependence(model, scaler, X: np.ndarray, feature_names: Sequence[str],
                            importance_df: pd.DataFrame, df: pd.DataFrame,
                            output_dir: str, is_composite: bool,
                            config: ExplainabilityConfig):
    """Create support-aware empirical marginal plots for the top features."""
    top_features = importance_df.head(config.n_top_features_pdp)['feature'].tolist()
    weights = df['context_weight'].to_numpy(dtype=float) if 'context_weight' in df.columns else np.ones(len(df))

    n_cols = 2
    n_rows = int(np.ceil(len(top_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.8 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for panel_idx, feature in enumerate(top_features):
        ax = axes[panel_idx]
        feat_idx = resolve_feature_index(feature, feature_names)
        full_name = resolve_feature_full_name(feature, feature_names)
        if feat_idx < 0:
            ax.set_visible(False)
            continue

        feature_values = X[:, feat_idx]
        x_min, x_max = quantile_range(feature_values, config, weights=weights)
        grid = np.linspace(x_min, x_max, config.n_pdp_points)

        predictions = []
        for val in grid:
            X_temp = X.copy()
            X_temp[:, feat_idx] = val
            pred = predict_model(model, scaler, X_temp, is_composite)
            predictions.append(pred)
        predictions = np.vstack(predictions)

        mean_curve = np.average(predictions, axis=1, weights=weights)
        lower_curve = np.quantile(predictions, 0.20, axis=1)
        upper_curve = np.quantile(predictions, 0.80, axis=1)
        supported, local_counts = infer_support_mask_1d(grid, feature_values, config)

        ax.fill_between(grid, lower_curve, upper_curve, color=config.line_primary, alpha=0.14, zorder=2)
        ax.plot(grid[supported], mean_curve[supported], color=config.line_primary, linewidth=2.8, zorder=4)
        if (~supported).any():
            ax.plot(grid[~supported], mean_curve[~supported], color=config.line_primary,
                    linewidth=2.2, linestyle='--', alpha=0.45, zorder=3)

        y_floor = float(min(lower_curve.min(), mean_curve.min()))
        rug_y = y_floor - 0.025 * max(1.0, upper_curve.max() - y_floor)
        masks = source_masks(df)
        if masks['literature'].any():
            ax.scatter(feature_values[masks['literature']], np.full(np.sum(masks['literature']), rug_y),
                       marker='|', s=90, linewidths=1.2, color=config.color_literature, alpha=0.35, zorder=5)
        if masks['wetlab'].any():
            ax.scatter(feature_values[masks['wetlab']], np.full(np.sum(masks['wetlab']), rug_y),
                       marker='|', s=110, linewidths=1.4, color=config.color_wetlab, alpha=0.85, zorder=6)

        ax.set_xlim(grid.min(), grid.max())
        ax.set_xlabel(f'{clean_feature_name(full_name)} ({get_unit(full_name)})')
        ax.set_ylabel('Predicted Viability (%)')
        ax.set_title(f'Empirical Marginal: {clean_feature_name(full_name)}', fontsize=13 + FONT_BUMP, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.22)

        legend_items = [
            Line2D([0], [0], color=config.line_primary, lw=2.8, label='Weighted mean response'),
            Line2D([0], [0], color=config.line_primary, lw=8, alpha=0.14, label='20-80% response band'),
        ] + source_legend_handles(config)
        ax.legend(handles=legend_items, loc='upper left', fontsize=8 + FONT_BUMP)

    for idx in range(len(top_features), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Support-Aware Empirical Marginals', fontsize=19 + FONT_BUMP, fontweight='bold', y=0.985)
    plt.tight_layout(rect=(0, 0, 1, 0.975))
    output_path = os.path.join(output_dir, 'partial_dependence_plots.png')
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"  ✓ Partial dependence plots saved: {output_path}")


def plot_interaction_contours(model, scaler, X: np.ndarray, feature_names: Sequence[str],
                              importance_df: pd.DataFrame, df: pd.DataFrame,
                              output_dir: str, is_composite: bool,
                              config: ExplainabilityConfig):
    """Create support-aware interaction contour plots for the strongest feature pairs."""
    pairs = select_interaction_pairs(model, scaler, X, feature_names, importance_df, df, is_composite, config)
    if not pairs:
        return

    fig, axes = plt.subplots(1, len(pairs), figsize=(6.4 * len(pairs), 5.9))
    axes = np.atleast_1d(axes)
    weights = df['context_weight'].to_numpy(dtype=float) if 'context_weight' in df.columns else np.ones(len(df))
    reference = infer_reference_formulation(X, weights, feature_names)
    viability_cmap = config.cmap_viability

    for ax, ((feat1, idx1, full1), (feat2, idx2, full2)) in zip(axes, pairs):
        weights = df['context_weight'].to_numpy(dtype=float) if 'context_weight' in df.columns else np.ones(len(df))
        x_min, x_max = quantile_range(X[:, idx1], config, weights=weights)
        y_min, y_max = quantile_range(X[:, idx2], config, weights=weights)
        x_range = np.linspace(x_min, x_max, config.n_contour_points)
        y_range = np.linspace(y_min, y_max, config.n_contour_points)
        X1, X2 = np.meshgrid(x_range, y_range)

        Z = np.zeros_like(X1, dtype=float)
        for row_idx in range(X1.shape[0]):
            for col_idx in range(X1.shape[1]):
                X_temp = reference.copy()
                X_temp[idx1] = X1[row_idx, col_idx]
                X_temp[idx2] = X2[row_idx, col_idx]
                X_temp = normalize_formulation_vector(X_temp, feature_names)
                Z[row_idx, col_idx] = predict_model(model, scaler, X_temp.reshape(1, -1), is_composite)[0]

        support_mask = infer_support_mask_2d(X1, X2, X[:, idx1], X[:, idx2], config)
        contour = ax.contourf(X1, X2, Z, levels=18, cmap=viability_cmap)
        contour_line_color = choose_contour_line_color(Z, viability_cmap, config)
        ax.contour(X1, X2, Z, levels=9, colors=contour_line_color, linewidths=2, alpha=0.22)
        ax.contour(X1, X2, support_mask.astype(float), levels=[0.5], colors=contour_line_color,
                   linewidths=2, alpha=0.82, linestyles='--', zorder=5)
        plt.colorbar(contour, ax=ax, label='Predicted Viability (%)')

        overlay_source_points(ax, X[:, idx1], X[:, idx2], df, config)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(f'{clean_feature_name(full1)} ({get_unit(full1)})')
        ax.set_ylabel(f'{clean_feature_name(full2)} ({get_unit(full2)})')
        ax.set_title(f'{clean_feature_name(full1)} × {clean_feature_name(full2)}', fontsize=13 + FONT_BUMP, fontweight='bold', pad=10)
        legend_loc = 'upper right'
        legend = ax.legend(
            handles=source_legend_handles(
                config,
                edge_color=choose_foreground_color_for_surface(Z, viability_cmap, config, loc=legend_loc),
            ),
            loc=legend_loc,
        )
        style_legend_for_surface(legend, Z, viability_cmap, config, loc=legend_loc)

    plt.suptitle('Support-Aware Ingredient Interaction Effects on Cell Viability',
                 fontsize=19 + FONT_BUMP, fontweight='bold', y=0.985)
    plt.tight_layout(rect=(0, 0, 1, 0.975))
    output_path = os.path.join(output_dir, 'interaction_contours.png')
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"  ✓ Interaction contour plots saved: {output_path}")


def plot_acquisition_landscape(model, scaler, X: np.ndarray, y: np.ndarray,
                               feature_names: Sequence[str], importance_df: pd.DataFrame,
                               df: pd.DataFrame, output_dir: str, is_composite: bool,
                               config: ExplainabilityConfig):
    """Create a support-aware static BO-score landscape using the acquisition visual language."""
    top_features = importance_df.head(2)['feature'].tolist()
    feat1, feat2 = top_features[0], top_features[1]
    idx1 = resolve_feature_index(feat1, feature_names)
    idx2 = resolve_feature_index(feat2, feature_names)
    full1 = resolve_feature_full_name(feat1, feature_names)
    full2 = resolve_feature_full_name(feat2, feature_names)

    weights = df['context_weight'].to_numpy(dtype=float) if 'context_weight' in df.columns else np.ones(len(df))
    reference = infer_reference_formulation(X, weights, feature_names)
    support_scaler = getattr(model, 'scaler_literature', None) if is_composite else scaler
    support_ctx = compute_bo_support_context(X, y, weights, feature_names, support_scaler)
    y_pred = predict_model(model, scaler, X, is_composite)
    y_best = float(np.max(y_pred))

    x_min, x_max = quantile_range(X[:, idx1], config, weights=weights)
    y_min, y_max = quantile_range(X[:, idx2], config, weights=weights)
    x_range = np.linspace(x_min, x_max, config.n_contour_points)
    y_range = np.linspace(y_min, y_max, config.n_contour_points)
    X1, X2 = np.meshgrid(x_range, y_range)

    Z_mean = np.zeros_like(X1, dtype=float)
    Z_std = np.zeros_like(X1, dtype=float)
    Z_score = np.zeros_like(X1, dtype=float)
    support_mask = infer_support_mask_2d(X1, X2, X[:, idx1], X[:, idx2], config)

    for row_idx in range(X1.shape[0]):
        for col_idx in range(X1.shape[1]):
            X_temp = reference.copy()
            X_temp[idx1] = X1[row_idx, col_idx]
            X_temp[idx2] = X2[row_idx, col_idx]
            X_temp = normalize_formulation_vector(X_temp, feature_names)

            mean, std = predict_model(model, scaler, X_temp.reshape(1, -1), is_composite, return_std=True)
            mean_value = float(mean[0])
            std_value = float(std[0])
            Z_mean[row_idx, col_idx] = mean_value
            Z_std[row_idx, col_idx] = std_value

            if config.acquisition_mode.lower() == 'ei':
                acquisition = float(expected_improvement(np.array([mean_value]), np.array([std_value]), y_best,
                                                        xi=config.acquisition_xi)[0])
            else:
                acquisition = float(upper_confidence_bound(np.array([mean_value]), np.array([std_value]),
                                                           kappa=config.acquisition_kappa)[0])

            n_ingredients = int(np.sum(np.abs(X_temp) > 1e-6))
            complexity_penalty = config.bo_sparsity_penalty * max(
                0, n_ingredients - int(support_ctx['reference_ingredient_count'])
            )

            support_penalty = 0.0
            if support_ctx['support_scaled'] is not None and np.isfinite(float(support_ctx['support_radius'])):
                X_scaled = support_scaler.transform(X_temp.reshape(1, -1))
                min_distance = float(np.min(np.linalg.norm(support_ctx['support_scaled'] - X_scaled, axis=1)))
                if min_distance > float(support_ctx['support_radius']):
                    overshoot = min_distance - float(support_ctx['support_radius'])
                    support_penalty = config.bo_support_penalty * overshoot * overshoot

            Z_score[row_idx, col_idx] = acquisition - complexity_penalty - support_penalty

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    contour1 = axes[0].contourf(X1, X2, Z_mean, levels=18, cmap=config.cmap_viability)
    contour2 = axes[1].contourf(X1, X2, Z_std, levels=18, cmap=config.cmap_uncertainty)
    contour3 = axes[2].contourf(X1, X2, Z_score, levels=18, cmap=config.cmap_acquisition)

    for ax, contour, surface, title, label, cmap_name in [
        (
            axes[0],
            contour1,
            Z_mean,
            'GP Mean Prediction',
            'Predicted Viability (%)',
            config.cmap_viability,
        ),
        (
            axes[1],
            contour2,
            Z_std,
            'GP Uncertainty',
            'Uncertainty (std)',
            config.cmap_uncertainty,
        ),
        (
            axes[2],
            contour3,
            Z_score,
            'Static BO Score (UCB - penalties)',
            'Static BO Score',
            config.cmap_acquisition,
        ),
    ]:
        contour_line_color = choose_contour_line_color(surface, cmap_name, config)
        ax.contour(X1, X2, surface, levels=9, colors=contour_line_color, linewidths=2, alpha=0.18)
        ax.contour(X1, X2, support_mask.astype(float), levels=[0.5], colors=contour_line_color,
                   linewidths=2, alpha=0.82, linestyles='--', zorder=5)
        overlay_source_points(ax, X[:, idx1], X[:, idx2], df, config, alpha_literature=0.35, alpha_wetlab=0.9)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        plt.colorbar(contour, ax=ax, label=label)
        ax.set_xlabel(f'{clean_feature_name(full1)} ({get_unit(full1)})')
        ax.set_ylabel(f'{clean_feature_name(full2)} ({get_unit(full2)})')
        ax.set_title(title, fontsize=14 + FONT_BUMP, fontweight='bold', pad=10)
        legend_loc = 'upper right'
        legend = ax.legend(
            handles=source_legend_handles(
                config,
                edge_color=choose_foreground_color_for_surface(surface, cmap_name, config, loc=legend_loc),
            ),
            loc=legend_loc,
        )
        style_legend_for_surface(legend, surface, cmap_name, config, loc=legend_loc)

    plt.suptitle('Acquisition Function Landscape: Exploration vs Exploitation',
                 fontsize=20 + FONT_BUMP, fontweight='bold', y=0.985)
    plt.tight_layout(rect=(0, 0, 1, 0.975))
    output_path = os.path.join(output_dir, 'acquisition_landscape.png')
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"  ✓ Acquisition landscape saved: {output_path}")


def plot_uncertainty_analysis(model, scaler, X: np.ndarray, y: np.ndarray,
                              df: pd.DataFrame, output_dir: str,
                              is_composite: bool, config: ExplainabilityConfig):
    """Create a calibrated, decision-oriented uncertainty dashboard."""
    y_pred, y_std = predict_model(model, scaler, X, is_composite, return_std=True)
    y_pred = np.asarray(y_pred, dtype=float)
    y_std = np.asarray(y_std, dtype=float)
    residuals = y - y_pred
    masks = source_masks(df)

    fig, axes = plt.subplots(2, 2, figsize=config.figsize_dashboard)

    ax = axes[0, 0]
    ax.plot([0, 100], [0, 100], linestyle='--', color='#4a5966', alpha=0.7, linewidth=2)
    if masks['literature'].any():
        sc = ax.scatter(y[masks['literature']], y_pred[masks['literature']], c=y_std[masks['literature']],
                        cmap=config.cmap_uncertainty, s=40, alpha=0.20, edgecolors='white', linewidths=0.35)
    else:
        sc = ax.scatter(y, y_pred, c=y_std, cmap=config.cmap_uncertainty, s=36, alpha=0.5,
                        edgecolors='white', linewidths=0.35)
    if masks['wetlab'].any():
        ax.scatter(y[masks['wetlab']], y_pred[masks['wetlab']], c=y_std[masks['wetlab']],
                   cmap=config.cmap_uncertainty, s=80, alpha=0.95, edgecolors='white', linewidths=0.55)
    plt.colorbar(sc, ax=ax, label='Uncertainty (std)')
    ax.set_xlabel('Actual Viability (%)')
    ax.set_ylabel('Predicted Viability (%)')
    ax.set_title('Predicted vs Actual', fontsize=14 + FONT_BUMP, fontweight='bold', pad=10)
    ax.legend(handles=alpha_legend_handles(config), loc='upper left')

    ax = axes[0, 1]
    multipliers = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 2.5])
    empirical = np.array([np.mean(np.abs(residuals) <= k * y_std) for k in multipliers], dtype=float)
    nominal = np.array([norm.cdf(k) - norm.cdf(-k) for k in multipliers], dtype=float)
    ax.plot(multipliers, nominal, color='#4a5966', linewidth=2.0, linestyle='--', label='Ideal Gaussian coverage')
    ax.plot(multipliers, empirical, color=config.line_primary, linewidth=2.8, marker='o', label='Empirical coverage')
    ax.set_xlabel('Uncertainty Multiplier')
    ax.set_ylabel('Coverage Fraction')
    ax.set_ylim(0.0, 1.05)
    ax.set_title('Calibration Curve', fontsize=14 + FONT_BUMP, fontweight='bold', pad=10)
    ax.legend(loc='lower right')

    ax = axes[1, 0]
    if masks['literature'].any():
        ax.scatter(y_std[masks['literature']], np.abs(residuals[masks['literature']]),
                   c=config.color_literature, s=50, alpha=0.4, edgecolors='white', linewidths=0.35)
    if masks['wetlab'].any():
        ax.scatter(y_std[masks['wetlab']], np.abs(residuals[masks['wetlab']]),
                   c=config.color_wetlab, s=100, alpha=0.9, edgecolors='white', linewidths=0.55)
    z = np.polyfit(y_std, np.abs(residuals), 1)
    trend = np.poly1d(z)
    x_line = np.linspace(y_std.min(), y_std.max(), 200)
    ax.plot(x_line, trend(x_line), linestyle='--', color=config.line_primary, linewidth=2.4, label='Trend')
    ax.set_xlabel('Prediction Uncertainty (std)')
    ax.set_ylabel('Absolute Error (%)')
    ax.set_title('Error vs Uncertainty', fontsize=14 + FONT_BUMP, fontweight='bold', pad=10)
    ax.legend(handles=source_legend_handles(config) + [
        Line2D([0], [0], color=config.line_primary, linestyle='--', lw=2.4, label='Trend')
    ], loc='upper left')

    ax = axes[1, 1]
    bins = [(0, 30), (30, 50), (50, 70), (70, 90), (90, 101)]
    labels = ['0-30%', '30-50%', '50-70%', '70-90%', '90-100%']
    wetlab_means = []
    overall_means = []
    for low, high in bins:
        mask = (y >= low) & (y < high)
        wetlab_mask = mask & masks['wetlab']
        overall_means.append(float(np.mean(y_std[mask])) if np.any(mask) else np.nan)
        wetlab_means.append(float(np.mean(y_std[wetlab_mask])) if np.any(wetlab_mask) else np.nan)
    x_pos = np.arange(len(labels))
    width = 0.36
    bars1 = ax.bar(x_pos - width / 2, overall_means, width=width, color='#96b6c7', edgecolor='white', label='All context')
    bars2 = ax.bar(x_pos + width / 2, wetlab_means, width=width, color=config.color_wetlab, edgecolor='white',
                   alpha=0.85, label='Wet lab only')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Mean Uncertainty (std)')
    ax.set_title('Uncertainty by Viability Band', fontsize=14 + FONT_BUMP, fontweight='bold', pad=10)
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            if np.isfinite(height):
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.25, f'{height:.1f}',
                        ha='center', va='bottom', fontsize=8.5 + FONT_BUMP)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.24), ncol=2)

    model_label = 'Composite' if is_composite else 'GP'
    plt.suptitle(f'{model_label} Model Uncertainty Analysis', fontsize=19 + FONT_BUMP, fontweight='bold', y=0.985)
    plt.tight_layout(rect=(0, 0, 1, 0.975))
    output_path = os.path.join(output_dir, 'uncertainty_analysis.png')
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"  ✓ Uncertainty analysis saved: {output_path}")


def plot_support_diagnostics(X: np.ndarray, y: np.ndarray, feature_names: Sequence[str],
                             importance_df: pd.DataFrame, df: pd.DataFrame,
                             output_dir: str, config: ExplainabilityConfig):
    """Create a compact support diagnostic figure for the top features and pair."""
    top_features = importance_df.head(2)['feature'].tolist()
    if len(top_features) < 2:
        return
    idx1 = resolve_feature_index(top_features[0], feature_names)
    idx2 = resolve_feature_index(top_features[1], feature_names)
    full1 = resolve_feature_full_name(top_features[0], feature_names)
    full2 = resolve_feature_full_name(top_features[1], feature_names)

    masks = source_masks(df)
    weights = df['context_weight'].to_numpy(dtype=float) if 'context_weight' in df.columns else np.ones(len(df))
    fig = plt.figure(figsize=config.figsize_dashboard)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    for ax, idx, full in [(ax1, idx1, full1), (ax2, idx2, full2)]:
        values = X[:, idx]
        x_min, x_max = quantile_range(values, config, weights=weights)
        literature_values = values[masks['literature']]
        wetlab_values = values[masks['wetlab']]
        support_color = config.line_primary
        if HAS_SEABORN:
            sns.kdeplot(x=literature_values, ax=ax, fill=True, alpha=0.3,
                        color=support_color, linewidth=1.2, label='Literature')
            if len(wetlab_values) >= 2:
                sns.kdeplot(x=wetlab_values, ax=ax, fill=True, alpha=0.8,
                            color=support_color, linewidth=1.2, label='Wet Lab')
        else:
            ax.hist(literature_values, bins=24, density=True, alpha=0.3,
                    color=support_color, label='Literature')
            if len(wetlab_values) >= 1:
                ax.hist(wetlab_values, bins=min(18, max(6, len(wetlab_values))), density=True, alpha=0.8,
                        color=support_color, label='Wet Lab')
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(f'{clean_feature_name(full)} ({get_unit(full)})')
        ax.set_ylabel('Observed density')
        ax.set_title(f'Support: {clean_feature_name(full)}', fontsize=13 + FONT_BUMP, fontweight='bold', pad=10)
        ax.legend(
            handles=support_diagnostic_density_legend_handles(config, base_color=support_color),
            loc='upper right',
            **support_diagnostic_legend_kwargs(config),
        )

    scatter = ax3.scatter(X[:, idx1], X[:, idx2], c=y, cmap=config.cmap_viability, s=180,
                          alpha=0.20, edgecolors='white', linewidths=0.35)
    if masks['wetlab'].any():
        ax3.scatter(X[masks['wetlab'], idx1], X[masks['wetlab'], idx2], c=y[masks['wetlab']],
                    cmap=config.cmap_viability, s=300, alpha=0.95, edgecolors='black', linewidths=0.6)
    x_min, x_max = quantile_range(X[:, idx1], config, weights=weights)
    y_min, y_max = quantile_range(X[:, idx2], config, weights=weights)
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    plt.colorbar(scatter, ax=ax3, label='Observed Viability (%)')
    ax3.set_xlabel(f'{clean_feature_name(full1)} ({get_unit(full1)})')
    ax3.set_ylabel(f'{clean_feature_name(full2)} ({get_unit(full2)})')
    ax3.set_title(f'Observed Support Map: {clean_feature_name(full1)} × {clean_feature_name(full2)}',
                  fontsize=14 + FONT_BUMP, fontweight='bold', pad=10)
    ax3.legend(
        handles=alpha_legend_handles(config, marker_scale=config.support_diagnostic_legend_scale),
        loc='upper right',
        **support_diagnostic_legend_kwargs(config),
    )

    plt.suptitle('Support Diagnostics', fontsize=19 + FONT_BUMP, fontweight='bold', y=0.985)
    plt.tight_layout(rect=(0, 0, 1, 0.975))
    output_path = os.path.join(output_dir, 'support_diagnostics.png')
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"  ✓ Support diagnostics saved: {output_path}")


def main():
    """Main entry point for explainability visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    base_output_dir = os.path.join(project_root, 'results', 'explainability')
    os.makedirs(base_output_dir, exist_ok=True)

    config = ExplainabilityConfig()
    apply_publication_style(config)

    print("=" * 80)
    print("CryoMN Model Explainability Analysis")
    print("=" * 80)
    print("\n📊 Loading model and data...")
    try:
        gp, scaler, feature_names, df, importance_df, is_composite, resolution = load_model_and_data(project_root)
    except ModelResolutionError as exc:
        print(f"ERROR: {exc}")
        return

    X = normalize_formulation_matrix(df[feature_names].to_numpy(dtype=float), feature_names)
    y = df['viability_percent'].to_numpy(dtype=float)
    weights = df['context_weight'].to_numpy(dtype=float) if 'context_weight' in df.columns else np.ones(len(df))

    print(f"  Model loaded with {len(feature_names)} features")
    print(f"  Data loaded with {len(df)} formulations")
    if resolution.iteration_dir:
        print(f"  Resolved active iteration: {resolution.iteration_dir}")
    elif resolution.iteration is not None:
        print(f"  Resolved active iteration: iteration_{resolution.iteration}")
    if 'source' in df.columns:
        n_lit = int((df['source'] == 'literature').sum())
        n_wet = int((df['source'] == 'wetlab').sum())
        print(f"  Sources: {n_lit} literature + {n_wet} wet lab")

    output_dir = build_explainability_output_dir(base_output_dir, resolution.iteration_dir, resolution.iteration)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n  Saving explainability outputs to: {output_dir}")

    print("\n0️⃣  Computing Feature Importance (permutation-based)")
    importance_df = compute_feature_importance(gp, scaler, feature_names, X, y, is_composite, weights)
    importance_csv_path = os.path.join(output_dir, 'feature_importance.csv')
    importance_df.to_csv(importance_csv_path, index=False)
    print(f"  ✓ Feature importance saved: {importance_csv_path}")

    print("\n📈 Generating visualizations...\n")
    print("1️⃣  Feature Importance Bar Chart")
    plot_feature_importance(importance_df, output_dir, config)

    print("\n2️⃣  SHAP Values Analysis")
    shap_values, X_explain = compute_shap_values(gp, scaler, X, feature_names, is_composite, config)
    if shap_values is not None:
        plot_shap_summary(shap_values, X_explain, feature_names, output_dir, config)

    print("\n3️⃣  Partial Dependence Plots")
    plot_partial_dependence(gp, scaler, X, feature_names, importance_df, df, output_dir, is_composite, config)

    print("\n4️⃣  2D Interaction Contour Plots")
    plot_interaction_contours(gp, scaler, X, feature_names, importance_df, df, output_dir, is_composite, config)

    print("\n5️⃣  Acquisition Function Landscape")
    plot_acquisition_landscape(gp, scaler, X, y, feature_names, importance_df, df, output_dir, is_composite, config)

    print("\n6️⃣  GP Uncertainty Visualization")
    plot_uncertainty_analysis(gp, scaler, X, y, df, output_dir, is_composite, config)

    print("\n7️⃣  Support Diagnostics")
    plot_support_diagnostics(X, y, feature_names, importance_df, df, output_dir, config)

    print("\n" + "=" * 80)
    print("✅ Explainability Analysis Complete!")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    for file_name in sorted(os.listdir(output_dir)):
        if file_name.endswith('.png') or file_name.endswith('.csv'):
            print(f"  • {file_name}")


if __name__ == '__main__':
    main()
