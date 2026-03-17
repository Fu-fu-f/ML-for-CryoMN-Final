#!/usr/bin/env python3
"""
CryoMN Bayesian Optimization with Differential Evolution

Proper Bayesian optimization using DE to maximize acquisition functions.
This provides better exploration-exploitation balance compared to random sampling.

Author: CryoMN ML Project
Date: 2026-01-26
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import Tuple, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution
from scipy.stats import norm

# Add shared helper modules to path for model resolution and observed-context loading
_script_dir = os.path.dirname(os.path.abspath(__file__))
_helper_dir = os.path.join(os.path.dirname(_script_dir), 'helper')
if _helper_dir not in sys.path:
    sys.path.insert(0, _helper_dir)
from active_model_resolver import ModelResolutionError, resolve_active_model  # noqa: E402
from formulation_formatting import format_formulation  # noqa: E402
from observed_context import (  # noqa: E402
    collapse_observed_context_for_bo,
    load_observed_context,
    weighted_quantile,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BOConfig:
    """Configuration for Bayesian optimization with DE."""
    max_ingredients: Optional[int] = None  # None = infer from observed formulations
    max_dmso_percent: float = 5.0  # Set to 0.5 for low-DMSO mode
    min_viability: float = 70.0
    n_candidates: int = 20
    acquisition: str = 'ucb'  # 'ei' or 'ucb'
    xi: float = 0.01  # EI exploration parameter
    kappa: float = 0.5  # UCB exploration parameter
    de_maxiter: int = 100  # DE iterations per candidate
    de_popsize: int = 15  # DE population size
    random_seed: int = 42
    diversity_penalty: float = 5.0  # Strength of local penalization for batch diversity
    diversity_radius: float = 0.05  # Fraction of feature range (reduced to stay on the predictive peak)
    sparsity_penalty: float = 0.35  # Mild preference for simpler formulations on flat plateaus
    support_penalty: float = 4.0  # Penalize candidates that drift far from observed support
    support_radius_scale: float = 1.25  # Slack multiplier on observed nearest-neighbor radius


# =============================================================================
# ACQUISITION FUNCTIONS
# =============================================================================

def expected_improvement(x: np.ndarray, gp, 
                         scaler: StandardScaler, y_best: float, 
                         xi: float = 0.01, is_composite: bool = False) -> float:
    """
    Calculate Expected Improvement for a single point.
    
    Args:
        x: Formulation vector (unscaled)
        gp: Trained Gaussian Process (or CompositeGP)
        scaler: Feature scaler (unused if is_composite)
        y_best: Best observed viability
        xi: Exploration-exploitation trade-off
        is_composite: If True, skip external scaling (model handles it)
        
    Returns:
        Negative EI (for minimization)
    """
    x_reshaped = x.reshape(1, -1)
    if is_composite:
        mean, std = gp.predict(x_reshaped, return_std=True)
    else:
        x_scaled = scaler.transform(x_reshaped)
        mean, std = gp.predict(x_scaled, return_std=True)
    mean, std = mean[0], std[0]
    
    # Handle zero variance
    if std < 1e-9:
        return 0.0
    
    z = (mean - y_best - xi) / std
    ei = (mean - y_best - xi) * norm.cdf(z) + std * norm.pdf(z)
    
    return ei


def upper_confidence_bound(x: np.ndarray, gp,
                           scaler: StandardScaler, kappa: float = 1.96,
                           is_composite: bool = False) -> float:
    """
    Calculate Upper Confidence Bound for a single point.
    
    Args:
        x: Formulation vector (unscaled)
        gp: Trained Gaussian Process (or CompositeGP)
        scaler: Feature scaler (unused if is_composite)
        kappa: Exploration weight
        is_composite: If True, skip external scaling (model handles it)
        
    Returns:
        UCB value
    """
    x_reshaped = x.reshape(1, -1)
    if is_composite:
        mean, std = gp.predict(x_reshaped, return_std=True)
    else:
        x_scaled = scaler.transform(x_reshaped)
        mean, std = gp.predict(x_scaled, return_std=True)
    mean, std = mean[0], std[0]
    
    return mean + kappa * std


# =============================================================================
# CONSTRAINT HANDLING
# =============================================================================

def count_nonzero(x: np.ndarray, threshold: float = 1e-6) -> int:
    """Count non-zero ingredients."""
    return np.sum(np.abs(x) > threshold)


def ingredient_constraint(x: np.ndarray, max_ingredients: int) -> float:
    """Constraint: n_ingredients <= max_ingredients. Returns >=0 if satisfied."""
    return max_ingredients - count_nonzero(x)


def dmso_constraint(x: np.ndarray, dmso_index: int, max_dmso_molar: float) -> float:
    """Constraint: DMSO <= max. Returns >=0 if satisfied."""
    if dmso_index < 0:
        return 1.0  # No DMSO feature, constraint satisfied
    return max_dmso_molar - x[dmso_index]


# =============================================================================
# DE-BASED OPTIMIZER
# =============================================================================

class BayesianOptimizer:
    """
    Bayesian Optimizer using Differential Evolution.
    
    Uses DE to maximize the configured acquisition function,
    providing proper exploration-exploitation balance.
    """
    
    def __init__(self, gp, scaler: StandardScaler,
                 feature_names: List[str], config: BOConfig = None,
                 is_composite: bool = False):
        """
        Initialize optimizer.
        
        Args:
            gp: Trained Gaussian Process model (or CompositeGP)
            scaler: Feature scaler (unused if is_composite)
            feature_names: List of feature names
            config: Optimization configuration
            is_composite: If True, model handles scaling internally
        """
        self.gp = gp
        self.scaler = scaler
        self.feature_names = feature_names
        self.config = config or BOConfig()
        self.is_composite = is_composite
        
        # Find DMSO index
        self.dmso_index = -1
        for i, name in enumerate(feature_names):
            if 'dmso' in name.lower():
                self.dmso_index = i
                break
        
        # Calculate max DMSO in molar (5% v/v ≈ 0.70 M)
        self.max_dmso_molar = (self.config.max_dmso_percent / 100.0) * 1.10 * 1000 / 78.13
        
        # Set feature bounds
        self.bounds = self._get_feature_bounds()
        self.effective_max_ingredients = len(self.feature_names)
        self.reference_ingredient_count = 1
        self.support_scaler = self._get_support_scaler()
        self.observed_support_scaled: Optional[np.ndarray] = None
        self.support_radius = np.inf
        self.seed_context: Optional[pd.DataFrame] = None
        
        np.random.seed(self.config.random_seed)

    def _get_support_scaler(self):
        """Return the scaler used to measure distance from observed support."""
        if self.is_composite:
            return getattr(self.gp, 'scaler_literature', None)
        return self.scaler

    def _fit_search_context(self, observed_df: pd.DataFrame):
        """Derive realistic sparsity/support constraints from observed formulations."""
        collapsed = collapse_observed_context_for_bo(observed_df, self.feature_names)
        self.seed_context = collapsed

        if len(collapsed) == 0:
            requested = self.config.max_ingredients
            self.effective_max_ingredients = max(1, requested or len(self.feature_names))
            self.reference_ingredient_count = min(2, self.effective_max_ingredients)
            self.observed_support_scaled = None
            self.support_radius = np.inf
            return

        X_observed = collapsed[self.feature_names].values
        context_weights = collapsed['context_weight'].to_numpy(dtype=float)
        observed_counts = np.array([count_nonzero(row) for row in X_observed], dtype=int)
        observed_nonzero = observed_counts[observed_counts > 0]
        observed_nonzero_weights = context_weights[observed_counts > 0]
        observed_max = int(observed_nonzero.max()) if len(observed_nonzero) else 1

        requested = self.config.max_ingredients
        if requested is None:
            self.effective_max_ingredients = observed_max
        else:
            self.effective_max_ingredients = max(1, min(requested, observed_max))

        if len(observed_nonzero):
            self.reference_ingredient_count = int(
                round(weighted_quantile(observed_nonzero, observed_nonzero_weights, 0.5))
            )
        else:
            self.reference_ingredient_count = 1

        if self.support_scaler is None:
            self.observed_support_scaled = None
            self.support_radius = np.inf
            return

        self.observed_support_scaled = self.support_scaler.transform(X_observed)
        if len(self.observed_support_scaled) < 2:
            self.support_radius = np.inf
            return

        # Calibrate the acceptable radius from the observed nearest-neighbor distances.
        diffs = self.observed_support_scaled[:, None, :] - self.observed_support_scaled[None, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(distances, np.inf)
        nearest_distances = np.min(distances, axis=1)
        self.support_radius = float(
            weighted_quantile(nearest_distances, context_weights, 0.9) * self.config.support_radius_scale
        )
    
    def _get_feature_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for each feature based on typical concentration ranges."""
        bounds = []
        for name in self.feature_names:
            name_lower = name.lower()
            if 'dmso' in name_lower:
                bounds.append((0.0, self.max_dmso_molar))
            elif any(x in name_lower for x in ['ethylene_glycol', 'glycerol', 'propylene_glycol']):
                bounds.append((0.0, 2.5))
            elif any(x in name_lower for x in ['trehalose', 'sucrose', 'raffinose']):
                bounds.append((0.0, 1.0))
            elif any(x in name_lower for x in ['proline', 'betaine', 'ectoin', 'taurine', 'isoleucine']):
                bounds.append((0.0, 0.5))
            elif any(x in name_lower for x in ['fbs', 'human_serum']):
                bounds.append((0.0, 90.0))
            elif any(x in name_lower for x in ['hyaluronic_acid', 'methylcellulose']):
                bounds.append((0.0, 2.0))
            else:
                bounds.append((0.0, 10.0))
        return bounds
    
    def _sparsify(self, x: np.ndarray) -> np.ndarray:
        """
        Enforce max ingredients by zeroing smallest components.
        """
        x_sparse = x.copy()
        n_ing = count_nonzero(x_sparse)
        if n_ing > self.effective_max_ingredients:
            nonzero_idx = np.where(np.abs(x_sparse) > 1e-6)[0]
            sorted_idx = nonzero_idx[np.argsort(np.abs(x_sparse[nonzero_idx]))]
            for idx in sorted_idx[:n_ing - self.effective_max_ingredients]:
                x_sparse[idx] = 0.0
        return x_sparse

    def _complexity_penalty(self, x: np.ndarray) -> float:
        """Prefer simpler formulations when the acquisition surface is nearly flat."""
        n_ing = count_nonzero(x)
        excess = max(0, n_ing - self.reference_ingredient_count)
        return self.config.sparsity_penalty * excess

    def _support_penalty(self, x: np.ndarray) -> float:
        """Penalize candidates that move well outside the observed formulation manifold."""
        if self.observed_support_scaled is None or not np.isfinite(self.support_radius):
            return 0.0

        x_scaled = self.support_scaler.transform(x.reshape(1, -1))
        min_distance = float(
            np.min(np.linalg.norm(self.observed_support_scaled - x_scaled, axis=1))
        )
        if min_distance <= self.support_radius:
            return 0.0

        overshoot = min_distance - self.support_radius
        return self.config.support_penalty * overshoot * overshoot

    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """Clip a formulation to the configured feature bounds."""
        x_clipped = x.copy()
        for idx, (low, high) in enumerate(self.bounds):
            x_clipped[idx] = np.clip(x_clipped[idx], low, high)
        return x_clipped

    def _build_initial_population(self,
                                  seed_points: Optional[List[np.ndarray]],
                                  seed: int) -> np.ndarray:
        """Construct a DE initial population anchored on observed high performers."""
        population_size = max(5, self.config.de_popsize * len(self.bounds))
        rng = np.random.default_rng(seed)
        ranges = np.array([high - low for low, high in self.bounds], dtype=float)
        jitter_scale = np.maximum(ranges * 0.05, 1e-6)
        population: List[np.ndarray] = []

        for point in seed_points or []:
            base = self._sparsify(self._clip_to_bounds(point))
            population.append(base)
            if len(population) >= population_size:
                break

            for _ in range(3):
                perturbed = base + rng.normal(0.0, jitter_scale)
                perturbed = self._sparsify(self._clip_to_bounds(perturbed))
                population.append(perturbed)
                if len(population) >= population_size:
                    break
            if len(population) >= population_size:
                break

        while len(population) < population_size:
            random_point = np.array(
                [rng.uniform(low, high) for low, high in self.bounds], dtype=float
            )
            population.append(self._sparsify(random_point))

        return np.array(population[:population_size])
    
    def _local_penalizer(self, x: np.ndarray, found_candidates: List[np.ndarray]) -> float:
        """
        Compute local penalty to push DE away from previously found candidates.
        Uses Gaussian-shaped repulsion centered on each found candidate.
        
        Args:
            x: Current candidate formulation
            found_candidates: List of previously found formulation vectors
            
        Returns:
            Penalty value (higher = more repulsion)
        """
        if not found_candidates:
            return 0.0
        
        # Compute characteristic length scale from feature bounds
        ranges = np.array([b[1] - b[0] for b in self.bounds])
        ranges = np.maximum(ranges, 1e-6)  # Avoid division by zero
        length_scale = ranges * self.config.diversity_radius
        
        penalty = 0.0
        for prev in found_candidates:
            # Normalized squared distance
            diff = (x - prev) / length_scale
            dist_sq = np.sum(diff ** 2)
            # Gaussian repulsion
            penalty += self.config.diversity_penalty * np.exp(-0.5 * dist_sq)
        
        return penalty

    def _is_duplicate(self, x: np.ndarray, found_candidates: List[np.ndarray]) -> bool:
        """Return True when a candidate matches an existing formulation closely."""
        return any(np.allclose(x, prev, atol=1e-3, rtol=1e-3) for prev in found_candidates)

    def _evaluate_candidate(self, x: np.ndarray, y_best: float) -> Dict[str, float]:
        """Evaluate one formulation and package it for ranking/export."""
        x_eval = self._sparsify(self._clip_to_bounds(x))
        x_reshaped = x_eval.reshape(1, -1)
        if self.is_composite:
            pred_mean, pred_std = self.gp.predict(x_reshaped, return_std=True)
        else:
            x_scaled = self.scaler.transform(x_reshaped)
            pred_mean, pred_std = self.gp.predict(x_scaled, return_std=True)

        if self.config.acquisition.lower() == 'ei':
            pure_acq = expected_improvement(
                x_eval, self.gp, self.scaler, y_best, self.config.xi, self.is_composite
            )
        else:
            pure_acq = upper_confidence_bound(
                x_eval, self.gp, self.scaler, self.config.kappa, self.is_composite
            )

        dmso_molar = x_eval[self.dmso_index] if self.dmso_index >= 0 else 0
        dmso_percent = dmso_molar * 78.13 / (1.10 * 10)
        return {
            'formulation': x_eval.copy(),
            'acq_value': pure_acq,
            'predicted_viability': pred_mean[0],
            'uncertainty': pred_std[0],
            'dmso_percent': dmso_percent,
            'n_ingredients': count_nonzero(x_eval),
        }
    
    def _objective_with_penalty(self, x: np.ndarray, y_best: float,
                                found_candidates: List[np.ndarray] = None) -> float:
        """
        Objective function for DE: negative acquisition value + constraint penalties + diversity penalty.
        """
        # Sparsify first to enforce ingredient constraint
        x_sparse = self._sparsify(x)
        
        # Calculate acquisition value on sparsified formulation
        if self.config.acquisition.lower() == 'ei':
            acq_val = expected_improvement(x_sparse, self.gp, self.scaler, y_best, self.config.xi, self.is_composite)
        else:
            acq_val = upper_confidence_bound(x_sparse, self.gp, self.scaler, self.config.kappa, self.is_composite)
        
        # We negate the acquisition value because DE minimizes
        neg_acq = -acq_val
        
        # Soft penalty for DMSO (in case it slightly exceeds)
        penalty = 0.0
        if self.dmso_index >= 0 and x_sparse[self.dmso_index] > self.max_dmso_molar:
            penalty += (x_sparse[self.dmso_index] - self.max_dmso_molar) * 50.0
        penalty += self._complexity_penalty(x_sparse)
        penalty += self._support_penalty(x_sparse)
        
        # Batch diversity penalty: repel from previously found candidates
        if found_candidates:
            penalty += self._local_penalizer(x_sparse, found_candidates)
        
        return neg_acq + penalty
    
    def _run_de_single(self, y_best: float, seed: int,
                       found_candidates: List[np.ndarray] = None,
                       seed_points: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, float]:
        """
        Run a single DE optimization to find one candidate.
        
        Args:
            y_best: Best observed viability
            seed: Random seed for DE
            found_candidates: Previously found candidates for diversity penalty
            
        Returns:
            Tuple of (best formulation, acquisition value)
        """
        result = differential_evolution(
            func=lambda x: self._objective_with_penalty(x, y_best, found_candidates),
            bounds=self.bounds,
            maxiter=self.config.de_maxiter,
            popsize=self.config.de_popsize,
            seed=seed,
            init=self._build_initial_population(seed_points, seed),
            polish=True,  # Use L-BFGS-B to polish the result
            disp=False,
        )
        
        return result.x, -result.fun  # Return positive acquisition value
    
    def optimize(self, observed_df: pd.DataFrame,
                 n_candidates: int = None) -> pd.DataFrame:
        """
        Generate optimized candidates using DE-based acquisition maximization.
        
        Args:
            observed_df: Observed context dataframe
            n_candidates: Number of candidates to generate
            
        Returns:
            DataFrame with candidate formulations ranked by EI
        """
        if n_candidates is None:
            n_candidates = self.config.n_candidates

        X_observed = observed_df[self.feature_names].values
        y_observed = observed_df['viability_percent'].values
        self._fit_search_context(observed_df)
        
        if self.is_composite:
            # When using composite model, compute y_best from model predictions
            y_pred = self.gp.predict(X_observed)
            y_best = np.max(y_pred)
            print(f"Best model-predicted viability: {y_best:.1f}% (raw observed max: {np.max(y_observed):.1f}%)")
        else:
            X_scaled = self.scaler.transform(X_observed)
            y_pred = self.gp.predict(X_scaled)
            y_best = np.max(y_pred)
            print(f"Best observed viability: {y_best:.1f}%")
            
        print(f"Running batch-mode DE optimization for {n_candidates} candidates...")
        print(
            f"  Effective max ingredients: {self.effective_max_ingredients} "
            f"(observed median: {self.reference_ingredient_count})"
        )
        if np.isfinite(self.support_radius):
            print(f"  Support radius: {self.support_radius:.2f} scaled units")
        print(f"  Diversity penalty: {self.config.diversity_penalty}, radius: {self.config.diversity_radius}")

        seed_df = self.seed_context if self.seed_context is not None else observed_df
        X_seed = seed_df[self.feature_names].values
        if self.is_composite:
            observed_pred = self.gp.predict(X_seed)
        else:
            observed_pred = self.gp.predict(self.scaler.transform(X_seed))
        seed_weights = (
            seed_df['context_weight'].to_numpy(dtype=float)
            if 'context_weight' in seed_df.columns
            else np.ones(len(seed_df), dtype=float)
        )
        seed_order = np.lexsort((-seed_weights, -observed_pred))
        seed_formulations = [X_seed[idx].copy() for idx in seed_order[: min(12, len(seed_order))]]
        
        candidates = []
        found_formulations = []  # Track found candidates for diversity penalty

        for seed_x in seed_formulations:
            seed_candidate = self._evaluate_candidate(seed_x, y_best)
            if self.dmso_index >= 0 and seed_candidate['formulation'][self.dmso_index] > self.max_dmso_molar:
                continue
            if self._is_duplicate(seed_candidate['formulation'], found_formulations):
                continue
            candidates.append(seed_candidate)
            found_formulations.append(seed_candidate['formulation'].copy())
            if len(candidates) >= n_candidates:
                break
        
        attempt = 0
        max_attempts = max(n_candidates * 10, 20)
        while len(candidates) < n_candidates and attempt < max_attempts:
            seed = self.config.random_seed + attempt
            x_opt, _ = self._run_de_single(
                y_best, seed, found_formulations, seed_formulations
            )
            attempt += 1
            
            # Enforce constraints by clipping
            if self.dmso_index >= 0:
                x_opt[self.dmso_index] = min(x_opt[self.dmso_index], self.max_dmso_molar)
            
            # Sparsify: zero out smallest components if too many ingredients
            n_ing = count_nonzero(x_opt)
            if n_ing > self.effective_max_ingredients:
                # Zero out smallest components
                nonzero_idx = np.where(np.abs(x_opt) > 1e-6)[0]
                sorted_idx = nonzero_idx[np.argsort(np.abs(x_opt[nonzero_idx]))]
                for idx in sorted_idx[:n_ing - self.effective_max_ingredients]:
                    x_opt[idx] = 0.0

            if self._is_duplicate(x_opt, found_formulations):
                continue
            
            candidates.append(self._evaluate_candidate(x_opt, y_best))
            
            # Track this candidate for diversity penalty in subsequent DE runs
            found_formulations.append(x_opt.copy())
            
            if len(candidates) % 5 == 0:
                print(f"  Generated {len(candidates)}/{n_candidates} candidates...")

        if len(candidates) < n_candidates:
            print(
                f"Warning: generated {len(candidates)} unique candidates after {attempt} attempts"
            )
        
        # Sort by predicted viability (primary ranking for diverse batch candidates)
        candidates.sort(key=lambda c: c['predicted_viability'], reverse=True)
        
        # Build output DataFrame
        output_data = []
        for rank, c in enumerate(candidates, 1):
            row = {
                'rank': rank,
                'acquisition_value': c['acq_value'],
                'predicted_viability': c['predicted_viability'],
                'uncertainty': c['uncertainty'],
                'dmso_percent': c['dmso_percent'],
                'n_ingredients': c['n_ingredients'],
            }
            
            # Add ingredient concentrations
            x = c['formulation']
            for j, name in enumerate(self.feature_names):
                if x[j] > 1e-6:
                    row[name] = x[j]
            
            output_data.append(row)
        
        return pd.DataFrame(output_data)
    
    def generate_dmso_free_candidates(self, observed_df: pd.DataFrame,
                                       n_candidates: int = 20) -> pd.DataFrame:
        """Generate low-DMSO candidates (<0.5% v/v)."""
        # Temporarily set DMSO bound to near-zero
        original_max = self.max_dmso_molar
        self.max_dmso_molar = 0.07  # ~0.5% DMSO
        
        if self.dmso_index >= 0:
            original_bound = self.bounds[self.dmso_index]
            self.bounds[self.dmso_index] = (0.0, 0.07)
        
        try:
            candidates = self.optimize(observed_df, n_candidates)
        finally:
            self.max_dmso_molar = original_max
            if self.dmso_index >= 0:
                self.bounds[self.dmso_index] = original_bound
        
        return candidates


# =============================================================================
# RESULTS EXPORT
# =============================================================================

def export_candidates(candidates_df: pd.DataFrame, feature_names: List[str],
                      output_path: str):
    """Export candidate formulations to CSV and summary."""
    candidates_df.to_csv(output_path, index=False)
    
    summary_path = output_path.replace('.csv', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CryoMN Bayesian Optimization Candidates (DE-based)\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 80 + "\n\n")
        
        for _, row in candidates_df.iterrows():
            f.write(f"Rank {int(row['rank'])}: {format_formulation(row, feature_names)}\n")
            f.write(f"  Acquisition Value: {row['acquisition_value']:.4f}\n")
            f.write(f"  Predicted viability: {row['predicted_viability']:.1f}% ± {row['uncertainty']:.1f}%\n")
            f.write(f"  DMSO: {row['dmso_percent']:.1f}%\n")
            f.write(f"  Ingredients: {int(row['n_ingredients'])}\n\n")
    
    print(f"Candidates saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")


def build_iteration_output_path(output_dir: str, base_filename: str,
                                iteration_dir: Optional[str],
                                iteration: Optional[int]) -> str:
    """Append the active iteration identity to exported result filenames."""
    stem, ext = os.path.splitext(base_filename)
    if iteration_dir:
        suffix = iteration_dir
    elif iteration is not None:
        suffix = f"iteration_{iteration}"
    else:
        suffix = "active_model"
    return os.path.join(output_dir, f"{stem}_{suffix}{ext}")


def load_observed_formulations(project_root: str, resolution) -> pd.DataFrame:
    """Load the active iteration's observed context for BO."""
    return load_observed_context(
        project_root=project_root,
        feature_names=resolution.metadata['feature_names'],
        model_method=resolution.model_method,
        iteration=resolution.iteration,
        iteration_dir=resolution.iteration_dir,
        metadata=resolution.metadata,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for DE-based Bayesian optimization."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    model_dir = os.path.join(project_root, 'models')
    output_dir = os.path.join(project_root, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("CryoMN Bayesian Optimization with Differential Evolution")
    print("=" * 80)
    
    print("\nLoading trained model...")
    try:
        resolution = resolve_active_model(project_root)
    except ModelResolutionError as exc:
        print(f"ERROR: {exc}")
        return
    gp = resolution.gp
    scaler = resolution.scaler
    metadata = resolution.metadata
    is_composite = resolution.is_composite
    feature_names = metadata['feature_names']
    print(f"Model loaded with {len(feature_names)} features")
    if resolution.iteration_dir:
        print(f"Resolved active iteration: {resolution.iteration_dir}")
    elif resolution.iteration is not None:
        print(f"Resolved active iteration: iteration_{resolution.iteration}")
    
    # Load literature + wet-lab observations for BO search context
    print("\nLoading observed formulations for BO context...")
    observed_df = load_observed_formulations(project_root, resolution)
    print(f"Loaded {len(observed_df)} total observed rows")
    if 'source' in observed_df.columns:
        n_lit = int((observed_df['source'] == 'literature').sum())
        n_wet = int((observed_df['source'] == 'wetlab').sum())
        wet_weight = (
            observed_df.loc[observed_df['source'] == 'wetlab', 'context_weight'].iloc[0]
            if n_wet > 0 else 'N/A'
        )
        print(f"Observed sources: {n_lit} literature + {n_wet} wet lab (weight={wet_weight})")
    
    # Initialize optimizer
    config = BOConfig(
        max_dmso_percent=5.0,
        n_candidates=20,
    )
    
    optimizer = BayesianOptimizer(gp, scaler, feature_names, config, is_composite=is_composite)
    
    # Generate candidates
    print("\n" + "-" * 40)
    print("Generating Candidates via DE")
    print("-" * 40)
    
    print("\n1. General optimization (≤5% DMSO)...")
    general_candidates = optimizer.optimize(observed_df, n_candidates=20)
    
    print("\n2. Low-DMSO optimization (<0.5% DMSO)...")
    dmso_free_candidates = optimizer.generate_dmso_free_candidates(observed_df, n_candidates=20)
    
    # Export results
    print("\n" + "-" * 40)
    print("Exporting Results")
    print("-" * 40)
    
    export_candidates(
        general_candidates,
        feature_names,
        build_iteration_output_path(
            output_dir,
            'bo_candidates_general.csv',
            resolution.iteration_dir,
            resolution.iteration,
        )
    )
    
    export_candidates(
        dmso_free_candidates,
        feature_names,
        build_iteration_output_path(
            output_dir,
            'bo_candidates_dmso_free.csv',
            resolution.iteration_dir,
            resolution.iteration,
        )
    )
    
    # Print top candidates
    print("\n" + "=" * 80)
    print("Top 20 General Candidates (by Predicted Viability)")
    print("=" * 80)
    for _, row in general_candidates.head(20).iterrows():
        print(f"\nRank {int(row['rank'])}: {config.acquisition.upper()} = {row['acquisition_value']:.4f}")
        print(f"  Predicted: {row['predicted_viability']:.1f}% ± {row['uncertainty']:.1f}%")
        print(f"  DMSO: {row['dmso_percent']:.1f}%, Ingredients: {int(row['n_ingredients'])}")
    
    print("\n" + "=" * 80)
    print("Top 20 Low-DMSO Candidates (<0.5% DMSO)")
    print("=" * 80)
    for _, row in dmso_free_candidates.head(20).iterrows():
        print(f"\nRank {int(row['rank'])}: {config.acquisition.upper()} = {row['acquisition_value']:.4f}")
        print(f"  Predicted: {row['predicted_viability']:.1f}% ± {row['uncertainty']:.1f}%")
        print(f"  DMSO: {row['dmso_percent']:.1f}%, Ingredients: {int(row['n_ingredients'])}")
    
    print("\n" + "=" * 80)
    print("Optimization Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
