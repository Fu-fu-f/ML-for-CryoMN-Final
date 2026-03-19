#!/usr/bin/env python3
"""
CryoMN Candidate Generation via Random Sampling

Generates cryoprotective formulation candidates by random sampling,
then ranks them by predicted viability under the active model.

Author: CryoMN ML Project
Date: 2026-01-24
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import Tuple, List, Optional
from datetime import datetime
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler

# Add shared helper modules to path for model resolution and observed-context loading
_script_dir = os.path.dirname(os.path.abspath(__file__))
_helper_dir = os.path.join(os.path.dirname(_script_dir), 'helper')
if _helper_dir not in sys.path:
    sys.path.insert(0, _helper_dir)
from active_model_resolver import ModelResolutionError, resolve_active_model  # noqa: E402
from formulation_formatting import (  # noqa: E402
    format_formulation,
    normalize_formulation_vector,
)
from observed_context import load_observed_context  # noqa: E402

# =============================================================================
# OPTIMIZATION CONFIGURATION
# =============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for random candidate generation."""
    max_ingredients: int = 10  # Maximum non-zero ingredients per formulation
    max_dmso_percent: float = 5.0  # Maximum DMSO percentage
    min_viability: float = 70.0  # Minimum target viability
    n_candidates: int = 20  # Number of candidate formulations to generate
    random_seed: int = 42


# =============================================================================
# CONSTRAINT HANDLING
# =============================================================================

def count_ingredients(x: np.ndarray, threshold: float = 1e-6) -> int:
    """Count non-zero ingredients in formulation."""
    return np.sum(np.abs(x) > threshold)


# =============================================================================
# OPTIMIZATION CORE
# =============================================================================

class FormulationOptimizer:
    """
    Random-sampling candidate generator for cryoprotective formulations.

    The optimizer samples formulations within practical bounds, filters
    by constraints, and ranks the surviving candidates by predicted viability.
    """
    
    def __init__(self, gp, scaler: StandardScaler,
                 feature_names: List[str], config: OptimizationConfig = None,
                 is_composite: bool = False):
        """
        Initialize optimizer.
        
        Args:
            gp: Trained Gaussian Process model (or CompositeGP)
            scaler: Feature scaler (unused if is_composite)
            feature_names: List of feature names
            config: Candidate generation configuration
            is_composite: If True, model handles scaling internally
        """
        self.gp = gp
        self.scaler = scaler
        self.feature_names = feature_names
        self.config = config or OptimizationConfig()
        self.is_composite = is_composite
        
        # Find DMSO index
        self.dmso_index = -1
        for i, name in enumerate(feature_names):
            if 'dmso' in name.lower():
                self.dmso_index = i
                break
        
        # Calculate max DMSO in molar (5% v/v ≈ 0.70 M)
        self.max_dmso_molar = (self.config.max_dmso_percent / 100.0) * 1.10 * 1000 / 78.13
        
        # Set feature bounds based on training data
        self.bounds = self._get_feature_bounds()
        
        np.random.seed(self.config.random_seed)

    def _apply_practical_floor(self, x: np.ndarray) -> np.ndarray:
        """Zero trace ingredients that should count as operationally absent."""
        return normalize_formulation_vector(x, self.feature_names)
    
    def _get_feature_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for each feature based on typical concentration ranges."""
        bounds = []
        for name in self.feature_names:
            name_lower = name.lower()
            if 'dmso' in name_lower:
                # DMSO: 0 to max allowed
                bounds.append((0.0, self.max_dmso_molar))
            elif any(x in name_lower for x in ['ethylene_glycol', 'glycerol', 'propylene_glycol']):
                # Permeating CPAs: 0 to 2.5 M
                bounds.append((0.0, 2.5))
            elif any(x in name_lower for x in ['trehalose', 'sucrose', 'raffinose']):
                # Sugars: 0 to 1 M
                bounds.append((0.0, 1.0))
            elif any(x in name_lower for x in ['proline', 'betaine', 'ectoin', 'taurine', 'isoleucine']):
                # Amino acids: 0 to 0.5 M
                bounds.append((0.0, 0.5))
            elif any(x in name_lower for x in ['fbs', 'human_serum']):
                # Sera: 0 to 90% (normalized value)
                bounds.append((0.0, 90.0))
            else:
                # Other: 0 to 10 (generic bound)
                bounds.append((0.0, 10.0))
        
        return bounds
    
    def _generate_random_candidate(self) -> np.ndarray:
        """Generate a random candidate formulation."""
        x = np.zeros(len(self.feature_names))
        
        # Select random subset of ingredients
        n_ingredients = np.random.randint(2, self.config.max_ingredients + 1)
        selected_indices = np.random.choice(
            len(self.feature_names), 
            size=n_ingredients, 
            replace=False
        )
        
        # Assign random concentrations
        for idx in selected_indices:
            low, high = self.bounds[idx]
            x[idx] = np.random.uniform(low, high)

        return self._apply_practical_floor(x)
    
    def optimize(self, X_observed: np.ndarray, y_observed: np.ndarray,
                 n_candidates: int = None) -> pd.DataFrame:
        """
        Generate optimized candidate formulations using random sampling + GP prediction.
        
        Args:
            X_observed: Observed formulation features
            y_observed: Observed viability values
            n_candidates: Number of candidates to generate
            
        Returns:
            DataFrame with candidate formulations
        """
        if n_candidates is None:
            n_candidates = self.config.n_candidates
        
        if self.is_composite:
            best_predicted = np.max(self.gp.predict(X_observed))
            print(
                f"Best model-predicted viability: {best_predicted:.1f}% "
                f"(raw observed max: {np.max(y_observed):.1f}%)"
            )
        else:
            print(f"Best observed viability: {np.max(y_observed):.1f}%")
        
        # Generate many random candidates and select the best
        n_samples = n_candidates * 50  # Over-sample
        candidates = []
        
        print(f"Generating {n_samples} random formulations...")
        
        for i in range(n_samples):
            x = self._apply_practical_floor(self._generate_random_candidate())
            
            # Skip if too many ingredients
            n_ing = count_ingredients(x)
            if n_ing > self.config.max_ingredients or n_ing < 1:
                continue
            
            # Check DMSO constraint
            if self.dmso_index >= 0:
                dmso_molar = x[self.dmso_index]
                if dmso_molar > self.max_dmso_molar:
                    continue
            
            # Predict viability
            x_reshaped = x.reshape(1, -1)
            if self.is_composite:
                pred_mean, pred_std = self.gp.predict(x_reshaped, return_std=True)
            else:
                x_scaled = self.scaler.transform(x_reshaped)
                pred_mean, pred_std = self.gp.predict(x_scaled, return_std=True)
            
            # Calculate DMSO percentage
            dmso_molar = x[self.dmso_index] if self.dmso_index >= 0 else 0
            dmso_percent = dmso_molar * 78.13 / (1.10 * 10)
            
            candidate = {
                'predicted_viability': pred_mean[0],
                'uncertainty': pred_std[0],
                'dmso_percent': dmso_percent,
                'n_ingredients': n_ing,
                'formulation': x.copy(),
            }
            
            candidates.append(candidate)
        
        print(f"Generated {len(candidates)} valid candidates")
        
        if len(candidates) == 0:
            # Fallback: generate at least some candidates without constraints
            print("Warning: No valid candidates, relaxing constraints...")
            for i in range(n_candidates):
                x = self._apply_practical_floor(self._generate_random_candidate())
                x_reshaped = x.reshape(1, -1)
                if self.is_composite:
                    pred_mean, pred_std = self.gp.predict(x_reshaped, return_std=True)
                else:
                    x_scaled = self.scaler.transform(x_reshaped)
                    pred_mean, pred_std = self.gp.predict(x_scaled, return_std=True)
                dmso_molar = x[self.dmso_index] if self.dmso_index >= 0 else 0
                dmso_percent = dmso_molar * 78.13 / (1.10 * 10)
                
                candidates.append({
                    'predicted_viability': pred_mean[0],
                    'uncertainty': pred_std[0],
                    'dmso_percent': dmso_percent,
                    'n_ingredients': count_ingredients(x),
                    'formulation': x.copy(),
                })
        
        # Sort by predicted viability and select top candidates
        candidates.sort(key=lambda c: c['predicted_viability'], reverse=True)
        top_candidates = candidates[:n_candidates]
        
        # Build output DataFrame
        output_data = []
        for rank, c in enumerate(top_candidates, 1):
            row = {
                'rank': rank,
                'predicted_viability': c['predicted_viability'],
                'uncertainty': c['uncertainty'],
                'dmso_percent': c['dmso_percent'],
                'n_ingredients': c['n_ingredients'],
            }
            
            # Add ingredient concentrations
            x = self._apply_practical_floor(c['formulation'])
            for j, name in enumerate(self.feature_names):
                if x[j] > 1e-6:
                    row[name] = x[j]
            
            output_data.append(row)
        
        candidates_df = pd.DataFrame(output_data)
        return candidates_df
    
    def generate_low_dmso_candidates(self, X_observed: np.ndarray, 
                                      y_observed: np.ndarray,
                                      n_candidates: int = 20) -> pd.DataFrame:
        """
        Generate candidates with very low DMSO (<0.5% v/v).
        
        Args:
            X_observed: Observed formulations
            y_observed: Observed viabilities
            n_candidates: Number of candidates
            
        Returns:
            DataFrame with low-DMSO candidates
        """
        # Temporarily set max DMSO to very low
        original_max = self.max_dmso_molar
        self.max_dmso_molar = 0.07  # ~0.5% DMSO
        
        # Force DMSO bound to near-zero
        if self.dmso_index >= 0:
            original_bound = self.bounds[self.dmso_index]
            self.bounds[self.dmso_index] = (0.0, 0.07)
        
        try:
            candidates = self.optimize(X_observed, y_observed, n_candidates)
        finally:
            # Restore original settings
            self.max_dmso_molar = original_max
            if self.dmso_index >= 0:
                self.bounds[self.dmso_index] = original_bound
        
        return candidates


# =============================================================================
# RESULTS EXPORT
# =============================================================================

def export_candidates(candidates_df: pd.DataFrame, feature_names: List[str],
                      output_path: str):
    """Export candidate formulations to CSV and human-readable format."""
    # Save full CSV
    candidates_df.to_csv(output_path, index=False)
    
    # Create human-readable summary
    summary_path = output_path.replace('.csv', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CryoMN Random-Sampling Formulation Candidates\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 80 + "\n\n")
        
        for _, row in candidates_df.iterrows():
            f.write(f"Rank {int(row['rank'])}: {format_formulation(row, feature_names)}\n")
            f.write(f"  Predicted viability: {row['predicted_viability']:.1f}% ± {row['uncertainty']:.1f}%\n")
            f.write(f"  DMSO: {row['dmso_percent']:.1f}%\n")
            f.write(f"  Ingredients: {int(row['n_ingredients'])}\n")
            f.write("\n")
    
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


def load_observed_data(project_root: str, resolution) -> pd.DataFrame:
    """Load the active iteration's observed context for optimization."""
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
    """Main entry point for optimization."""
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    model_dir = os.path.join(project_root, 'models')
    output_dir = os.path.join(project_root, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("CryoMN Random Candidate Generation")
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
    
    print("\nLoading observed context...")
    df = load_observed_data(project_root, resolution)
    X = df[feature_names].values
    y = df['viability_percent'].values
    print(f"Loaded {len(df)} observed rows")
    if 'source' in df.columns:
        n_lit = int((df['source'] == 'literature').sum())
        n_wet = int((df['source'] == 'wetlab').sum())
        print(f"Observed sources: {n_lit} literature + {n_wet} wet lab")
    
    # Initialize optimizer
    config = OptimizationConfig(
        max_ingredients=10,
        max_dmso_percent=5.0,
        min_viability=70.0,
        n_candidates=20,
    )
    
    optimizer = FormulationOptimizer(gp, scaler, feature_names, config, is_composite=is_composite)
    
    # Generate candidates
    print("\n" + "-" * 40)
    print("Generating Ranked Candidates")
    print("-" * 40)
    
    print("\n1. General candidate generation (up to 5% DMSO allowed)...")
    general_candidates = optimizer.optimize(X, y, n_candidates=20)
    
    print("\n2. Low-DMSO candidate generation (<0.5% DMSO)...")
    dmso_free_candidates = optimizer.generate_low_dmso_candidates(X, y, n_candidates=20)
    
    # Export results
    print("\n" + "-" * 40)
    print("Exporting Results")
    print("-" * 40)
    
    export_candidates(
        general_candidates, 
        feature_names,
        build_iteration_output_path(
            output_dir,
            'candidates_general.csv',
            resolution.iteration_dir,
            resolution.iteration,
        )
    )
    
    export_candidates(
        dmso_free_candidates,
        feature_names,
        build_iteration_output_path(
            output_dir,
            'candidates_dmso_free.csv',
            resolution.iteration_dir,
            resolution.iteration,
        )
    )
    
    # Print top candidates
    print("\n" + "=" * 80)
    print("Top 20 General Candidates")
    print("=" * 80)
    for _, row in general_candidates.head(20).iterrows():
        print(f"\nRank {int(row['rank'])}: Viability = {row['predicted_viability']:.1f}% ± {row['uncertainty']:.1f}%")
        print(f"  DMSO: {row['dmso_percent']:.1f}%, Ingredients: {int(row['n_ingredients'])}")
    
    print("\n" + "=" * 80)
    print("Top 20 Low-DMSO Candidates (<0.5% DMSO)")
    print("=" * 80)
    for _, row in dmso_free_candidates.head(20).iterrows():
        print(f"\nRank {int(row['rank'])}: Viability = {row['predicted_viability']:.1f}% ± {row['uncertainty']:.1f}%")
        print(f"  DMSO: {row['dmso_percent']:.1f}%, Ingredients: {int(row['n_ingredients'])}")
    
    print("\n" + "=" * 80)
    print("Optimization Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
