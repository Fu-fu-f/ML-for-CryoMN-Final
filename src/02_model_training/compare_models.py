import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime

# ML Imports
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def load_data(data_path):
    df = pd.read_csv(data_path)
    # Filter to only include rows with at least 3 non-zero values for an ingredient
    # to avoid noise, same as 'his' logic
    ingredient_cols = [c for c in df.columns if c.endswith('_M') or c.endswith('_pct')]
    active_ingredients = [col for col in ingredient_cols if (df[col] > 0).sum() >= 3]
    
    X = df[active_ingredients].values
    y = df['viability_percent'].values
    
    return X, y, active_ingredients

def compare_models_repeated(X, y, n_repeats=10):
    # Scale features (GPR and others work better)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5-fold, 10 repeats for total 50 evaluations
    rkf = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=42)
    
    print(f"\nRunning {5 * n_repeats} evaluations (5-Fold × {n_repeats} Repeats)...")
    
    # 1. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_r2 = cross_val_score(rf, X_scaled, y, cv=rkf, scoring='r2')
    rf_rmse = -cross_val_score(rf, X_scaled, y, cv=rkf, scoring='neg_root_mean_squared_error')
    
    # 2. XGBoost
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    xgb_r2 = cross_val_score(xgb, X_scaled, y, cv=rkf, scoring='r2')
    xgb_rmse = -cross_val_score(xgb, X_scaled, y, cv=rkf, scoring='neg_root_mean_squared_error')
    
    # 3. Gaussian Process
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(noise_level=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True, random_state=42)
    gp_r2 = cross_val_score(gp, X_scaled, y, cv=rkf, scoring='r2')
    gp_rmse = -cross_val_score(gp, X_scaled, y, cv=rkf, scoring='neg_root_mean_squared_error')
    
    print("\n" + "="*60)
    print(f"Robust Model Comparison (Repeated 5-Fold, {5*n_repeats} Iterations Scale)")
    print("="*60)
    print(f"Random Forest    : Mean R² = {rf_r2.mean():.4f} | Mean RMSE = {rf_rmse.mean():.4f}%")
    print(f"XGBoost          : Mean R² = {xgb_r2.mean():.4f} | Mean RMSE = {xgb_rmse.mean():.4f}%")
    print(f"Gaussian Process : Mean R² = {gp_r2.mean():.4f} | Mean RMSE = {gp_rmse.mean():.4f}%")
    print("="*60)
    
    # Check Stability
    min_score = gp_r2.min()
    max_score = gp_r2.max()
    print(f"Gaussian Process Score Stability: Min R²: {min_score:.4f}, Max R²: {max_score:.4f}")
    
    # Finalize on GP
    print("\n[Final Decision] Selecting Gaussian Process for its consistent performance.")
    gp.fit(X_scaled, y)
    return gp, scaler

def save_model(model, scaler, feature_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'gp_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    metadata = {
        'feature_names': feature_names,
        'trained_at': datetime.now().isoformat(),
        'model_type': 'GaussianProcessRegressor'
    }
    with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nFinalized GP model saved to: {output_dir}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    data_path = os.path.join(project_root, 'data', 'processed', 'parsed_formulations.csv')
    model_dir = os.path.join(project_root, 'models')
    
    if not os.path.exists(data_path):
        print(f"Error: Processed data not found at {data_path}")
        return

    print(f"Loading data from {data_path}...")
    X, y, features = load_data(data_path)
    
    model, scaler = compare_models_repeated(X, y, n_repeats=10)
    save_model(model, scaler, features, model_dir)

if __name__ == "__main__":
    main()
