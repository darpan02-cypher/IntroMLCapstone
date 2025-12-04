"""
Test script to verify classical ML models work correctly.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append('src')

from models import (
    LinearRegressionModel, 
    RandomForestModel, 
    XGBoostModel,
    compare_models
)

print("="*70)
print("TESTING CLASSICAL ML MODELS")
print("="*70)

# Load data
print("\n1. Loading preprocessed data...")
data_dir = Path('data/processed')
X_train = pd.read_csv(data_dir / 'S2_X_train.csv')
y_train = pd.read_csv(data_dir / 'S2_y_train.csv').values.ravel()
X_test = pd.read_csv(data_dir / 'S2_X_test.csv')
y_test = pd.read_csv(data_dir / 'S2_y_test.csv').values.ravel()

feature_names = X_train.columns.tolist()
print(f"   ✓ Train: {X_train.shape}, Test: {X_test.shape}")

# Test Ridge Regression
print("\n2. Testing Ridge Regression...")
ridge_model = LinearRegressionModel(model_type='ridge', random_state=42)
ridge_model.train(
    X_train.values, 
    y_train, 
    feature_names=feature_names,
    tune_hyperparams=True,
    cv_folds=3,  # Reduced for speed
    verbose=0
)
ridge_metrics = ridge_model.evaluate(X_test.values, y_test, dataset_name="Test")

# Test Random Forest
print("\n3. Testing Random Forest...")
rf_model = RandomForestModel(random_state=42)
# Use smaller param grid for testing
rf_model.get_param_grid = lambda: {
    'n_estimators': [50, 100],
    'max_depth': [10, None],
    'min_samples_split': [2, 5]
}
rf_model.train(
    X_train.values, 
    y_train, 
    feature_names=feature_names,
    tune_hyperparams=True,
    cv_folds=3,
    verbose=0
)
rf_metrics = rf_model.evaluate(X_test.values, y_test, dataset_name="Test")

# Test XGBoost
print("\n4. Testing XGBoost...")
xgb_model = XGBoostModel(random_state=42)
# Use smaller param grid for testing
xgb_model.get_param_grid = lambda: {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.3]
}
xgb_model.train(
    X_train.values, 
    y_train, 
    feature_names=feature_names,
    tune_hyperparams=True,
    cv_folds=3,
    verbose=0
)
xgb_metrics = xgb_model.evaluate(X_test.values, y_test, dataset_name="Test")

# Compare models
print("\n5. Comparing all models...")
all_models = [ridge_model, rf_model, xgb_model]
comparison = compare_models(all_models, X_test.values, y_test)

print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
print(comparison.to_string(index=False))
print("="*70)

# Test feature importance
print("\n6. Testing feature importance extraction...")
ridge_importance = ridge_model.get_feature_importance()
rf_importance = rf_model.get_feature_importance()
xgb_importance = xgb_model.get_feature_importance()

print(f"   ✓ Ridge top feature: {ridge_importance.iloc[0]['feature']}")
print(f"   ✓ RF top feature: {rf_importance.iloc[0]['feature']}")
print(f"   ✓ XGBoost top feature: {xgb_importance.iloc[0]['feature']}")

print("\n" + "="*70)
print("✓ ALL TESTS PASSED SUCCESSFULLY!")
print("="*70)
