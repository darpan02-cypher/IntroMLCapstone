"""
Train classical ML models on multi-subject dataset.
Quick evaluation for tomorrow's report.
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
print("MULTI-SUBJECT MODEL TRAINING")
print("="*70)

# Load multi-subject data
print("\n1. Loading multi-subject data...")
data_dir = Path('data/processed')
X_train = pd.read_csv(data_dir / 'multi_X_train.csv')
y_train = pd.read_csv(data_dir / 'multi_y_train.csv').values.ravel()
X_test = pd.read_csv(data_dir / 'multi_X_test.csv')
y_test = pd.read_csv(data_dir / 'multi_y_test.csv').values.ravel()

feature_names = X_train.columns.tolist()
print(f"   ✓ Train: {X_train.shape}, Test: {X_test.shape}")
print(f"   ✓ Features: {len(feature_names)}")

# Train Ridge Regression
print("\n2. Training Ridge Regression...")
ridge_model = LinearRegressionModel(model_type='ridge', random_state=42)
ridge_model.train(
    X_train.values, 
    y_train, 
    feature_names=feature_names,
    tune_hyperparams=True,
    cv_folds=5,
    verbose=1
)
ridge_metrics = ridge_model.evaluate(X_test.values, y_test, dataset_name="Test")

# Train Random Forest
print("\n3. Training Random Forest...")
rf_model = RandomForestModel(random_state=42)
rf_model.train(
    X_train.values, 
    y_train, 
    feature_names=feature_names,
    tune_hyperparams=True,
    cv_folds=5,
    verbose=1
)
rf_metrics = rf_model.evaluate(X_test.values, y_test, dataset_name="Test")

# Train XGBoost
print("\n4. Training XGBoost...")
xgb_model = XGBoostModel(random_state=42)
xgb_model.train(
    X_train.values, 
    y_train, 
    feature_names=feature_names,
    tune_hyperparams=True,
    cv_folds=5,
    verbose=1
)
xgb_metrics = xgb_model.evaluate(X_test.values, y_test, dataset_name="Test")

# Compare models
print("\n5. Comparing all models...")
all_models = [ridge_model, rf_model, xgb_model]
comparison = compare_models(all_models, X_test.values, y_test)

print("\n" + "="*80)
print("MULTI-SUBJECT MODEL COMPARISON (4 Subjects: S2, S3, S4, S5)")
print("="*80)
print(comparison.to_string(index=False))
print("="*80)

# Save results
comparison.to_csv(data_dir / 'multi_subject_model_comparison.csv', index=False)
print(f"\n✓ Results saved to {data_dir / 'multi_subject_model_comparison.csv'}")

# Save best model
best_model_idx = comparison['R²'].idxmax()
best_model = all_models[best_model_idx]
best_model.save(data_dir / 'best_multi_subject_model.pkl')

print(f"\n🏆 Best Model: {best_model.name}")
print(f"   R² Score: {comparison.iloc[best_model_idx]['R²']:.4f}")
print(f"   Best Parameters: {best_model.best_params_}")

# Feature importance
print(f"\n6. Top 10 Most Important Features ({best_model.name}):")
importance = best_model.get_feature_importance()
print(importance.head(10).to_string(index=False))

print("\n" + "="*70)
print("✓ MULTI-SUBJECT MODEL TRAINING COMPLETE")
print("="*70)
