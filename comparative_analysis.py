"""
Phase 4: Comparative Analysis of All 5 Models
Compare Ridge, Random Forest, XGBoost, Ensemble, and LSTM
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.append('src')

print("="*80)
print("PHASE 4: COMPARATIVE ANALYSIS")
print("="*80)

# Load data
data_dir = Path('data/processed')
X_test = pd.read_csv(data_dir / 'multi_X_test.csv')
y_test = pd.read_csv(data_dir / 'multi_y_test.csv').values.ravel()

# Load all model results
print("\n1. Loading all model results...")

# Classical models
classical_results = pd.read_csv(data_dir / 'multi_subject_model_comparison.csv')

# Paper A: Ensemble
with open(data_dir / 'paper_a_ensemble.pkl', 'rb') as f:
    ensemble_results = pickle.load(f)

# Paper B: LSTM
with open(data_dir / 'paper_b_lstm.pkl', 'rb') as f:
    lstm_results = pickle.load(f)

# ============================================================================
# CREATE COMPREHENSIVE COMPARISON TABLE
# ============================================================================
print("\n2. Creating comprehensive comparison...")

all_models_comparison = []

# Add classical models
for _, row in classical_results.iterrows():
    all_models_comparison.append({
        'Model': row['Model'],
        'Type': 'Classical ML',
        'R²': row['R²'],
        'RMSE': row['RMSE'],
        'MAE': row['MAE'],
        'MSE': row['MSE']
    })

# Add ensemble
all_models_comparison.append({
    'Model': 'Ensemble (Paper A)',
    'Type': 'Literature',
    'R²': ensemble_results['metrics']['R2'],
    'RMSE': ensemble_results['metrics']['RMSE'],
    'MAE': ensemble_results['metrics']['MAE'],
    'MSE': ensemble_results['metrics']['MSE']
})

# Add LSTM
all_models_comparison.append({
    'Model': lstm_results['model_name'],
    'Type': 'Literature',
    'R²': lstm_results['metrics']['R2'],
    'RMSE': lstm_results['metrics']['RMSE'],
    'MAE': lstm_results['metrics']['MAE'],
    'MSE': lstm_results['metrics']['MSE']
})

comparison_df = pd.DataFrame(all_models_comparison)
comparison_df = comparison_df.sort_values('R²', ascending=False)

print("\n" + "="*80)
print("ALL 5 MODELS COMPARISON (7 Subjects, 69 Test Samples)")
print("="*80)
print(comparison_df.to_string(index=False))
print("="*80)

# Save comparison
comparison_df.to_csv(data_dir / 'final_all_models_comparison.csv', index=False)
print(f"\n✓ Saved to {data_dir / 'final_all_models_comparison.csv'}")

# ============================================================================
# STATISTICAL SIGNIFICANCE TESTING
# ============================================================================
print("\n3. Statistical significance testing...")

# Get predictions for all models
predictions = {
    'XGBoost': None,
    'Random Forest': None,
    'Ridge': None,
    'Ensemble': ensemble_results['predictions'],
    'LSTM': lstm_results['predictions']
}

# Load classical model predictions (need to regenerate)
from models import LinearRegressionModel, RandomForestModel, XGBoostModel

ridge_model = LinearRegressionModel(model_type='ridge', random_state=42)
rf_model = RandomForestModel(random_state=42)
xgb_model = XGBoostModel(random_state=42)

X_train = pd.read_csv(data_dir / 'multi_X_train.csv')
y_train = pd.read_csv(data_dir / 'multi_y_train.csv').values.ravel()

ridge_model.train(X_train.values, y_train, tune_hyperparams=False, verbose=0)
rf_model.train(X_train.values, y_train, tune_hyperparams=False, verbose=0)
xgb_model.train(X_train.values, y_train, tune_hyperparams=False, verbose=0)

predictions['Ridge'] = ridge_model.predict(X_test.values)
predictions['Random Forest'] = rf_model.predict(X_test.values)
predictions['XGBoost'] = xgb_model.predict(X_test.values)

# Wilcoxon signed-rank test (pairwise comparisons)
print("\nWilcoxon Signed-Rank Test (p-values):")
print("Comparing best model (Ensemble) vs others:")

best_pred = predictions['Ensemble']
for model_name, pred in predictions.items():
    if model_name != 'Ensemble':
        # Compute residuals
        residuals_best = np.abs(y_test - best_pred)
        residuals_other = np.abs(y_test - pred)
        
        # Wilcoxon test
        statistic, p_value = stats.wilcoxon(residuals_best, residuals_other)
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"  Ensemble vs {model_name:20s}: p={p_value:.4f} {significance}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n4. Creating visualizations...")

# Create figure directory
viz_dir = Path('visualizations')
viz_dir.mkdir(exist_ok=True)

# Plot 1: Model Comparison Bar Chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# R² comparison
ax = axes[0, 0]
sorted_df = comparison_df.sort_values('R²', ascending=True)
colors = ['#2ecc71' if t == 'Literature' else '#3498db' for t in sorted_df['Type']]
ax.barh(sorted_df['Model'], sorted_df['R²'], color=colors)
ax.set_xlabel('R² Score', fontsize=12)
ax.set_title('Model Comparison: R² Score', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# MAE comparison
ax = axes[0, 1]
sorted_df = comparison_df.sort_values('MAE', ascending=False)
colors = ['#2ecc71' if t == 'Literature' else '#3498db' for t in sorted_df['Type']]
ax.barh(sorted_df['Model'], sorted_df['MAE'], color=colors)
ax.set_xlabel('MAE', fontsize=12)
ax.set_title('Model Comparison: MAE', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# RMSE comparison
ax = axes[1, 0]
sorted_df = comparison_df.sort_values('RMSE', ascending=False)
colors = ['#2ecc71' if t == 'Literature' else '#3498db' for t in sorted_df['Type']]
ax.barh(sorted_df['Model'], sorted_df['RMSE'], color=colors)
ax.set_xlabel('RMSE', fontsize=12)
ax.set_title('Model Comparison: RMSE', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Legend
ax = axes[1, 1]
ax.axis('off')
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', label='Classical ML'),
    Patch(facecolor='#2ecc71', label='Literature-based')
]
ax.legend(handles=legend_elements, loc='center', fontsize=14)
ax.text(0.5, 0.3, f'Total Models: 5\n7 Subjects\n69 Test Samples', 
        ha='center', va='center', fontsize=12, transform=ax.transAxes)

plt.tight_layout()
plt.savefig(viz_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved {viz_dir / 'model_comparison.png'}")

# Plot 2: Actual vs Predicted for All Models
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

model_names = ['Ridge', 'Random Forest', 'XGBoost', 'Ensemble', 'LSTM']
for idx, model_name in enumerate(model_names):
    ax = axes[idx]
    pred = predictions[model_name]
    
    ax.scatter(y_test, pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
            'r--', lw=2, label='Perfect Prediction')
    
    r2 = comparison_df[comparison_df['Model'].str.contains(model_name.split()[0])]['R²'].values[0]
    ax.set_xlabel('Actual Mindfulness Index', fontsize=11)
    ax.set_ylabel('Predicted Mindfulness Index', fontsize=11)
    ax.set_title(f'{model_name} (R² = {r2:.3f})', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

axes[5].axis('off')
plt.tight_layout()
plt.savefig(viz_dir / 'all_models_predictions.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved {viz_dir / 'all_models_predictions.png'}")

plt.close('all')

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PHASE 4 COMPLETE: COMPARATIVE ANALYSIS")
print("="*80)
print("\n✓ All 5 models compared")
print("✓ Statistical significance testing completed")
print("✓ Visualizations created")
print(f"\n🏆 BEST MODEL: {comparison_df.iloc[0]['Model']}")
print(f"   R² = {comparison_df.iloc[0]['R²']:.4f}")
print(f"   MAE = {comparison_df.iloc[0]['MAE']:.4f}")
print("\nNext: Phase 5 - Final Report & Documentation")
print("="*80)
