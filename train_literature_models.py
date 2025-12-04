"""
Literature-based models implementation.

Paper A: Ensemble Method (Classical ML Pipeline)
Paper B: LSTM Deep Learning Model
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

sys.path.append('src')

from models import (
    LinearRegressionModel,
    RandomForestModel,
    XGBoostModel,
    compute_regression_metrics
)

print("="*70)
print("PHASE 3: LITERATURE-BASED MODELS")
print("="*70)

# Load data
print("\n1. Loading multi-subject data...")
data_dir = Path('data/processed')
X_train = pd.read_csv(data_dir / 'multi_X_train.csv')
y_train = pd.read_csv(data_dir / 'multi_y_train.csv').values.ravel()
X_test = pd.read_csv(data_dir / 'multi_X_test.csv')
y_test = pd.read_csv(data_dir / 'multi_y_test.csv').values.ravel()

feature_names = X_train.columns.tolist()
print(f"   ✓ Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================================================
# PAPER A: ENSEMBLE METHOD (Classical ML Pipeline)
# ============================================================================
print("\n" + "="*70)
print("PAPER A: ENSEMBLE METHOD")
print("="*70)
print("Approach: Weighted ensemble of Ridge, Random Forest, and XGBoost")
print("Rationale: Combines strengths of linear and non-linear models")

# Load pre-trained models
print("\nLoading pre-trained classical models...")
ridge_model = LinearRegressionModel(model_type='ridge', random_state=42)
rf_model = RandomForestModel(random_state=42)
xgb_model = XGBoostModel(random_state=42)

# Quick training (we already have best hyperparameters)
print("Training models with best hyperparameters...")

# Ridge with best params
ridge_model.train(X_train.values, y_train, feature_names=feature_names,
                  tune_hyperparams=False, verbose=0)

# RF with best params  
rf_model.train(X_train.values, y_train, feature_names=feature_names,
               tune_hyperparams=False, verbose=0)

# XGBoost with best params
xgb_model.train(X_train.values, y_train, feature_names=feature_names,
                tune_hyperparams=False, verbose=0)

# Get predictions
print("\nGenerating ensemble predictions...")
ridge_pred = ridge_model.predict(X_test.values)
rf_pred = rf_model.predict(X_test.values)
xgb_pred = xgb_model.predict(X_test.values)

# Weighted ensemble (weights based on validation performance)
# XGBoost: 0.50, RF: 0.30, Ridge: 0.20
ensemble_pred = 0.50 * xgb_pred + 0.30 * rf_pred + 0.20 * ridge_pred

# Evaluate ensemble
ensemble_metrics = compute_regression_metrics(y_test, ensemble_pred)

print("\n[Paper A: Ensemble] Test Set Performance:")
print(f"  R² Score:  {ensemble_metrics['R2']:.4f}")
print(f"  RMSE:      {ensemble_metrics['RMSE']:.4f}")
print(f"  MAE:       {ensemble_metrics['MAE']:.4f}")
print(f"  MSE:       {ensemble_metrics['MSE']:.4f}")

# Save ensemble predictions
ensemble_results = {
    'model_name': 'Ensemble (Paper A)',
    'predictions': ensemble_pred,
    'metrics': ensemble_metrics,
    'weights': {'XGBoost': 0.50, 'RandomForest': 0.30, 'Ridge': 0.20}
}

with open(data_dir / 'paper_a_ensemble.pkl', 'wb') as f:
    pickle.dump(ensemble_results, f)

print("\n✓ Paper A ensemble model saved")

# ============================================================================
# PAPER B: LSTM DEEP LEARNING MODEL
# ============================================================================
print("\n" + "="*70)
print("PAPER B: LSTM DEEP LEARNING MODEL")
print("="*70)
print("Approach: Simple LSTM for time-series feature learning")
print("Architecture: 2 LSTM layers + Dense output")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.preprocessing import StandardScaler
    
    print("\nBuilding LSTM model...")
    
    # Scale features for neural network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape for LSTM (samples, timesteps, features)
    # Since we don't have true time-series, we'll treat each feature as a timestep
    X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    # Build LSTM model
    model = keras.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Output 0-1 for mindfulness index
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    print("\nTraining LSTM model...")
    history = model.fit(
        X_train_lstm, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ]
    )
    
    # Predict
    lstm_pred = model.predict(X_test_lstm, verbose=0).flatten()
    
    # Evaluate
    lstm_metrics = compute_regression_metrics(y_test, lstm_pred)
    
    print("\n[Paper B: LSTM] Test Set Performance:")
    print(f"  R² Score:  {lstm_metrics['R2']:.4f}")
    print(f"  RMSE:      {lstm_metrics['RMSE']:.4f}")
    print(f"  MAE:       {lstm_metrics['MAE']:.4f}")
    print(f"  MSE:       {lstm_metrics['MSE']:.4f}")
    
    # Save model
    model.save(data_dir / 'paper_b_lstm_model.h5')
    
    lstm_results = {
        'model_name': 'LSTM (Paper B)',
        'predictions': lstm_pred,
        'metrics': lstm_metrics,
        'scaler': scaler,
        'history': history.history
    }
    
    with open(data_dir / 'paper_b_lstm.pkl', 'wb') as f:
        pickle.dump(lstm_results, f)
    
    print("\n✓ Paper B LSTM model saved")
    
except ImportError:
    print("\n⚠ TensorFlow not available. Using simpler alternative...")
    print("Implementing Gradient Boosting as Paper B alternative...")
    
    from sklearn.ensemble import GradientBoostingRegressor
    
    # Train Gradient Boosting
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    print("\nTraining Gradient Boosting model...")
    gb_model.fit(X_train, y_train)
    
    # Predict
    gb_pred = gb_model.predict(X_test)
    
    # Evaluate
    gb_metrics = compute_regression_metrics(y_test, gb_pred)
    
    print("\n[Paper B: Gradient Boosting] Test Set Performance:")
    print(f"  R² Score:  {gb_metrics['R2']:.4f}")
    print(f"  RMSE:      {gb_metrics['RMSE']:.4f}")
    print(f"  MAE:       {gb_metrics['MAE']:.4f}")
    print(f"  MSE:       {gb_metrics['MSE']:.4f}")
    
    # Save model
    lstm_results = {
        'model_name': 'Gradient Boosting (Paper B)',
        'predictions': gb_pred,
        'metrics': gb_metrics,
        'model': gb_model
    }
    
    with open(data_dir / 'paper_b_lstm.pkl', 'wb') as f:
        pickle.dump(lstm_results, f)
    
    print("\n✓ Paper B Gradient Boosting model saved")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("PHASE 3 COMPLETE: LITERATURE-BASED MODELS")
print("="*70)
print("\n✓ Paper A: Ensemble Method implemented and evaluated")
print("✓ Paper B: LSTM/Gradient Boosting implemented and evaluated")
print("\nNext: Phase 4 - Comparative Analysis of all 5 models")
print("="*70)
