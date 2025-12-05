"""models.py
Model training and evaluation utilities for mindfulness prediction.

Provides wrapper classes for:
- Linear Regression with regularization (Ridge, Lasso, ElasticNet)
- Random Forest Regressor
- XGBoost Regressor

Includes hyperparameter tuning, evaluation metrics, and visualization utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import pickle
import time

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
try:
    import xgboost as xgb
except ImportError:
    xgb = None


# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
    
    Returns:
        Dictionary with MSE, RMSE, MAE, R² metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }


# ============================================================================
# BASE MODEL WRAPPER
# ============================================================================

class BaseRegressorWrapper(ABC):
    """
    Abstract base class for regression model wrappers.
    Provides standardized interface for training, prediction, and evaluation.
    """
    
    def __init__(self, name: str, random_state: int = 42):
        """
        Initialize base regressor.
        
        Args:
            name: Model name for display
            random_state: Random seed for reproducibility
        """
        self.name = name
        self.random_state = random_state
        self.model = None
        self.best_params_ = None
        self.cv_results_ = None
        self.training_time_ = None
        self.feature_names_ = None
        
    @abstractmethod
    def get_param_grid(self) -> Dict[str, List]:
        """Return hyperparameter grid for tuning."""
        pass
    
    @abstractmethod
    def create_model(self, **params) -> Any:
        """Create model instance with given parameters."""
        pass
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              feature_names: Optional[List[str]] = None,
              tune_hyperparams: bool = True,
              cv_folds: int = 5,
              verbose: int = 1) -> 'BaseRegressorWrapper':
        """
        Train the model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            feature_names: List of feature names (for importance analysis)
            tune_hyperparams: Whether to perform hyperparameter tuning
            cv_folds: Number of cross-validation folds
            verbose: Verbosity level
        
        Returns:
            Self (for method chaining)
        """
        self.feature_names_ = feature_names
        start_time = time.time()
        
        if tune_hyperparams:
            if verbose:
                print(f"[{self.name}] Starting hyperparameter tuning with {cv_folds}-fold CV...")
            
            param_grid = self.get_param_grid()
            base_model = self.create_model()
            
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv_folds,
                scoring='r2',
                n_jobs=-1,
                verbose=verbose
            )
            
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.best_params_ = grid_search.best_params_
            self.cv_results_ = grid_search.cv_results_
            
            if verbose:
                print(f"[{self.name}] Best parameters: {self.best_params_}")
                print(f"[{self.name}] Best CV R² score: {grid_search.best_score_:.4f}")
        else:
            if verbose:
                print(f"[{self.name}] Training with default parameters...")
            
            self.model = self.create_model()
            self.model.fit(X_train, y_train)
        
        self.training_time_ = time.time() - start_time
        
        if verbose:
            print(f"[{self.name}] Training completed in {self.training_time_:.2f}s")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
        
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray, 
                 dataset_name: str = "Test") -> Dict[str, float]:
        """
        Evaluate model on given dataset.
        
        Args:
            X: Features
            y_true: True target values
            dataset_name: Name of dataset (for display)
        
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        metrics = compute_regression_metrics(y_true, y_pred)
        
        print(f"\n[{self.name}] {dataset_name} Set Performance:")
        print(f"  R² Score:  {metrics['R2']:.4f}")
        print(f"  RMSE:      {metrics['RMSE']:.4f}")
        print(f"  MAE:       {metrics['MAE']:.4f}")
        print(f"  MSE:       {metrics['MSE']:.4f}")
        
        return metrics
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance (if available).
        
        Returns:
            DataFrame with features and importance scores, or None
        """
        return None  # Override in subclasses
    
    def save(self, filepath: str):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"[{self.name}] Model saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'BaseRegressorWrapper':
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model


# ============================================================================
# LINEAR REGRESSION MODELS
# ============================================================================

class LinearRegressionModel(BaseRegressorWrapper):
    """
    Wrapper for linear regression with regularization.
    Supports Ridge, Lasso, and ElasticNet.
    """
    
    def __init__(self, model_type: str = 'ridge', random_state: int = 42):
        """
        Initialize linear regression model.
        
        Args:
            model_type: Type of regularization ('ridge', 'lasso', 'elasticnet')
            random_state: Random seed
        """
        self.model_type = model_type.lower()
        name = f"Linear Regression ({model_type.capitalize()})"
        super().__init__(name, random_state)
    
    def create_model(self, **params):
        """Create linear model instance."""
        if self.model_type == 'ridge':
            return Ridge(random_state=self.random_state, **params)
        elif self.model_type == 'lasso':
            return Lasso(random_state=self.random_state, **params)
        elif self.model_type == 'elasticnet':
            return ElasticNet(random_state=self.random_state, **params)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def get_param_grid(self) -> Dict[str, List]:
        """Return hyperparameter grid."""
        if self.model_type in ['ridge', 'lasso']:
            return {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            }
        elif self.model_type == 'elasticnet':
            return {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance based on coefficient magnitudes.
        
        Returns:
            DataFrame with features and absolute coefficient values
        """
        if self.model is None or self.feature_names_ is None:
            return None
        
        coefficients = np.abs(self.model.coef_)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': coefficients
        }).sort_values('importance', ascending=False)
        
        return importance_df


# ============================================================================
# RANDOM FOREST MODEL
# ============================================================================

class RandomForestModel(BaseRegressorWrapper):
    """Wrapper for Random Forest Regressor."""
    
    def __init__(self, random_state: int = 42):
        """Initialize Random Forest model."""
        super().__init__("Random Forest", random_state)
    
    def create_model(self, **params):
        """Create Random Forest instance."""
        default_params = {
            'n_estimators': 100,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        default_params.update(params)
        return RandomForestRegressor(**default_params)
    
    def get_param_grid(self) -> Dict[str, List]:
        """Return hyperparameter grid."""
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance based on Gini importance.
        
        Returns:
            DataFrame with features and importance scores
        """
        if self.model is None or self.feature_names_ is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


# ============================================================================
# XGBOOST MODEL
# ============================================================================

class XGBoostModel(BaseRegressorWrapper):
    """Wrapper for XGBoost Regressor."""
    
    def __init__(self, random_state: int = 42):
        """Initialize XGBoost model."""
        super().__init__("XGBoost", random_state)
    
    def create_model(self, **params):
        """Create XGBoost instance."""
        if xgb is None:
            raise ImportError("XGBoost is not installed. Please install it with: pip install xgboost")
            
        default_params = {
            'n_estimators': 100,
            'random_state': self.random_state,
            'n_jobs': -1,
            'objective': 'reg:squarederror'
        }
        default_params.update(params)
        return xgb.XGBRegressor(**default_params)
    
    def get_param_grid(self) -> Dict[str, List]:
        """Return hyperparameter grid."""
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Optional[pd.DataFrame]:
        """
        Get feature importance.
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')
        
        Returns:
            DataFrame with features and importance scores
        """
        if self.model is None or self.feature_names_ is None:
            return None
        
        # Get importance scores
        importance_dict = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Map feature indices to names
        importance_data = []
        for i, feature_name in enumerate(self.feature_names_):
            feature_key = f'f{i}'
            importance_data.append({
                'feature': feature_name,
                'importance': importance_dict.get(feature_key, 0.0)
            })
        
        importance_df = pd.DataFrame(importance_data).sort_values('importance', ascending=False)
        
        return importance_df


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, 
                             model_name: str = "Model",
                             ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot actual vs predicted values.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        model_name: Name of model for title
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate R²
    r2 = r2_score(y_true, y_pred)
    
    ax.set_xlabel('Actual Mindfulness Index', fontsize=12)
    ax.set_ylabel('Predicted Mindfulness Index', fontsize=12)
    ax.set_title(f'{model_name}: Actual vs Predicted (R² = {r2:.4f})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                   model_name: str = "Model",
                   ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot residuals.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        model_name: Name of model for title
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    residuals = y_true - y_pred
    
    # Scatter plot of residuals
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    
    ax.set_xlabel('Predicted Mindfulness Index', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title(f'{model_name}: Residual Plot', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_residual_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                               model_name: str = "Model",
                               ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot residual distribution histogram.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        model_name: Name of model for title
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    residuals = y_true - y_pred
    
    ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', lw=2)
    
    ax.set_xlabel('Residuals', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{model_name}: Residual Distribution', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_feature_importance(importance_df: pd.DataFrame, 
                            top_n: int = 20,
                            model_name: str = "Model",
                            ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot feature importance as horizontal bar chart.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
        model_name: Name of model for title
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get top N features
    top_features = importance_df.head(top_n)
    
    # Create horizontal bar chart
    ax.barh(range(len(top_features)), top_features['importance'], align='center')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()  # Highest importance at top
    
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'{model_name}: Top {top_n} Feature Importance', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    return ax


def compare_models(models: List[BaseRegressorWrapper],
                   X_test: np.ndarray, 
                   y_test: np.ndarray) -> pd.DataFrame:
    """
    Compare multiple models on test set.
    
    Args:
        models: List of trained model wrappers
        X_test: Test features
        y_test: Test targets
    
    Returns:
        DataFrame with comparison metrics
    """
    results = []
    
    for model in models:
        y_pred = model.predict(X_test)
        metrics = compute_regression_metrics(y_test, y_pred)
        
        results.append({
            'Model': model.name,
            'R²': metrics['R2'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'Training Time (s)': model.training_time_
        })
    
    comparison_df = pd.DataFrame(results).sort_values('R²', ascending=False)
    
    return comparison_df


def plot_model_comparison(comparison_df: pd.DataFrame,
                         metric: str = 'R2',
                         ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot model comparison bar chart.
    
    Args:
        comparison_df: DataFrame from compare_models()
        metric: Metric to plot ('R2', 'RMSE', 'MAE', 'MSE')
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by metric (descending for R², ascending for errors)
    ascending = metric not in ['R2', 'R²']
    sorted_df = comparison_df.sort_values(metric, ascending=ascending)
    
    ax.barh(sorted_df['Model'], sorted_df[metric], color='skyblue', edgecolor='black')
    ax.set_xlabel(metric, fontsize=12)
    ax.set_title(f'Model Comparison: {metric}', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    return ax
