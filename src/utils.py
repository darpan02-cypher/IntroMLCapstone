"""utils.py
Utility functions for data loading, visualization, and evaluation.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================================
# DATA LOADING
# ============================================================================

def load_wesad_subject(subject_id, data_dir='../data/WESAD'):
    """
    Load WESAD data for a single subject.
    
    Parameters:
    -----------
    subject_id : str or int
        Subject ID (e.g., 'S2' or 2)
    data_dir : str
        Path to WESAD data directory
    
    Returns:
    --------
    dict : Dictionary containing subject data with keys:
        - 'signal': physiological signals
        - 'label': activity labels
        - 'subject': subject metadata
    """
    # Format subject ID
    if isinstance(subject_id, int):
        subject_id = f'S{subject_id}'
    
    # Load pickle file
    data_dir = Path(data_dir)
    pkl_path = data_dir / subject_id / f'{subject_id}.pkl'
    
    if not pkl_path.exists():
        raise FileNotFoundError(f'Subject file not found: {pkl_path}')
    
    print(f'Loading {pkl_path}...')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    return data


def load_all_subjects(data_dir='../data/WESAD'):
    """
    Load all available WESAD subjects.
    
    Parameters:
    -----------
    data_dir : str
        Path to WESAD data directory
    
    Returns:
    --------
    dict : Dictionary mapping subject_id -> subject_data
    """
    data_dir = Path(data_dir)
    subject_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('S')])
    subject_ids = [d.name for d in subject_dirs]
    
    all_subjects_data = {}
    for subject_id in subject_ids:
        try:
            all_subjects_data[subject_id] = load_wesad_subject(subject_id, data_dir)
            print(f"Loaded {subject_id}")
        except Exception as e:
            print(f"Error loading {subject_id}: {e}")
    
    return all_subjects_data


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_signal_comparison(features_df, feature_cols, condition_col='label', 
                           condition_names=None, figsize=(14, 8)):
    """
    Plot feature distributions across different conditions.
    
    Parameters:
    -----------
    features_df : DataFrame
        Feature matrix with condition labels
    feature_cols : list
        List of feature column names to plot
    condition_col : str
        Column name for condition labels
    condition_names : dict
        Mapping from condition values to names
    """
    if condition_names is None:
        condition_names = {
            0: 'Transient',
            1: 'Baseline',
            2: 'Stress',
            3: 'Amusement',
            4: 'Meditation'
        }
    
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(feature_cols):
        ax = axes[idx]
        
        # Create boxplot
        data_to_plot = []
        labels_to_plot = []
        
        for condition_val in sorted(features_df[condition_col].unique()):
            if condition_val in condition_names:
                data_to_plot.append(features_df[features_df[condition_col] == condition_val][feature].dropna())
                labels_to_plot.append(condition_names[condition_val])
        
        ax.boxplot(data_to_plot, labels=labels_to_plot)
        ax.set_title(feature)
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def plot_actual_vs_predicted(y_true, y_pred, model_name='Model', figsize=(10, 5)):
    """
    Plot actual vs predicted values and residuals.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str
        Name of the model for plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Scatter plot: actual vs predicted
    axes[0].scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect prediction')
    axes[0].set_xlabel('Actual Mindfulness Index')
    axes[0].set_ylabel('Predicted Mindfulness Index')
    axes[0].set_title(f'{model_name}: Actual vs Predicted')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Mindfulness Index')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title(f'{model_name}: Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(feature_names, importances, top_n=20, figsize=(10, 8)):
    """
    Plot feature importances.
    
    Parameters:
    -----------
    feature_names : list
        List of feature names
    importances : array-like
        Feature importance values
    top_n : int
        Number of top features to display
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(top_n), importances[indices])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_regression_metrics(y_true, y_pred):
    """
    Compute comprehensive regression metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    
    Returns:
    --------
    dict : Dictionary of metrics
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100  # Avoid division by zero
    }
    
    return metrics


def print_metrics(metrics, model_name='Model'):
    """
    Print regression metrics in a formatted way.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics from compute_regression_metrics()
    model_name : str
        Name of the model
    """
    print(f"\n{'='*50}")
    print(f"{model_name} Performance Metrics")
    print(f"{'='*50}")
    print(f"MSE:   {metrics['mse']:.6f}")
    print(f"RMSE:  {metrics['rmse']:.6f}")
    print(f"MAE:   {metrics['mae']:.6f}")
    print(f"R²:    {metrics['r2']:.6f}")
    print(f"MAPE:  {metrics['mape']:.2f}%")
    print(f"{'='*50}\n")


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def leave_one_subject_out_split(features_df, subject_col='subject_id'):
    """
    Create Leave-One-Subject-Out (LOSO) cross-validation splits.
    
    Parameters:
    -----------
    features_df : DataFrame
        Feature matrix with subject identifiers
    subject_col : str
        Column name for subject IDs
    
    Returns:
    --------
    generator : Yields (train_idx, test_idx) tuples
    """
    logo = LeaveOneGroupOut()
    groups = features_df[subject_col].values
    
    for train_idx, test_idx in logo.split(features_df, groups=groups):
        yield train_idx, test_idx


def create_train_test_split(features_df, test_size=0.2, random_state=42, 
                            stratify_col=None):
    """
    Create train/test split with optional stratification.
    
    Parameters:
    -----------
    features_df : DataFrame
        Feature matrix
    test_size : float
        Fraction of data for testing
    random_state : int
        Random seed
    stratify_col : str
        Column to stratify on (e.g., 'label')
    
    Returns:
    --------
    tuple : (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    if stratify_col and stratify_col in features_df.columns:
        stratify = features_df[stratify_col]
    else:
        stratify = None
    
    train_df, test_df = train_test_split(
        features_df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify
    )
    
    return train_df, test_df
