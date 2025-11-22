
"""models.py
Helper functions to train and evaluate classical ML models.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_evaluate_rf(X, y, param_grid=None, random_state=42):
    if param_grid is None:
        param_grid = {'n_estimators':[100,200], 'max_depth':[5,10,None]}
    rf = RandomForestRegressor(random_state=random_state)
    gs = GridSearchCV(rf, param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    gs.fit(X, y)
    best = gs.best_estimator_
    preds = best.predict(X)
    return best, {'mse': mean_squared_error(y, preds), 'mae': mean_absolute_error(y, preds), 'r2': r2_score(y, preds)}, gs.best_params_

def train_evaluate_ridge(X, y, param_grid=None):
    if param_grid is None:
        param_grid = {'alpha':[0.1,1.0,10.0]}
    ridge = Ridge()
    gs = GridSearchCV(ridge, param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    gs.fit(X, y)
    best = gs.best_estimator_
    preds = best.predict(X)
    return best, {'mse': mean_squared_error(y, preds), 'mae': mean_absolute_error(y, preds), 'r2': r2_score(y, preds)}, gs.best_params_
