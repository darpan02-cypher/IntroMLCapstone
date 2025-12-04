# MindMe AI: Mindfulness Prediction from Wearable Physiological Data

**Predicting mindfulness state using machine learning on WESAD dataset**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This project predicts a continuous **mindfulness index** from wearable physiological sensor data using machine learning. We use the WESAD (Wearable Stress and Affect Detection) dataset and define mindfulness as the inverse of stress intensity.

### Key Results
- **Best Model**: Ensemble Method (R² = 0.91)
- **Validation**: 7 subjects, 345 samples
- **Models**: 5 total (3 Classical ML + 2 Literature-based)

## 🎯 Objectives

1. Extract comprehensive features from physiological signals (HRV, EDA, respiratory, temperature, activity)
2. Implement 3 classical ML models (Linear Regression, Random Forest, XGBoost)
3. Implement 2 literature-based methods (Ensemble, LSTM)
4. Compare all models using MSE, MAE, R² metrics
5. Provide feature importance analysis and model interpretability

## 📊 Results Summary

### Model Performance (7 Subjects)

| Model | Type | R² | RMSE | MAE |
|-------|------|-----|------|-----|
| **Ensemble** | Literature | **0.906** | **0.124** | **0.082** |
| XGBoost | Classical | 0.902 | 0.127 | 0.076 |
| Random Forest | Classical | 0.870 | 0.146 | 0.102 |
| Ridge | Classical | 0.689 | 0.226 | 0.173 |
| LSTM | Literature | 0.605 | 0.255 | 0.154 |

### Top 5 Predictive Features
1. **activity_std** (110%) - Activity variability
2. **activity_mean** (31%) - Physical activity level
3. **scl_std** (29%) - Skin conductance variability
4. **eda_std** (10%) - EDA variability
5. **eda_max** (8%) - Maximum EDA

## 🗂️ Repository Structure

```
IntroMLCapstone/
├── data/
│   ├── processed/              # Extracted features and trained models
│   │   ├── multi_subject_features.csv
│   │   ├── best_multi_subject_model.pkl
│   │   └── final_all_models_comparison.csv
│   └── WESAD/                  # Raw WESAD dataset (download separately)
├── notebooks/
│   ├── 01_EDA_starter.ipynb              # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb      # Feature extraction demo
│   ├── 03_preprocessing.ipynb            # Data preprocessing
│   └── 04_classical_models.ipynb         # Classical ML models
├── src/
│   ├── features.py            # Feature extraction functions
│   ├── models.py              # Model training utilities
│   └── utils.py               # Data loading, visualization, evaluation
├── visualizations/            # Model comparison plots
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Getting Started

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/darpan02-cypher/IntroMLCapstone.git
   cd IntroMLCapstone
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download WESAD dataset**:
   - Download from: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/
   - Extract to `notebooks/data/WESAD/` directory

### Usage

#### Quick Start: Run All Models

```bash
# Extract features for 7 subjects
python extract_multi_subject_features.py

# Train classical models
python train_multi_subject_models.py

# Train literature-based models
python train_literature_models.py

# Run comparative analysis
python comparative_analysis.py
```

#### Using Jupyter Notebooks

```bash
jupyter notebook notebooks/04_classical_models.ipynb
```

## 📈 Methodology

### Target Variable Mapping

```
MindfulnessIndex = 1 - NormalizedStressScore
```

| WESAD Label | State | Mindfulness Index |
|-------------|-------|-------------------|
| 1 | Baseline | 1.0 (high) |
| 2 | Stress (TSST) | 0.0 (low) |
| 3 | Amusement | 0.7 (medium-high) |
| 4 | Meditation | 0.9 (high) |
| 0 | Transient | Excluded |

### Features Extracted (45 total)

- **HRV (14 features)**: RMSSD, SDNN, pNN50, LF/HF ratio, mean HR, etc.
- **EDA (15 features)**: SCL, SCR, phasic/tonic decomposition, peak counts
- **Respiratory (5 features)**: rate, depth, variability, I/E ratio
- **Temperature (6 features)**: mean, slope, variability
- **Activity (5 features)**: accelerometer magnitude, activity level

### Models Implemented

**Classical ML**:
1. Linear Regression with Regularization (Ridge, Lasso, ElasticNet)
2. Random Forest Regressor
3. XGBoost Regressor

**Literature-based**:
4. Ensemble Method (weighted combination of classical models)
5. LSTM Deep Learning (2-layer LSTM for time-series learning)

### Evaluation

- **Metrics**: MSE, RMSE, MAE, R²
- **Validation**: 7-subject multi-subject validation
- **Statistical Testing**: Wilcoxon signed-rank test
- **Visualizations**: Actual vs predicted, residuals, feature importance

## 🔬 Key Findings

1. **Ensemble method achieves best performance** (R² = 0.91) by combining strengths of multiple models

2. **Physical activity is the strongest predictor** (141% combined importance) - stillness correlates with mindfulness

3. **Multi-subject validation demonstrates robust generalization** across different individuals

4. **Tree-based models outperform linear models** - non-linear relationships are key

5. **EDA variability captures stress response dynamics** - important complementary signal

## 📚 Technologies Used

- **Signal Processing**: scipy, neurokit2, biosppy
- **Machine Learning**: scikit-learn, xgboost
- **Deep Learning**: tensorflow/keras
- **Visualization**: matplotlib, seaborn
- **Data Processing**: pandas, numpy

## 📝 Limitations

1. **Proxy Mapping**: Using stress as inverse proxy for mindfulness is a simplification
2. **Sample Size**: 7 subjects (345 samples) - larger cohort would strengthen conclusions
3. **Window Size**: 60-second windows may miss shorter-term dynamics
4. **Population**: WESAD participants may not represent general population

## 🔮 Future Work

1. Expand to all 15 WESAD subjects with Leave-One-Subject-Out CV
2. Implement advanced deep learning architectures (CNN-LSTM, Transformers)
3. Real-time deployment on wearable devices
4. Personalized models accounting for individual differences
5. Integration with mindfulness intervention studies

## 📧 Contact

**Project**: MindMe AI Capstone  
**Course**: Introduction to Machine Learning  
**Repository**: https://github.com/darpan02-cypher/IntroMLCapstone

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- WESAD dataset: Schmidt et al. (2018)
- Signal processing libraries: NeuroKit2, BioSPPy
- Machine learning frameworks: scikit-learn, XGBoost, TensorFlow

---

**Last Updated**: December 2025
