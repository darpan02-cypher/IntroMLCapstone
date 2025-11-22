# MindMe AI Capstone: Mindfulness Prediction from Wearable Data

**Predicting mindfulness state using physiological signals from the WESAD dataset**

## 📋 Project Overview

This project predicts a continuous **mindfulness index** from wearable physiological sensor data using machine learning. We use the WESAD (Wearable Stress and Affect Detection) dataset and define mindfulness as the inverse of stress intensity:

```
MindfulnessIndex = 1 - NormalizedStressScore
```

**Rationale**: Research shows that mindfulness is associated with lower physiological stress. By predicting stress intensity from wearable sensors, we can infer mindfulness state.

### Objectives

1. Extract comprehensive features from physiological signals (HRV, EDA, respiratory, temperature)
2. Implement 3 classical ML models (Linear Regression, Random Forest, XGBoost)
3. Reproduce 2 literature-based methods from recent stress detection papers
4. Compare all models using MSE, MAE, R² metrics
5. Provide feature importance analysis and model interpretability

### Dataset

**WESAD** (Wearable Stress and Affect Detection)
- 15 subjects performing stress-inducing tasks
- Chest device (RespiBAN): ECG, EDA, EMG, Temp, Respiration, Accelerometer @ 700 Hz
- Wrist device (Empatica E4): BVP, EDA, Temp, Accelerometer @ varying rates
- Labels: Baseline, Stress (TSST), Amusement, Meditation

**Target Variable Mapping**:
- Baseline (label=1) → Mindfulness = 1.0 (high)
- Stress (label=2) → Mindfulness = 0.0 (low)
- Amusement (label=3) → Mindfulness = 0.7 (medium-high)
- Meditation (label=4) → Mindfulness = 0.9 (high)
- Transient (label=0) → Excluded

## 🗂️ Repository Structure

```
IntroMLCapstone/
├── data/
│   ├── WESAD/              # Raw WESAD dataset (download separately)
│   └── processed/          # Extracted feature matrices
├── notebooks/
│   ├── 01_EDA_starter.ipynb              # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb      # Feature extraction demo
│   ├── 03_preprocessing.ipynb            # Data preprocessing (TODO)
│   ├── 04_model_linear.ipynb             # Linear regression (TODO)
│   ├── 05_model_rf_xgb.ipynb             # RF & XGBoost (TODO)
│   ├── 06_paperA_repro.ipynb             # Literature method #1 (TODO)
│   └── 07_paperB_repro.ipynb             # Literature method #2 (TODO)
├── src/
│   ├── features.py         # Feature extraction functions
│   ├── models.py           # Model training utilities
│   └── utils.py            # Data loading, visualization, evaluation
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Getting Started

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd IntroMLCapstone
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download WESAD dataset**:
   - Download from: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/
   - Extract to `data/WESAD/` directory
   - Structure should be: `data/WESAD/S2/S2.pkl`, `data/WESAD/S3/S3.pkl`, etc.

### Usage

1. **Exploratory Data Analysis**:
   ```bash
   jupyter notebook notebooks/01_EDA_starter.ipynb
   ```

2. **Feature Engineering**:
   ```bash
   jupyter notebook notebooks/02_feature_engineering.ipynb
   ```
   This will extract features and save to `data/processed/`

3. **Model Training** (upcoming):
   - Run preprocessing notebook
   - Train classical models (Linear, RF, XGBoost)
   - Reproduce literature methods
   - Compare results

## 📊 Features Extracted

### HRV (Heart Rate Variability) - 14 features
**Time-domain**: RMSSD, SDNN, pNN50, mean HR, HR std, HR range  
**Frequency-domain**: LF power, HF power, LF/HF ratio, normalized LF/HF, total power

### EDA (Electrodermal Activity) - 15 features
**Tonic (SCL)**: mean, std, min, max, range  
**Phasic (SCR)**: count, rate, mean amplitude, max amplitude  
**Overall**: mean, std, slope

### Respiratory - 5 features
Rate (breaths/min), depth (mean, std), variability, I/E ratio

### Temperature - 6 features
Mean, std, min, max, range, slope

### Activity (Accelerometer) - 5 features
Mean magnitude, std, max, activity level, posture stability

**Total: ~45 features per 60-second window**

## 🎯 Timeline (20-Day Plan)

| Days | Phase | Tasks |
|------|-------|-------|
| 1-2 | Data Understanding | EDA, target variable definition |
| **3-4** | **Feature Engineering** | **Extract all features (CURRENT)** |
| 5-6 | Classical Model #1 | Linear Regression |
| 7-8 | Classical Model #2 | Random Forest |
| 9-10 | Classical Model #3 | XGBoost |
| 11-12 | Literature Method #1 | Paper A reproduction |
| 13-14 | Literature Method #2 | Paper B reproduction |
| 15-16 | Comparative Analysis | Evaluation, visualization |
| 17-19 | Report Writing | IEEE-style paper |
| 20 | Finalization | Code cleanup, submission |

## 📈 Current Progress

- ✅ EDA completed (Notebook 01)
- ✅ Feature extraction implemented (`features.py`)
- ✅ Feature engineering notebook created (Notebook 02)
- ✅ Utility functions (`utils.py`)
- ⏳ Preprocessing (Notebook 03)
- ⏳ Classical models (Notebooks 04-05)
- ⏳ Literature methods (Notebooks 06-07)

## 🔬 Key Technologies

- **Signal Processing**: scipy, neurokit2, biosppy
- **Machine Learning**: scikit-learn, xgboost
- **Deep Learning**: tensorflow (for literature methods)
- **Visualization**: matplotlib, seaborn
- **Explainability**: SHAP

## 📝 Notes & Limitations

1. **Proxy Mapping**: We use stress as an inverse proxy for mindfulness. This is a simplification and will be discussed in the final report.

2. **Single Subject Development**: Initial development uses Subject S2. Full pipeline will be extended to all subjects.

3. **Signal Quality**: Some features may have missing values due to signal artifacts or insufficient data in short windows.

4. **Window Size**: 60-second windows are used based on literature. This is a hyperparameter that could be optimized.

## 📚 References

- Schmidt et al. (2018). "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection"
- Garg et al. (2021). "Stress Detection by Machine Learning and Wearable Sensors"
- Additional papers to be cited in literature reproduction notebooks

## 📧 Contact

**Project**: MindMe AI Capstone  
**Course**: Introduction to Machine Learning  
**Document**: https://docs.google.com/document/d/1gJCX5k3KoeVUo9sWz0jhugut1wbEJvxMhEa38w_UFx0/edit?tab=t.0

---

**Last Updated**: November 2025
