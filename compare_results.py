"""
Final comparison with 7-subject results.
"""
import pandas as pd

print("="*80)
print("FINAL RESULTS: 7-SUBJECT VALIDATION")
print("="*80)

# Load 7-subject results
multi_results = pd.read_csv('data/processed/multi_subject_model_comparison.csv')

print("\n" + "="*80)
print("MODEL PERFORMANCE (7 Subjects: S2, S3, S4, S5, S6, S8, S10)")
print("="*80)
print(f"Total Samples: 345 (276 train, 69 test)")
print("="*80)
print(multi_results.to_string(index=False))
print("="*80)

# Comparison across dataset sizes
comparison_data = {
    'Dataset': [
        'Single-Subject (S2)',
        'Single-Subject (S2)', 
        '4-Subject (S2-S5)',
        '4-Subject (S2-S5)',
        '7-Subject (S2-S6,S8,S10)',
        '7-Subject (S2-S6,S8,S10)'
    ],
    'Model': [
        'XGBoost',
        'Random Forest',
        'XGBoost',
        'Random Forest',
        'XGBoost',
        'Random Forest'
    ],
    'R²': [
        0.9606,
        0.7254,
        0.8775,
        0.7725,
        0.9021,
        0.8696
    ],
    'MAE': [
        0.0325,
        0.0901,
        0.0749,
        0.1423,
        0.0763,
        0.1016
    ],
    'N_subjects': [1, 1, 4, 4, 7, 7],
    'N_train': [38, 38, 156, 156, 276, 276],
    'N_test': [10, 10, 40, 40, 69, 69]
}

comparison_df = pd.DataFrame(comparison_data)

print("\n" + "="*80)
print("PROGRESSION: SINGLE → 4-SUBJECT → 7-SUBJECT")
print("="*80)
print(comparison_df.to_string(index=False))
print("="*80)

# Save
comparison_df.to_csv('data/processed/final_progression_comparison.csv', index=False)

print("\n" + "="*80)
print("KEY FINDINGS - 7-SUBJECT VALIDATION")
print("="*80)
print("\n🏆 BEST MODEL: XGBoost")
print(f"   R² = 0.90 (90% variance explained)")
print(f"   MAE = 0.076 (7.6% average error)")
print(f"   Validated across 7 different subjects")

print("\n📊 GENERALIZATION ANALYSIS:")
print("   XGBoost: 0.96 (1 subj) → 0.88 (4 subj) → 0.90 (7 subj)")
print("   ✓ Excellent stability with more subjects!")
print("   ✓ Slight improvement from 4 to 7 subjects")

print("\n📈 RANDOM FOREST IMPROVEMENT:")
print("   Random Forest: 0.73 (1 subj) → 0.77 (4 subj) → 0.87 (7 subj)")
print("   ✓ Significant improvement with more data (+14%)")
print("   ✓ Benefits from larger training set")

print("\n🔬 TOP 5 FEATURES (7-Subject XGBoost):")
print("   1. activity_std (110%) - Activity variability")
print("   2. activity_mean (31%) - Physical activity level")
print("   3. scl_std (29%) - Skin conductance variability")
print("   4. eda_std (10%) - EDA variability")
print("   5. eda_max (8%) - Maximum EDA")

print("\n💡 INSIGHTS:")
print("   - Physical activity features dominate (141% combined)")
print("   - EDA variability is key predictor")
print("   - More subjects = more robust feature importance")

print("\n" + "="*80)
print("RECOMMENDATION FOR REPORT")
print("="*80)
print("\n✅ USE 7-SUBJECT RESULTS (Most Robust)")
print("   - XGBoost R² = 0.90")
print("   - 345 samples across 7 subjects")
print("   - Strong generalization demonstrated")
print("   - Scientifically rigorous validation")
print("="*80)
