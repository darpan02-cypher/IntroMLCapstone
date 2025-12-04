"""
Extract features for multiple WESAD subjects and create combined dataset.
Fast implementation for multi-subject validation.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

sys.path.append('src')

from features import create_feature_matrix
from utils import load_wesad_subject

print("="*70)
print("MULTI-SUBJECT FEATURE EXTRACTION")
print("="*70)

# Configuration
SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S8', 'S10']  # 7 subjects
DATA_DIR = Path('notebooks/data/WESAD')
OUTPUT_DIR = Path('data/processed')
WINDOW_SIZE = 60  # seconds
RANDOM_STATE = 42

# Create output directory if needed
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

all_features = []

for subject_id in SUBJECTS:
    print(f"\n{'='*70}")
    print(f"Processing {subject_id}...")
    print(f"{'='*70}")
    
    try:
        # Load subject data
        subject_data = load_wesad_subject(subject_id, data_dir=DATA_DIR)
        
        # Extract features
        print(f"Extracting features with {WINDOW_SIZE}s windows...")
        features_df = create_feature_matrix(
            subject_data, 
            window_size_sec=WINDOW_SIZE, 
            overlap=0.0
        )
        
        # Add subject ID column
        features_df['subject_id'] = subject_id
        
        print(f"✓ Extracted {len(features_df)} windows")
        print(f"  Features: {features_df.shape[1] - 1} (excluding subject_id)")
        print(f"  Target distribution:")
        print(features_df['mindfulness_index'].value_counts().sort_index())
        
        all_features.append(features_df)
        
        # Save individual subject features
        subject_output = OUTPUT_DIR / f'{subject_id}_features.csv'
        features_df.to_csv(subject_output, index=False)
        print(f"✓ Saved to {subject_output}")
        
    except Exception as e:
        print(f"✗ Error processing {subject_id}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Combine all subjects
print(f"\n{'='*70}")
print("COMBINING ALL SUBJECTS")
print(f"{'='*70}")

if len(all_features) > 0:
    combined_df = pd.concat(all_features, ignore_index=True)
    
    print(f"\nCombined dataset:")
    print(f"  Total samples: {len(combined_df)}")
    print(f"  Subjects: {combined_df['subject_id'].unique()}")
    print(f"  Features: {combined_df.shape[1] - 5}")  # Excluding metadata columns
    
    print(f"\nSamples per subject:")
    print(combined_df['subject_id'].value_counts().sort_index())
    
    print(f"\nTarget distribution (all subjects):")
    print(combined_df['mindfulness_index'].value_counts().sort_index())
    
    # Save combined dataset
    combined_output = OUTPUT_DIR / 'multi_subject_features.csv'
    combined_df.to_csv(combined_output, index=False)
    print(f"\n✓ Combined dataset saved to {combined_output}")
    
    # Create train/test split
    from sklearn.model_selection import train_test_split
    
    # Separate features and target
    metadata_cols = ['window_start', 'window_end', 'label', 'mindfulness_index', 'subject_id']
    feature_cols = [col for col in combined_df.columns if col not in metadata_cols]
    
    X = combined_df[feature_cols]
    y = combined_df['mindfulness_index']
    subjects = combined_df['subject_id']
    
    # Stratified split by subject to ensure all subjects in both sets
    X_train, X_test, y_train, y_test, subj_train, subj_test = train_test_split(
        X, y, subjects,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=subjects  # Ensure each subject represented in both sets
    )
    
    print(f"\n{'='*70}")
    print("TRAIN/TEST SPLIT")
    print(f"{'='*70}")
    print(f"Training set: {len(X_train)} samples")
    print(f"  Subjects: {subj_train.value_counts().sort_index().to_dict()}")
    print(f"\nTest set: {len(X_test)} samples")
    print(f"  Subjects: {subj_test.value_counts().sort_index().to_dict()}")
    
    # Save splits
    X_train.to_csv(OUTPUT_DIR / 'multi_X_train.csv', index=False)
    X_test.to_csv(OUTPUT_DIR / 'multi_X_test.csv', index=False)
    pd.DataFrame(y_train).to_csv(OUTPUT_DIR / 'multi_y_train.csv', index=False, header=['mindfulness_index'])
    pd.DataFrame(y_test).to_csv(OUTPUT_DIR / 'multi_y_test.csv', index=False, header=['mindfulness_index'])
    
    # Save feature names
    with open(OUTPUT_DIR / 'multi_feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_cols))
    
    print(f"\n✓ Train/test splits saved to {OUTPUT_DIR}/multi_*.csv")
    
    # Save metadata
    metadata = {
        'subjects': SUBJECTS,
        'n_subjects': len(SUBJECTS),
        'total_samples': len(combined_df),
        'n_features': len(feature_cols),
        'window_size_sec': WINDOW_SIZE,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'random_state': RANDOM_STATE
    }
    
    with open(OUTPUT_DIR / 'multi_metadata.json', 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*70}")
    print("✓ MULTI-SUBJECT FEATURE EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"\nNext step: Run multi-subject model training")
    print(f"  python train_multi_subject_models.py")
    
else:
    print("\n✗ No features extracted. Check errors above.")
