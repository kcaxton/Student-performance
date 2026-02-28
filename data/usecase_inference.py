"""
Use-case: load saved Linear Regression math model and demonstrate inference.
Creates `feature_names.json` in the artifacts folder for future use.
Run from repository root: `python data/usecase_inference.py`
"""
from pathlib import Path
import joblib
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / 'artifacts' / 'lr_math.joblib'  # Changed from rf_math
DATA_CSV = ROOT / 'StudentsPerformance.csv'
ARTIFACTS = ROOT / 'artifacts'
ARTIFACTS.mkdir(exist_ok=True)

# Load model
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise SystemExit(f"Model not found at {MODEL_PATH}. Run analysing.ipynb first.")
print(f"Loaded model: {MODEL_PATH}")

# Reconstruct feature names from original dataset encoding
cat_cols = ['gender','race/ethnicity','parental level of education','lunch','test preparation course']
if not DATA_CSV.exists():
    raise SystemExit(f"Data CSV not found at {DATA_CSV}. Place StudentsPerformance.csv in {ROOT}")

df = pd.read_csv(DATA_CSV)
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
feature_names = df_encoded.drop(['math score','reading score','writing score'], axis=1).columns.tolist()

# Save feature names for reproducible inference
with open(ARTIFACTS / 'feature_names.json', 'w') as f:
    json.dump(feature_names, f)
print(f"Saved feature names ({len(feature_names)}) to {ARTIFACTS / 'feature_names.json'}")

# Example: single-sample prediction
sample = {
    "gender_male": 1,                           # 1 = male, 0 = female
    "race/ethnicity_group B": 0,
    "race/ethnicity_group C": 1,                # Student is group C
    "race/ethnicity_group D": 0,
    "race/ethnicity_group E": 0,
    "parental level of education_some high school": 0,
    "parental level of education_high school": 0,
    "parental level of education_some college": 1,  # Parents: some college
    "parental level of education_associate's degree": 0,
    "parental level of education_bachelor's degree": 0,
    "parental level of education_master's degree": 0,
    "lunch_standard": 1,                        # 1 = standard, 0 = free/reduced
    "test preparation course_none": 0           # 0 = completed course
}

# Build DataFrame and align columns (critical!)
X_sample = pd.DataFrame([sample])
X_sample = X_sample.reindex(columns=feature_names, fill_value=0)

pred = model.predict(X_sample)
print(f"Predicted math score (single sample): {pred[0]:.2f}")

# Batch example: predict first 5 rows from the encoded training data
X_train_like = df_encoded.drop(['math score','reading score','writing score'], axis=1)
batch_preds = model.predict(X_train_like.iloc[:5])
print("Predictions for first 5 rows of original dataset:", [round(float(p),2) for p in batch_preds])