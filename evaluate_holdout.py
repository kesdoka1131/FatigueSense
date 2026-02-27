"""
evaluate_holdout.py
-------------------
Evaluates the trained XGBoost model (trained on participants 1–9) on the
truly held-out test set: participants 10–12 (outputs/XGBoost-v2/unseen_test_data.csv).

Run:
    python evaluate_holdout.py
"""

import pandas as pd
import numpy as np
import json
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ─── Output directory ────────────────────────────────────────────────────────
OUTPUT_DIR = 'outputs/XGBoost-v2'

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load artifacts (trained on participants 1–9)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading deployment artifacts...")
model   = xgb.XGBClassifier()
model.load_model(f'{OUTPUT_DIR}/xgboost_fatigue_model.json')
scaler  = joblib.load(f'{OUTPUT_DIR}/scaler.pkl')
imputer = joblib.load(f'{OUTPUT_DIR}/imputer.pkl')
with open(f'{OUTPUT_DIR}/feature_cols.json') as f:
    feature_cols = json.load(f)

label_map     = {0: 'Low', 1: 'Medium', 2: 'High'}
label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load held-out test set (participants 10–12)
# ─────────────────────────────────────────────────────────────────────────────
test_df = pd.read_csv(f'{OUTPUT_DIR}/unseen_test_data.csv')
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)

print(f"\n=== Held-Out Test Set ===")
print(f"Participants : {sorted(test_df['participant_id'].unique())}")
print(f"Total rows   : {len(test_df)}")
print(f"Label dist   :\n{test_df['fatigue_level'].value_counts()}")

X_test = test_df[feature_cols]
y_test = test_df['fatigue_level'].map(label_mapping)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Impute → Scale → Predict
# ─────────────────────────────────────────────────────────────────────────────
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)
X_test_scaled  = scaler.transform(X_test_imputed)

y_pred     = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Overall metrics
# ─────────────────────────────────────────────────────────────────────────────
target_names = ['Low', 'Medium', 'High']
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')

print(f"\n{'='*50}")
print(f"  HELD-OUT TEST ACCURACY : {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"  HELD-OUT TEST MACRO-F1 : {macro_f1:.4f}")
print(f"{'='*50}")
print(f"\nClassification Report (participants 10–12):")
print(classification_report(y_test, y_pred, target_names=target_names))

# ─────────────────────────────────────────────────────────────────────────────
# 5. Per-participant breakdown
# ─────────────────────────────────────────────────────────────────────────────
print("Per-Participant Accuracy:")
test_df_copy = test_df.copy()
test_df_copy['y_true'] = y_test.values
test_df_copy['y_pred'] = y_pred

for pid in sorted(test_df_copy['participant_id'].unique()):
    sub = test_df_copy[test_df_copy['participant_id'] == pid]
    acc = accuracy_score(sub['y_true'], sub['y_pred'])
    print(f"  Participant {pid:02d} : {acc:.4f}  ({len(sub)} windows)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title(f'Held-Out Confusion Matrix — Participants 10–12\n(Accuracy: {accuracy*100:.1f}%, Macro-F1: {macro_f1:.2f})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/holdout_confusion_matrix.png')
print(f"\nSaved: {OUTPUT_DIR}/holdout_confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Prediction confidence summary
# ─────────────────────────────────────────────────────────────────────────────
print("\nMean prediction confidence per true class:")
for true_label, idx in label_mapping.items():
    mask = (y_test == idx).values
    if mask.sum() == 0:
        continue
    mean_conf = y_pred_proba[mask, idx].mean()
    print(f"  True={true_label:6s} → avg P({true_label}) = {mean_conf:.3f}")

print("\nDone.")

