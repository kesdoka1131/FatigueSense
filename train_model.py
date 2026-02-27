import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

# ─── Output directory ────────────────────────────────────────────────────────
OUTPUT_DIR = 'outputs/XGBoost-v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data — train on participants 1–9 only.
# Participants 10–12 are reserved as a true held-out test set.
df_all = pd.read_csv(f'{OUTPUT_DIR}/processed_features_windowed.csv')

# Prepare features and labels
# exclude the non-feature columns
exclude_cols = ['timestamp', 'participant_id', 'session_id', 'fatigue_level']
feature_cols = [col for col in df_all.columns if col not in exclude_cols]

# ── Per-Participant Z-Score Normalization ──────────────────────────────────
# Each participant has a different physiological baseline (e.g. resting EDA
# can differ by 10x). Normalising per-participant converts absolute values
# into "how far from *this person's* normal?" — dramatically improving
# generalisation to unseen participants.
print("Applying per-participant z-score normalization...")
for pid in df_all['participant_id'].unique():
    mask = df_all['participant_id'] == pid
    pid_data = df_all.loc[mask, feature_cols]
    pid_mean = pid_data.mean()
    pid_std  = pid_data.std().replace(0, 1)  # avoid division by zero
    df_all.loc[mask, feature_cols] = (pid_data - pid_mean) / pid_std
# ──────────────────────────────────────────────────────────────────────────

# Split and save unseen test data (participants 10–12)
test_df = df_all[df_all['participant_id'].astype(int) > 9].copy()
test_df.to_csv(f'{OUTPUT_DIR}/unseen_test_data.csv', index=False)
print(f"Saved held-out test set: {len(test_df)} rows (participants 10–12)")

df = df_all[df_all['participant_id'].astype(int) <= 9].copy()

X = df[feature_cols].copy()
y = df['fatigue_level']
groups = df['participant_id']

# Encode labels: Low=0, Medium=1, High=2
label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
y_encoded = y.map(label_mapping)

# Handle NaNs and Infs
X.replace([np.inf, -np.inf], np.nan, inplace=True)
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Compute balanced sample weights — upweights underrepresented patterns
sample_weights_all = compute_sample_weight('balanced', y_encoded)

print(f"Dataset shape (train only, participants 1-9): {df.shape}")
print(f"Features: {len(feature_cols)}")
print(f"Target distribution:\n{y.value_counts()}")

# Setup Leave-One-Subject-Out Cross Validation
logo = LeaveOneGroupOut()

def objective(trial):
    """Optuna objective function for tuning XGBoost with LOSO CV — optimises macro F1"""
    
    # Define hyperparameters to optimize
    param = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'random_state': 42
    }
    
    y_true_cv = []
    y_pred_cv = []
    
    for train_idx, test_idx in logo.split(X_imputed, y_encoded, groups):
        X_train, X_test = X_imputed.iloc[train_idx], X_imputed.iloc[test_idx]
        y_train, y_test = y_encoded.iloc[train_idx], y_encoded.iloc[test_idx]
        train_weights = sample_weights_all[train_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = xgb.XGBClassifier(**param)
        model.fit(X_train_scaled, y_train, sample_weight=train_weights)
        
        preds = model.predict(X_test_scaled)
        y_true_cv.extend(y_test)
        y_pred_cv.extend(preds)
    
    # Optimise macro F1 — penalises poor Medium recall equally
    macro_f1 = f1_score(y_true_cv, y_pred_cv, average='macro')
    return macro_f1

print("\nRunning Optuna Hyperparameter Optimization (100 trials, macro-F1 objective)...")
optuna.logging.set_verbosity(optuna.logging.WARNING) # Keep output clean
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("\nBest Optuna Hyperparameters:")
best_params = study.best_params
for key, value in best_params.items():
    print(f"  {key}: {value}")
print(f"Best Optuna Cross-Validation Macro-F1: {study.best_value:.4f}")

print("\nTraining final XGBoost model with optimized parameters...")
best_params['objective'] = 'multi:softprob'
best_params['num_class'] = 3
best_params['eval_metric'] = 'mlogloss'
best_params['random_state'] = 42

y_true = []
y_pred = []
feature_importances = np.zeros(len(feature_cols))

for fold, (train_idx, test_idx) in enumerate(logo.split(X_imputed, y_encoded, groups)):
    X_train, X_test = X_imputed.iloc[train_idx], X_imputed.iloc[test_idx]
    y_train, y_test = y_encoded.iloc[train_idx], y_encoded.iloc[test_idx]
    train_weights = sample_weights_all[train_idx]
    
    # Apply Normalization (Standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train best XGBoost with class weighting
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train_scaled, y_train, sample_weight=train_weights)
    
    # Predict
    preds = model.predict(X_test_scaled)
    
    # Store predictions and ground truth
    y_true.extend(y_test)
    y_pred.extend(preds)
    
    # Accumulate feature importances
    feature_importances += model.feature_importances_

# Calculate overall metrics
accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average='macro')
print(f"\n======================================")
print(f"Final Optimized Accuracy:  {accuracy:.4f}")
print(f"Final Optimized Macro-F1:  {macro_f1:.4f}")
print(f"======================================")
print("\nClassification Report:")
target_names = ['Low', 'Medium', 'High']
print(classification_report(y_true, y_pred, target_names=target_names))

# Average feature importances across folds
feature_importances /= logo.get_n_splits(groups=groups)

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot top 15 feature importances (more features now)
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
plt.title('Top 15 Feature Importances (XGBoost v2 — Macro-F1 Tuned)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/feature_importance.png')
print(f"Saved feature importance plot to {OUTPUT_DIR}/feature_importance.png")

# Generate and plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title(f'Optimized Confusion Matrix (Acc: {accuracy:.2f}, F1: {macro_f1:.2f})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrix.png')
print(f"Saved confusion matrix plot to {OUTPUT_DIR}/confusion_matrix.png")

# ──────────────────────────────────────────────────────────────────────────────
# Save deployment artifacts — model trained on ALL data with best Optuna params
# ──────────────────────────────────────────────────────────────────────────────
print("\nTraining final deployment model on full dataset...")
final_scaler = StandardScaler()
X_final_scaled = final_scaler.fit_transform(X_imputed)

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_final_scaled, y_encoded, sample_weight=sample_weights_all)

# Save artifacts
final_model.save_model(f'{OUTPUT_DIR}/xgboost_fatigue_model.json')
joblib.dump(final_scaler, f'{OUTPUT_DIR}/scaler.pkl')
joblib.dump(imputer, f'{OUTPUT_DIR}/imputer.pkl')
with open(f'{OUTPUT_DIR}/feature_cols.json', 'w') as f:
    json.dump(feature_cols, f)

print(f"Saved: {OUTPUT_DIR}/xgboost_fatigue_model.json")
print(f"Saved: {OUTPUT_DIR}/scaler.pkl")
print(f"Saved: {OUTPUT_DIR}/imputer.pkl")
print(f"Saved: {OUTPUT_DIR}/feature_cols.json")
print(f"\nAll deployment artifacts ready in {OUTPUT_DIR}/")

