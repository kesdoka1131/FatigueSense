import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os

# Resolve paths relative to project root (one level above this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, 'outputs')


def load_artifacts():
    """Load all saved deployment artifacts from outputs/."""
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(OUTPUTS_DIR, 'xgboost_fatigue_model.json'))
    scaler   = joblib.load(os.path.join(OUTPUTS_DIR, 'scaler.pkl'))
    imputer  = joblib.load(os.path.join(OUTPUTS_DIR, 'imputer.pkl'))
    with open(os.path.join(OUTPUTS_DIR, 'feature_cols.json')) as f:
        feature_cols = json.load(f)
    return model, scaler, imputer, feature_cols


def evaluate_worker_state(worker_id, session_data, xgb_model, scaler, imputer, feature_cols):
    """
    Evaluates a worker's fatigue state using the trained XGBoost model
    and prescribes HR interventions based on Causal AI rules.
    """
    print(f"\n{'='*50}")
    print(f"👷 HR Alert System: Worker [{worker_id}] Evaluation")
    print(f"{'='*50}")

    # 1. Predict Fatigue Level
    row_imputed = pd.DataFrame(imputer.transform(session_data[feature_cols]), columns=feature_cols)
    row_scaled  = scaler.transform(row_imputed)
    prediction  = xgb_model.predict(row_scaled)[0]

    label_map     = {0: 'Low', 1: 'Medium', 2: 'High'}
    fatigue_state = label_map[int(prediction)]

    print(f"🚨 Predicted Fatigue State: {fatigue_state.upper()}")

    if fatigue_state == 'Low':
        print("✅ Status: Optimal. Worker is fully alert and focused.")
        print("💡 Intervention: None required. Continue current tasks.")
        return

    # 2. Causal AI Root-Cause Analysis
    # PCMCI confirmed: Beta_TP9 → HeartRate (val=0.15), EDA(t-1) → EDA (val=0.74)
    beta_eeg = session_data['muse_eeg_beta_TP9_mean'].values[0]
    alpha_eeg = session_data['muse_eeg_alpha_TP9_mean'].values[0]
    eda      = session_data['wrist_eda_eda_mean'].values[0]
    hr       = session_data['wrist_hr_hr_mean'].values[0]

    # Dataset-derived approximate medians as naive baselines
    baseline_beta  = 0.45
    baseline_eda   = 0.90
    baseline_hr    = 75.0

    print("\n🔍 Causal Root-Cause Analysis:")

    if beta_eeg > baseline_beta * 1.5 and (alpha_eeg < 1e-6 or beta_eeg / alpha_eeg > 1.2):
        print("   - Root Cause: Sustained high-frequency Beta brainwaves detected.")
        print("   - Causal Pathway: [Beta_EEG] -> [Mental Fatigue]")
        print("   - Diagnosis: Worker experiencing cognitive overload.")
        print("\n💊 HR Prescription:")
        print("   - Action: Task Rotation.")
        print("   - Suggestion: Move to a physically active but cognitively simple task for 30 minutes.")

    elif eda > baseline_eda * 1.5 and hr > baseline_hr * 1.1:
        print("   - Root Cause: EDA spike (stress/sweating) → elevated Heart Rate.")
        print("   - Causal Pathway: [EDA] -> [Heart Rate] -> [Physical/Global Fatigue]")
        print("   - Diagnosis: Acute physiological stress or intense exertion.")
        print("\n💊 HR Prescription:")
        print("   - Action: Mandatory Physical Rest.")
        print("   - Suggestion: 15-minute hydration break in the cool-down area.")

    else:
        print("   - Root Cause: Gradual compounding of previous fatigue states.")
        print("   - Causal Pathway: [Fatigue (t-1)] -> [Fatigue (t)]  (val=0.695)")
        print("   - Diagnosis: End-of-Shift compounded tiredness.")
        print("\n💊 HR Prescription:")
        print("   - Action: Monitor closely.")
        print("   - Suggestion: Do not assign to high-risk machinery for the rest of shift.")


def main():
    model, scaler, imputer, feature_cols = load_artifacts()
    print("✅ Loaded deployment artifacts from outputs/")

    df = pd.read_csv(os.path.join(OUTPUTS_DIR, 'processed_features_windowed.csv'))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ─────────────────────────────────────────────────────────────────
    # DEMO 1: Cognitive Overload — very high Beta EEG, normal EDA/HR
    # ─────────────────────────────────────────────────────────────────
    dummy_cognitive = df[feature_cols].dropna().iloc[50:51].copy()
    dummy_cognitive['muse_eeg_beta_TP9_mean']  = 1.8    # Massive Beta spike
    dummy_cognitive['muse_eeg_alpha_TP9_mean'] = 0.4    # Suppressed Alpha
    dummy_cognitive['wrist_eda_eda_mean']      = 0.85   # Normal EDA
    dummy_cognitive['wrist_hr_hr_mean']        = 74.0   # Normal HR

    evaluate_worker_state("P12 (Machinery Inspector)", dummy_cognitive,
                          model, scaler, imputer, feature_cols)

    # ─────────────────────────────────────────────────────────────────
    # DEMO 2: Acute Physical Stress — high EDA and HR, normal brainwaves
    # ─────────────────────────────────────────────────────────────────
    dummy_physical = df[feature_cols].dropna().iloc[0:1].copy()
    dummy_physical['muse_eeg_beta_TP9_mean']  = 0.35    # Normal Beta
    dummy_physical['wrist_eda_eda_mean']      = 2.2     # High EDA
    dummy_physical['wrist_hr_hr_mean']        = 112.0   # High HR

    evaluate_worker_state("P04 (Heavy Lifter)", dummy_physical,
                          model, scaler, imputer, feature_cols)


if __name__ == "__main__":
    main()
