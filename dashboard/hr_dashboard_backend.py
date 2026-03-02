import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os

# Resolve paths relative to project root (one level above this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, 'outputs', 'XGBoost-v2')


def load_artifacts():
    """Load all saved deployment artifacts from outputs/XGBoost-v2/."""
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
    and prescribes HR interventions based on V2 Causal AI rules (z-scored).
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

    # 2. Causal AI Root-Cause Analysis (using v2 z-scored features)
    beta     = session_data['muse_eeg_beta_TP9_mean'].values[0]
    tb_ratio = session_data.get('theta_beta_ratio_TP9', pd.Series([np.nan])).values[0]
    ab_ratio = session_data.get('alpha_beta_ratio_TP9', pd.Series([np.nan])).values[0]
    eda      = session_data['wrist_eda_eda_mean'].values[0]
    hr       = session_data['wrist_hr_hr_mean'].values[0]

    print("\n🔍 Causal Root-Cause Analysis:")

    if not np.isnan(tb_ratio) and tb_ratio > 1.0:
        print("   - Root Cause: Elevated Theta/Beta Ratio (z-score: >+1.0).")
        print("   - Causal Pathway: [Theta/Beta] -> [Drowsiness / Microsleeps]")
        print("   - Diagnosis: Worker shows neurological signs of micro-sleeps.")
        print("\n💊 HR Prescription:")
        print("   - Action: Immediate Break.")
        print("   - Suggestion: Remove from hazardous task immediately and enforce a 20-minute rest.")

    elif (not np.isnan(beta) and beta > 1.0) or (not np.isnan(ab_ratio) and ab_ratio < -1.0):
        print("   - Root Cause: Sustained high-frequency Beta brainwaves detected (z-score: >+1.0).")
        print("   - Causal Pathway: [Beta_EEG] -> [Cognitive Overload]")
        print("   - Diagnosis: Worker experiencing mental stress and cognitive forcing.")
        print("\n💊 HR Prescription:")
        print("   - Action: Task Rotation.")
        print("   - Suggestion: Move to a physically active but cognitively simple task for 30 minutes.")

    elif eda > 1.0 and hr > 1.0:
        print("   - Root Cause: EDA and Heart Rate concurrently elevated (z-scores: >+1.0).")
        print("   - Causal Pathway: [EDA] & [Heart Rate] -> [Physical/Global Fatigue]")
        print("   - Diagnosis: Acute physiological stress or intense exertion.")
        print("\n💊 HR Prescription:")
        print("   - Action: Mandatory Physical Rest.")
        print("   - Suggestion: 15-minute hydration break in the cool-down area.")

    else:
        print("   - Root Cause: Gradual compounding of physiological shifts (no single acute spike).")
        print("   - Causal Pathway: Multiple interacting variables over time.")
        print("   - Diagnosis: End-of-Shift compounded wear.")
        print("\n💊 HR Prescription:")
        print("   - Action: Monitor closely.")
        print("   - Suggestion: Do not assign to high-risk machinery for the rest of shift.")


def main():
    model, scaler, imputer, feature_cols = load_artifacts()
    print("✅ Loaded deployment artifacts from outputs/XGBoost-v2/")

    df = pd.read_csv(os.path.join(OUTPUTS_DIR, 'unseen_test_data.csv'))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ─────────────────────────────────────────────────────────────────
    # DEMO 1: Microsleep Risk (High Theta/Beta)
    # ─────────────────────────────────────────────────────────────────
    dummy_drowsy = df[feature_cols].dropna().iloc[50:51].copy()
    dummy_drowsy['theta_beta_ratio_TP9']      = 1.5     # High Drowsiness marker
    dummy_drowsy['muse_eeg_beta_TP9_mean']    = -0.5    # Low alertness
    dummy_drowsy['wrist_eda_eda_mean']        = 0.0     # Normal EDA
    dummy_drowsy['wrist_hr_hr_mean']          = -0.2    # Normal HR

    evaluate_worker_state("P12 (Machinery Inspector)", dummy_drowsy,
                          model, scaler, imputer, feature_cols)

    # ─────────────────────────────────────────────────────────────────
    # DEMO 2: Acute Physical Stress (High EDA + HR)
    # ─────────────────────────────────────────────────────────────────
    dummy_physical = df[feature_cols].dropna().iloc[0:1].copy()
    dummy_physical['theta_beta_ratio_TP9']    = 0.0     # Normal
    dummy_physical['muse_eeg_beta_TP9_mean']  = 0.2     # Normal 
    dummy_physical['wrist_eda_eda_mean']      = 1.8     # High EDA (z-score > 1)
    dummy_physical['wrist_hr_hr_mean']        = 1.5     # High HR (z-score > 1)

    evaluate_worker_state("P04 (Heavy Lifter)", dummy_physical,
                          model, scaler, imputer, feature_cols)


if __name__ == "__main__":
    main()
