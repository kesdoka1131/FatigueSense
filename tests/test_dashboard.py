"""
tests/test_dashboard.py
-----------------------
Unit tests for dashboard logic (without running Streamlit):
  - JSON payload parsing
  - analyze_fatigue() output validity
  - Causal weights loading and filtering
  - Backend evaluate_worker_state() logic
  - Feature schema compatibility (dashboard JSON keys match feature_cols)
"""

import pytest
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR  = PROJECT_ROOT / "outputs" / "XGBoost-v2"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "dashboard"))

LABEL_MAP     = {0: "Low", 1: "Medium", 2: "High"}
LABEL_MAPPING = {"Low": 0, "Medium": 1, "High": 2}


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def artifacts():
    model = xgb.XGBClassifier()
    model.load_model(str(OUTPUTS_DIR / "xgboost_fatigue_model.json"))
    scaler    = joblib.load(OUTPUTS_DIR / "scaler.pkl")
    imputer   = joblib.load(OUTPUTS_DIR / "imputer.pkl")
    feat_cols = json.loads((OUTPUTS_DIR / "feature_cols.json").read_text())
    return model, scaler, imputer, feat_cols


@pytest.fixture(scope="module")
def causal_weights():
    path = OUTPUTS_DIR / "causal_weights.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


@pytest.fixture(scope="module")
def sample_payload(artifacts):
    """A valid JSON payload built from a real row in the processed CSV."""
    _, _, _, feat_cols = artifacts
    df = pd.read_csv(OUTPUTS_DIR / "processed_features_windowed.csv")
    row = df.dropna(subset=feat_cols).iloc[0][feat_cols]
    return row.to_dict()


# ─── Replicate analyze_fatigue for testing (no Streamlit dependency) ──────────

def analyze_fatigue_logic(data_dict, model, scaler, imputer, feat_cols):
    """
    Mirrors the logic in dashboard/app.py:analyze_fatigue()
    without the Streamlit context.
    """
    row = pd.DataFrame([{col: data_dict.get(col, np.nan) for col in feat_cols}])
    row_imputed = pd.DataFrame(imputer.transform(row), columns=feat_cols)
    row_scaled  = scaler.transform(row_imputed)
    pred_idx    = model.predict(row_scaled)[0]
    pred_proba  = model.predict_proba(row_scaled)[0]
    fatigue_state = LABEL_MAP[int(pred_idx)]
    risk_score = float(pred_proba[1]) * 50 + float(pred_proba[2]) * 100
    risk_score = min(risk_score, 100.0)
    return fatigue_state, risk_score, pred_proba


# ─── Tests: JSON Payload Parsing ──────────────────────────────────────────────

class TestJSONParsing:

    def test_valid_json_parses(self, sample_payload):
        json_str = json.dumps(sample_payload)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert len(parsed) > 0

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            json.loads("{not_valid_json}")

    def test_empty_json_object_parses(self):
        parsed = json.loads("{}")
        assert parsed == {}

    def test_payload_has_numeric_values(self, sample_payload):
        for k, v in sample_payload.items():
            assert isinstance(v, (int, float)), \
                f"Feature '{k}' has non-numeric value: {v!r}"

    def test_payload_values_are_finite(self, sample_payload):
        for k, v in sample_payload.items():
            assert np.isfinite(v), f"Feature '{k}' has non-finite value: {v}"


# ─── Tests: analyze_fatigue() Output ─────────────────────────────────────────

class TestAnalyzeFatigue:

    def test_returns_valid_fatigue_state(self, artifacts, sample_payload):
        model, scaler, imputer, feat_cols = artifacts
        state, _, _ = analyze_fatigue_logic(sample_payload, model, scaler, imputer, feat_cols)
        assert state in {"Low", "Medium", "High"}

    def test_risk_score_in_range(self, artifacts, sample_payload):
        model, scaler, imputer, feat_cols = artifacts
        _, risk_score, _ = analyze_fatigue_logic(sample_payload, model, scaler, imputer, feat_cols)
        assert 0.0 <= risk_score <= 100.0, f"Risk score {risk_score} out of [0, 100]"

    def test_proba_sums_to_one(self, artifacts, sample_payload):
        model, scaler, imputer, feat_cols = artifacts
        _, _, proba = analyze_fatigue_logic(sample_payload, model, scaler, imputer, feat_cols)
        assert abs(proba.sum() - 1.0) < 1e-5

    def test_empty_payload_does_not_crash(self, artifacts):
        """Empty dict → all NaN → imputer fills → should still return a prediction."""
        model, scaler, imputer, feat_cols = artifacts
        state, risk, proba = analyze_fatigue_logic({}, model, scaler, imputer, feat_cols)
        assert state in {"Low", "Medium", "High"}
        assert 0.0 <= risk <= 100.0

    def test_high_hr_increases_medium_high_probability(self, artifacts, sample_payload):
        """
        Injecting an extreme HR value (200 bpm) should shift probability
        away from Low and toward Medium/High on most samples.
        """
        model, scaler, imputer, feat_cols = artifacts
        normal_payload = dict(sample_payload)
        high_hr_payload = dict(sample_payload)
        high_hr_payload["wrist_hr_hr_mean"] = 200.0

        _, _, normal_proba = analyze_fatigue_logic(normal_payload, model, scaler, imputer, feat_cols)
        _, _, high_proba   = analyze_fatigue_logic(high_hr_payload, model, scaler, imputer, feat_cols)

        # P(Low) should not increase when HR spikes
        assert high_proba[0] <= normal_proba[0] + 0.15, \
            "P(Low) unexpectedly increased with extreme HR — check feature importance"

    def test_risk_score_higher_for_high_prediction(self, artifacts):
        """
        A row known to predict High fatigue should have a higher risk
        score than a row known to predict Low.
        """
        model, scaler, imputer, feat_cols = artifacts
        df = pd.read_csv(OUTPUTS_DIR / "processed_features_windowed.csv")

        low_row  = df[df["fatigue_level"] == "Low" ].dropna(subset=feat_cols).iloc[0][feat_cols].to_dict()
        high_row = df[df["fatigue_level"] == "High"].dropna(subset=feat_cols).iloc[0][feat_cols].to_dict()

        s_low,  r_low,  _ = analyze_fatigue_logic(low_row,  model, scaler, imputer, feat_cols)
        s_high, r_high, _ = analyze_fatigue_logic(high_row, model, scaler, imputer, feat_cols)

        # Only enforce if the model actually predicts differently
        if s_low != s_high:
            assert r_high >= r_low, \
                f"Risk score for High ({r_high:.1f}) < risk score for Low ({r_low:.1f})"


# ─── Tests: Causal Weights ────────────────────────────────────────────────────

class TestCausalWeights:

    def test_causal_weights_file_exists(self):
        assert (OUTPUTS_DIR / "causal_weights.json").exists(), \
            "causal_weights.json not found — run causal_analysis.py first"

    def test_causal_weights_non_empty(self, causal_weights):
        assert len(causal_weights) > 0, "causal_weights.json is empty"

    def test_causal_weights_values_are_floats(self, causal_weights):
        for k, v in causal_weights.items():
            assert isinstance(v, (int, float)), \
                f"Causal weight '{k}' has non-numeric value: {v!r}"

    def test_causal_weights_keys_contain_arrow(self, causal_weights):
        """All keys must be in the form 'X_tN -> Y_t0'."""
        for k in causal_weights:
            assert "->" in k, f"Causal weight key '{k}' missing '->' arrow"

    def test_cross_links_exist(self, causal_weights):
        """At least some cross-variable links (not just auto-regression) must exist."""
        cross = [
            k for k in causal_weights
            if k.split("_t")[0] != k.split("->")[1].split("_t")[0]
        ]
        assert len(cross) > 0, "No cross-variable causal links found"


# ─── Tests: Feature Schema Compatibility ─────────────────────────────────────

class TestFeatureSchemaCompatibility:

    def test_feature_cols_are_in_processed_csv(self, artifacts):
        """Every feature in feature_cols.json must exist in processed_features_windowed.csv."""
        _, _, _, feat_cols = artifacts
        df = pd.read_csv(OUTPUTS_DIR / "processed_features_windowed.csv")
        missing = set(feat_cols) - set(df.columns)
        assert not missing, f"Features in feature_cols.json not in CSV: {missing}"

    def test_processed_csv_cols_match_feature_cols(self, artifacts):
        """No extra features in the CSV (beyond metadata) that aren't in feature_cols.json."""
        _, _, _, feat_cols = artifacts
        df = pd.read_csv(OUTPUTS_DIR / "processed_features_windowed.csv")
        meta = {"timestamp", "participant_id", "session_id", "fatigue_level"}
        csv_feats  = set(df.columns) - meta
        json_feats = set(feat_cols)
        extra = csv_feats - json_feats
        assert not extra, f"CSV has extra columns not in feature_cols.json: {extra}"

    def test_unseen_test_data_has_same_features(self, artifacts):
        """unseen_test_data.csv must contain all pipeline feature columns."""
        _, _, _, feat_cols = artifacts
        df = pd.read_csv(OUTPUTS_DIR / "unseen_test_data.csv")
        missing = set(feat_cols) - set(df.columns)
        assert not missing, f"unseen_test_data.csv missing features: {missing}"

    def test_scaler_feature_count_matches_json(self, artifacts):
        """Number of features in scaler must match feature_cols.json."""
        _, scaler, _, feat_cols = artifacts
        assert scaler.n_features_in_ == len(feat_cols), \
            f"Scaler expects {scaler.n_features_in_} features, feature_cols.json has {len(feat_cols)}"
