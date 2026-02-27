"""
tests/test_inference.py
-----------------------
Unit tests for the inference pipeline:
  - Artifact loading (model, scaler, imputer, feature_cols)
  - Impute → Scale → Predict chain
  - Output validity (shape, probabilities sum to 1, label in {Low,Medium,High})
  - Graceful handling of missing / NaN features
  - Held-out evaluation sanity checks (better than random baseline)
"""

import pytest
import numpy as np
import pandas as pd
import json
import joblib
import xgboost as xgb
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR  = PROJECT_ROOT / "outputs" / "XGBoost-v2"

LABEL_MAP     = {0: "Low", 1: "Medium", 2: "High"}
LABEL_MAPPING = {"Low": 0, "Medium": 1, "High": 2}


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def artifacts():
    """Load all 4 deployment artifacts once per module."""
    model = xgb.XGBClassifier()
    model.load_model(str(OUTPUTS_DIR / "xgboost_fatigue_model.json"))
    scaler      = joblib.load(OUTPUTS_DIR / "scaler.pkl")
    imputer     = joblib.load(OUTPUTS_DIR / "imputer.pkl")
    feat_cols   = json.loads((OUTPUTS_DIR / "feature_cols.json").read_text())
    return model, scaler, imputer, feat_cols


@pytest.fixture(scope="module")
def sample_rows(artifacts):
    """
    Returns one real row per class from the processed CSV
    (training data, so guaranteed to have no NaN after imputation).
    """
    _, _, _, feat_cols = artifacts
    df = pd.read_csv(OUTPUTS_DIR / "processed_features_windowed.csv")
    rows = {}
    for label in ["Low", "Medium", "High"]:
        rows[label] = df[df["fatigue_level"] == label].dropna(subset=feat_cols).iloc[0:1][feat_cols]
    return rows


def predict(model, scaler, imputer, feat_cols, row_df):
    """Helper: impute → scale → predict (returns idx, proba)."""
    imp = pd.DataFrame(imputer.transform(row_df), columns=feat_cols)
    sc  = scaler.transform(imp)
    return model.predict(sc)[0], model.predict_proba(sc)[0]


# ─── Tests: Artifact Loading ──────────────────────────────────────────────────

class TestArtifactLoading:

    def test_model_loads(self):
        model = xgb.XGBClassifier()
        model.load_model(str(OUTPUTS_DIR / "xgboost_fatigue_model.json"))
        assert model is not None

    def test_scaler_loads(self):
        scaler = joblib.load(OUTPUTS_DIR / "scaler.pkl")
        assert hasattr(scaler, "transform")

    def test_imputer_loads(self):
        imputer = joblib.load(OUTPUTS_DIR / "imputer.pkl")
        assert hasattr(imputer, "transform")

    def test_feature_cols_loads(self):
        feat_cols = json.loads((OUTPUTS_DIR / "feature_cols.json").read_text())
        assert isinstance(feat_cols, list)
        assert len(feat_cols) == 153

    def test_feature_cols_no_duplicates(self):
        feat_cols = json.loads((OUTPUTS_DIR / "feature_cols.json").read_text())
        assert len(feat_cols) == len(set(feat_cols)), "Duplicate feature names in feature_cols.json"

    def test_scaler_fitted_on_153_features(self):
        scaler = joblib.load(OUTPUTS_DIR / "scaler.pkl")
        assert scaler.n_features_in_ == 153

    def test_imputer_fitted_on_153_features(self):
        imputer = joblib.load(OUTPUTS_DIR / "imputer.pkl")
        assert imputer.n_features_in_ == 153


# ─── Tests: Prediction Output Shape & Validity ────────────────────────────────

class TestPredictionOutput:

    def test_predict_returns_valid_label(self, artifacts, sample_rows):
        model, scaler, imputer, feat_cols = artifacts
        for label, row in sample_rows.items():
            pred_idx, _ = predict(model, scaler, imputer, feat_cols, row)
            assert LABEL_MAP[int(pred_idx)] in {"Low", "Medium", "High"}

    def test_proba_sums_to_one(self, artifacts, sample_rows):
        model, scaler, imputer, feat_cols = artifacts
        for label, row in sample_rows.items():
            _, proba = predict(model, scaler, imputer, feat_cols, row)
            assert abs(proba.sum() - 1.0) < 1e-5, \
                f"Probabilities for class {label} don't sum to 1: {proba.sum()}"

    def test_proba_shape_is_3(self, artifacts, sample_rows):
        model, scaler, imputer, feat_cols = artifacts
        for label, row in sample_rows.items():
            _, proba = predict(model, scaler, imputer, feat_cols, row)
            assert len(proba) == 3, f"Expected 3 class probs, got {len(proba)}"

    def test_all_proba_non_negative(self, artifacts, sample_rows):
        model, scaler, imputer, feat_cols = artifacts
        for label, row in sample_rows.items():
            _, proba = predict(model, scaler, imputer, feat_cols, row)
            assert (proba >= 0).all(), f"Negative probability found for class {label}"

    def test_proba_max_index_matches_predict(self, artifacts, sample_rows):
        """predict() must return argmax of predict_proba()."""
        model, scaler, imputer, feat_cols = artifacts
        for label, row in sample_rows.items():
            pred_idx, proba = predict(model, scaler, imputer, feat_cols, row)
            assert int(pred_idx) == int(np.argmax(proba)), \
                "predict() and argmax(predict_proba()) disagree"


# ─── Tests: Missing Feature Handling ─────────────────────────────────────────

class TestMissingFeatureHandling:

    def test_all_nan_row_still_predicts(self, artifacts):
        """A fully NaN row should not crash — imputer fills with medians."""
        model, scaler, imputer, feat_cols = artifacts
        all_nan = pd.DataFrame([{col: np.nan for col in feat_cols}])
        imp = pd.DataFrame(imputer.transform(all_nan), columns=feat_cols)
        sc  = scaler.transform(imp)
        pred = model.predict(sc)
        assert len(pred) == 1
        assert int(pred[0]) in {0, 1, 2}

    def test_half_nan_row_still_predicts(self, artifacts):
        """Row with 50% NaN features should still produce a valid prediction."""
        model, scaler, imputer, feat_cols = artifacts
        row = {col: np.nan for col in feat_cols}
        for i, col in enumerate(feat_cols):
            if i % 2 == 0:
                row[col] = 0.5
        row_df = pd.DataFrame([row])
        imp = pd.DataFrame(imputer.transform(row_df), columns=feat_cols)
        sc  = scaler.transform(imp)
        pred = model.predict(sc)
        assert int(pred[0]) in {0, 1, 2}

    def test_imputer_fills_nans(self, artifacts):
        """After imputation no NaNs should remain."""
        _, _, imputer, feat_cols = artifacts
        all_nan = pd.DataFrame([{col: np.nan for col in feat_cols}])
        imp = pd.DataFrame(imputer.transform(all_nan), columns=feat_cols)
        assert not imp.isna().any().any(), "Imputer left NaN values in output"

    def test_imputer_replaces_inf(self, artifacts):
        """Infinite values must be handled before imputation (replace then impute)."""
        _, scaler, imputer, feat_cols = artifacts
        row = {col: np.inf for col in feat_cols}
        row_df = pd.DataFrame([row])
        row_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        imp = pd.DataFrame(imputer.transform(row_df), columns=feat_cols)
        assert not np.isinf(imp.values).any()


# ─── Tests: Held-Out Evaluation Sanity ───────────────────────────────────────

class TestHeldOutEvaluation:
    """
    High-level sanity checks on the held-out test set (participants 10–12).
    These are not strict assertions on exact accuracy — they verify the
    model performs meaningfully better than random chance.
    """
    RANDOM_BASELINE = 1 / 3   # 33.3% for 3-class problem

    @pytest.fixture(scope="class")
    def holdout_results(self, artifacts):
        model, scaler, imputer, feat_cols = artifacts
        df = pd.read_csv(OUTPUTS_DIR / "unseen_test_data.csv")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = df[feat_cols]
        y = df["fatigue_level"].map(LABEL_MAPPING)
        imp = pd.DataFrame(imputer.transform(X), columns=feat_cols)
        sc  = scaler.transform(imp)
        y_pred  = model.predict(sc)
        y_proba = model.predict_proba(sc)
        return y.values, y_pred, y_proba

    def test_accuracy_above_random(self, holdout_results):
        """Held-out accuracy must beat the 33.3% random baseline."""
        y_true, y_pred, _ = holdout_results
        acc = (y_true == y_pred).mean()
        assert acc > self.RANDOM_BASELINE, \
            f"Accuracy {acc:.3f} is not above random baseline {self.RANDOM_BASELINE:.3f}"

    def test_prediction_covers_all_classes(self, holdout_results):
        """Model must predict all 3 classes on 200 held-out rows (not degenerate)."""
        _, y_pred, _ = holdout_results
        assert set(y_pred.tolist()) == {0, 1, 2}, \
            f"Model only predicted classes: {set(y_pred.tolist())}"

    def test_proba_rows_sum_to_one(self, holdout_results):
        """Every row's probabilities must sum to 1."""
        _, _, y_proba = holdout_results
        row_sums = y_proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5,
            err_msg="Some probability rows don't sum to 1")

    def test_low_class_recall_above_threshold(self, holdout_results):
        """Low fatigue should have recall > 0.4 (model showed avg 0.70 for Low)."""
        y_true, y_pred, _ = holdout_results
        low_mask = (y_true == LABEL_MAPPING["Low"])
        recall = (y_pred[low_mask] == LABEL_MAPPING["Low"]).mean()
        assert recall > 0.40, f"Low class recall {recall:.3f} is unexpectedly poor"

    def test_high_class_recall_above_threshold(self, holdout_results):
        """High fatigue should have recall > 0.35 (critical safety class)."""
        y_true, y_pred, _ = holdout_results
        high_mask = (y_true == LABEL_MAPPING["High"])
        recall = (y_pred[high_mask] == LABEL_MAPPING["High"]).mean()
        assert recall > 0.35, f"High class recall {recall:.3f} is unexpectedly poor"
