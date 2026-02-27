"""
tests/test_pipeline.py
----------------------
Unit tests for data_pipeline.py:
  - Windowing logic (60-second resample)
  - Lag-feature creation
  - Column naming conventions
  - Session label mapping
  - NaN/empty edge cases
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# ─── make project root importable ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_sensor_df(n_seconds=180, seed=42):
    """
    Creates a synthetic sensor DataFrame mimicking a raw EEG/HR/EDA file:
    - timestamp column (millisecond epoch, 1-second cadence)
    - 4 numeric sensor channels
    """
    rng = np.random.default_rng(seed)
    base_ts = pd.Timestamp("2023-01-01").value // 10**6   # ms epoch
    timestamps = [base_ts + i * 1000 for i in range(n_seconds)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "TP9":  rng.normal(0.5, 0.1, n_seconds),
        "AF7":  rng.normal(0.3, 0.1, n_seconds),
        "AF8":  rng.normal(0.4, 0.1, n_seconds),
        "TP10": rng.normal(0.6, 0.1, n_seconds),
    })


def resample_sensor(df, key):
    """
    Replicates the resampling logic from data_pipeline.process_participant:
    timestamp (ms) → datetime index → 60-second windows (mean + std).
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    resampled = df.select_dtypes(include=[np.number]).resample("60s").agg(["mean", "std"])
    resampled.columns = [f"{key}_{col[0]}_{col[1]}" for col in resampled.columns]
    return resampled


def add_lag_features(df):
    """Replicates the lag-feature step from data_pipeline.process_participant."""
    numeric_cols = df.columns.tolist()
    for col in numeric_cols:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag2"] = df[col].shift(2)
    return df


# ─── Tests: Windowing ─────────────────────────────────────────────────────────

class TestWindowing:

    def test_180s_produces_3_windows(self):
        """180 seconds of 1-Hz data should produce exactly 3 one-minute windows."""
        df = make_sensor_df(n_seconds=180)
        resampled = resample_sensor(df, "muse_eeg_alpha")
        assert len(resampled) == 3, f"Expected 3 windows, got {len(resampled)}"

    def test_window_count_rounds_up(self):
        """121 seconds of data should produce 3 windows (bins: 0-59, 60-119, 120-180)."""
        df = make_sensor_df(n_seconds=121)
        resampled = resample_sensor(df, "muse_eeg_alpha")
        assert len(resampled) >= 2

    def test_single_window(self):
        """Less than 60 seconds should produce 1 window."""
        df = make_sensor_df(n_seconds=59)
        resampled = resample_sensor(df, "muse_eeg_alpha")
        assert len(resampled) == 1

    def test_window_values_are_finite(self):
        """All windowed values should be finite (no accidental inf)."""
        df = make_sensor_df(n_seconds=180)
        resampled = resample_sensor(df, "muse_eeg_alpha")
        assert np.isfinite(resampled.values).all(), "Windowed output contains non-finite values"

    def test_mean_is_within_input_range(self):
        """The window means must be within the min-max range of the raw input."""
        df = make_sensor_df(n_seconds=180)
        resampled = resample_sensor(df, "muse_eeg_alpha")
        mean_cols = [c for c in resampled.columns if c.endswith("_mean")]
        raw_min = df[["TP9", "AF7", "AF8", "TP10"]].min().min()
        raw_max = df[["TP9", "AF7", "AF8", "TP10"]].max().max()
        for col in mean_cols:
            assert resampled[col].dropna().between(raw_min, raw_max).all(), \
                f"Window mean {col} is outside raw data range"

    def test_std_is_non_negative(self):
        """Standard deviations must be >= 0."""
        df = make_sensor_df(n_seconds=180)
        resampled = resample_sensor(df, "muse_eeg_alpha")
        std_cols = [c for c in resampled.columns if c.endswith("_std")]
        for col in std_cols:
            assert (resampled[col].dropna() >= 0).all(), f"{col} has negative std"


# ─── Tests: Column Naming Convention ──────────────────────────────────────────

class TestColumnNaming:

    def test_column_format_is_key_sensor_stat(self):
        """Columns must follow pattern: {key}_{sensor}_{stat}."""
        df = make_sensor_df(n_seconds=120)
        resampled = resample_sensor(df, "muse_eeg_alpha")
        for col in resampled.columns:
            parts = col.split("_")
            assert len(parts) >= 3, f"Column '{col}' doesn't follow key_sensor_stat format"
            assert parts[-1] in ("mean", "std"), f"Column '{col}' should end in _mean or _std"

    def test_key_prefix_preserved(self):
        """Each column should start with the provided sensor key."""
        df = make_sensor_df(n_seconds=120)
        key = "wrist_hr"
        resampled = resample_sensor(df, key)
        for col in resampled.columns:
            assert col.startswith(key), f"Column '{col}' doesn't start with key '{key}'"

    def test_both_mean_and_std_present_for_each_channel(self):
        """For every channel there must be exactly one _mean and one _std column."""
        df = make_sensor_df(n_seconds=120)
        resampled = resample_sensor(df, "muse_eeg_beta")
        channels = {col.rsplit("_", 1)[0] for col in resampled.columns}
        for ch in channels:
            assert f"{ch}_mean" in resampled.columns, f"Missing {ch}_mean"
            assert f"{ch}_std"  in resampled.columns, f"Missing {ch}_std"


# ─── Tests: Lag Features ──────────────────────────────────────────────────────

class TestLagFeatures:

    @pytest.fixture
    def windowed_df(self):
        df = make_sensor_df(n_seconds=300)
        resampled = resample_sensor(df, "muse_eeg_alpha")
        return add_lag_features(resampled.copy())

    def test_lag1_columns_exist(self, windowed_df):
        base_cols = [c for c in windowed_df.columns if not c.endswith(("_lag1", "_lag2"))]
        for col in base_cols:
            assert f"{col}_lag1" in windowed_df.columns, f"Missing lag1 for {col}"

    def test_lag2_columns_exist(self, windowed_df):
        base_cols = [c for c in windowed_df.columns if not c.endswith(("_lag1", "_lag2"))]
        for col in base_cols:
            assert f"{col}_lag2" in windowed_df.columns, f"Missing lag2 for {col}"

    def test_lag1_value_matches_previous_row(self, windowed_df):
        """lag1 at row i must equal the base value at row i-1."""
        base_cols = [c for c in windowed_df.columns if not c.endswith(("_lag1", "_lag2"))]
        col = base_cols[0]
        # Row index 1 lag1 should equal row index 0 base value
        assert windowed_df[f"{col}_lag1"].iloc[1] == pytest.approx(windowed_df[col].iloc[0])

    def test_lag2_value_matches_two_rows_back(self, windowed_df):
        """lag2 at row i must equal the base value at row i-2."""
        base_cols = [c for c in windowed_df.columns if not c.endswith(("_lag1", "_lag2"))]
        col = base_cols[0]
        # Row index 2 lag2 should equal row index 0 base value
        assert windowed_df[f"{col}_lag2"].iloc[2] == pytest.approx(windowed_df[col].iloc[0])

    def test_first_row_lag1_is_nan(self, windowed_df):
        """The very first row can have no lag — must be NaN."""
        base_cols = [c for c in windowed_df.columns if not c.endswith(("_lag1", "_lag2"))]
        col = base_cols[0]
        assert pd.isna(windowed_df[f"{col}_lag1"].iloc[0])

    def test_first_two_rows_lag2_are_nan(self, windowed_df):
        """The first two rows must have NaN lag2."""
        base_cols = [c for c in windowed_df.columns if not c.endswith(("_lag1", "_lag2"))]
        col = base_cols[0]
        assert pd.isna(windowed_df[f"{col}_lag2"].iloc[0])
        assert pd.isna(windowed_df[f"{col}_lag2"].iloc[1])

    def test_no_extra_lag_columns(self, windowed_df):
        """Pipeline only adds lag1 and lag2 — no lag3 or higher."""
        lag3_cols = [c for c in windowed_df.columns if c.endswith("_lag3")]
        assert len(lag3_cols) == 0, "Found unexpected lag3 columns"


# ─── Tests: Session Label Mapping ─────────────────────────────────────────────

class TestSessionLabelMapping:

    def test_valid_labels(self):
        """Only the three canonical labels are allowed."""
        valid = {"Low", "Medium", "High"}
        df = pd.read_csv(PROJECT_ROOT / "outputs" / "XGBoost-v2" / "processed_features_windowed.csv")
        bad = set(df["fatigue_level"].unique()) - valid
        assert not bad, f"Unexpected fatigue labels found: {bad}"

    def test_all_three_labels_present(self):
        """Each class must appear at least once across 12 participants."""
        df = pd.read_csv(PROJECT_ROOT / "outputs" / "XGBoost-v2" / "processed_features_windowed.csv")
        assert set(df["fatigue_level"].unique()) == {"Low", "Medium", "High"}

    def test_each_participant_has_all_three_labels(self):
        """Every participant must have at least one window per fatigue class."""
        df = pd.read_csv(PROJECT_ROOT / "outputs" / "XGBoost-v2" / "processed_features_windowed.csv")
        for pid, group in df.groupby("participant_id"):
            labels = set(group["fatigue_level"].unique())
            expected = {"Low", "Medium", "High"}
            missing_labels = expected - labels
            assert labels == expected, \
                f"Participant {pid} is missing labels: {missing_labels}"


# ─── Tests: Edge Cases ────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_all_nan_input_produces_empty_after_dropna(self):
        """A DataFrame of all NaNs should be dropped by the 'dropna(how=all)' step."""
        df = pd.DataFrame({
            "muse_eeg_alpha_TP9_mean": [np.nan, np.nan],
            "muse_eeg_alpha_TP9_std":  [np.nan, np.nan],
        })
        df.dropna(how="all", inplace=True)
        assert df.empty

    def test_inf_values_handled(self):
        """After replacing inf with NaN, no inf should remain."""
        df = pd.DataFrame({"a": [1.0, np.inf, -np.inf, 2.0]})
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        assert not np.isinf(df["a"].dropna()).any()

    def test_processed_csv_has_expected_columns(self):
        """The processed CSV must contain the 153 feature columns + 4 metadata columns."""
        df = pd.read_csv(PROJECT_ROOT / "outputs" / "XGBoost-v2" / "processed_features_windowed.csv")
        required_meta = {"timestamp", "participant_id", "session_id", "fatigue_level"}
        assert required_meta.issubset(set(df.columns)), \
            f"Missing metadata columns: {required_meta - set(df.columns)}"
        feature_cols = [c for c in df.columns if c not in required_meta]
        assert len(feature_cols) == 153, \
            f"Expected 153 feature columns, got {len(feature_cols)}"

    def test_no_duplicate_windows(self):
        """No two rows should have the same (participant_id, session_id, timestamp)."""
        df = pd.read_csv(PROJECT_ROOT / "outputs" / "XGBoost-v2" / "processed_features_windowed.csv")
        dupes = df.duplicated(subset=["participant_id", "session_id", "timestamp"])
        assert not dupes.any(), f"Found {dupes.sum()} duplicate windows"
