import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import plotly.graph_objects as go
import json
import os

st.set_page_config(page_title="Factory Safety HR Dashboard", layout="wide")

st.markdown("""
<style>
    /* ── Component styles ── */
    .big-font { font-size: 24px !important; font-weight: bold; }
    .alert-card { padding: 10px 14px; border-radius: 8px; margin-bottom: 10px; color: #000000; }
    .alert-high { background-color: #ffcccc; border-left: 5px solid red; }
    .alert-med  { background-color: #ffe6cc; border-left: 5px solid orange; }
    .alert-low  { background-color: #e6ffcc; border-left: 5px solid green; }
    .intervention { background-color: #e6f2ff; padding: 10px 14px; border-radius: 5px; font-weight: bold; color: #000000; }

    /* ── Global layout tightening ── */
    /* Reduce the huge default top padding on the main area */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0.2rem !important;
    }
    /* Tighten the page title and subtitle */
    h1 { margin-top: 0.2rem !important; margin-bottom: 0.1rem !important; font-size: 1.8rem !important; }
    [data-testid="stMarkdownContainer"] > p { margin-top: 0 !important; margin-bottom: 0.2rem !important; }
    /* Shrink the gap between every Streamlit widget/block element */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockWrapper"] {
        gap: 0.3rem !important;
    }
    /* Tighten vertical spacing inside st.metric widgets */
    [data-testid="stMetric"] {
        padding: 4px 0 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        margin-bottom: 0 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
    }
    /* Reduce h3 heading margin (### sections) */
    h3 { margin-top: 0.4rem !important; margin-bottom: 0.2rem !important; }
    /* Reduce caption spacing */
    [data-testid="stCaptionContainer"] { margin-top: -4px !important; margin-bottom: 2px !important; }
    /* Tighten dataframe wrapper */
    [data-testid="stDataFrame"] { margin-top: 0 !important; margin-bottom: 0 !important; }
    /* Hide Streamlit's top header bar and toolbar */
    [data-testid="stHeader"]   { display: none !important; }
    [data-testid="stToolbar"]  { display: none !important; }
    #MainMenu                  { display: none !important; }
    /* ── Sidebar tightening ── */
    /* Reduce sidebar top padding */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0rem !important;
    }
    /* Gaps between sidebar widget blocks */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockWrapper"] {
        gap: 0.2rem !important;
    }
    /* Collapse hr divider margins in sidebar */
    [data-testid="stSidebar"] hr {
        margin-top: 0.4rem !important;
        margin-bottom: 0.4rem !important;
    }
    /* Tighten info/success alert boxes in sidebar */
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        padding: 6px 10px !important;
    }
    /* Sidebar header spacing */
    [data-testid="stSidebar"] h2 {
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
    }
    [data-testid="stSidebar"] h3 {
        margin-top: 0.2rem !important;
        margin-bottom: 0.1rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏭 Factory Safety: Live Worker Telemetry")
st.write("Input the raw telemetry payload (1-minute window) to see the causal AI analysis & HR prescription.")

# ──────────────────────────────────────────────────────────────────────────────
# Load deployment artifacts from outputs/
# ──────────────────────────────────────────────────────────────────────────────
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'XGBoost-v2')

@st.cache_resource
def load_artifacts():
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(OUTPUTS_DIR, 'xgboost_fatigue_model.json'))
    scaler  = joblib.load(os.path.join(OUTPUTS_DIR, 'scaler.pkl'))
    imputer = joblib.load(os.path.join(OUTPUTS_DIR, 'imputer.pkl'))
    with open(os.path.join(OUTPUTS_DIR, 'feature_cols.json')) as f:
        feature_cols = json.load(f)
    return model, scaler, imputer, feature_cols

try:
    model, scaler, imputer, feature_cols = load_artifacts()
    artifacts_ok = True
except Exception as e:
    st.error(f"⚠️ Could not load model artifacts from `outputs/`. "
             f"Please run `train_model.py` first.\n\nError: {e}")
    artifacts_ok = False
    st.stop()

@st.cache_data
def load_holdout_data():
    """Cache the held-out test set so the button is instant."""
    path = os.path.join(OUTPUTS_DIR, 'unseen_test_data.csv')
    df = pd.read_csv(path)
    df = df.replace([float('inf'), float('-inf')], float('nan'))
    return df

label_map = {0: 'Low', 1: 'Medium', 2: 'High'}

def analyze_fatigue(data_dict):
    """
    Real inference using the trained XGBoost v2 model.
    Expects a dict with the pipeline feature keys.
    Returns fatigue_state, risk_score (0-100), and confidence probabilities.
    """
    # Build feature row — fill missing keys with NaN (imputer will handle them)
    row = pd.DataFrame([{col: data_dict.get(col, np.nan) for col in feature_cols}])

    # Impute → Scale → Predict
    row_imputed = pd.DataFrame(imputer.transform(row), columns=feature_cols)
    row_scaled  = scaler.transform(row_imputed)

    pred_idx   = model.predict(row_scaled)[0]
    pred_proba = model.predict_proba(row_scaled)[0]   # [P(Low), P(Medium), P(High)]

    fatigue_state = label_map[int(pred_idx)]

    # Map prediction confidence to a 0-100 risk score
    # Low confidence → stays near 0; Medium confidence → ~50; High confidence → ~100
    risk_score = float(pred_proba[1]) * 50 + float(pred_proba[2]) * 100
    risk_score = min(risk_score, 100.0)

    return fatigue_state, risk_score, pred_proba

# ──────────────────────────────────────────────────────────────────────────────
# Default JSON — a real sample row from the v2 pipeline (Medium fatigue, z-scored)
# Note: values are per-participant z-scored (0 = participant's mean, 1 = 1 std above)
# ──────────────────────────────────────────────────────────────────────────────
default_json = '''{
  "muse_eeg_alpha_TP9_mean": -0.4505, "muse_eeg_alpha_TP9_std": -0.2057,
  "muse_eeg_alpha_AF7_mean": -0.7203, "muse_eeg_alpha_AF7_std": -0.562,
  "muse_eeg_alpha_AF8_mean": -0.4282, "muse_eeg_alpha_AF8_std": -0.1347,
  "muse_eeg_alpha_TP10_mean": -0.2802, "muse_eeg_alpha_TP10_std": 0.4111,
  "muse_eeg_beta_TP9_mean": 0.1717, "muse_eeg_beta_TP9_std": -0.26,
  "muse_eeg_beta_AF7_mean": -0.4703, "muse_eeg_beta_AF7_std": -0.4441,
  "muse_eeg_beta_AF8_mean": -0.0641, "muse_eeg_beta_AF8_std": -0.544,
  "muse_eeg_beta_TP10_mean": -0.2254, "muse_eeg_beta_TP10_std": -0.3081,
  "muse_eeg_theta_TP9_mean": -0.8463, "muse_eeg_theta_TP9_std": 0.0493,
  "muse_eeg_theta_AF7_mean": -0.9615, "muse_eeg_theta_AF7_std": -0.6654,
  "muse_eeg_theta_AF8_mean": -0.8427, "muse_eeg_theta_AF8_std": 0.217,
  "muse_eeg_theta_TP10_mean": -0.4876, "muse_eeg_theta_TP10_std": 0.6252,
  "muse_eeg_delta_TP9_mean": -0.828, "muse_eeg_delta_TP9_std": -0.2419,
  "muse_eeg_delta_AF7_mean": -1.1568, "muse_eeg_delta_AF7_std": -0.1131,
  "muse_eeg_delta_AF8_mean": -0.9179, "muse_eeg_delta_AF8_std": 0.1025,
  "muse_eeg_delta_TP10_mean": -0.7268, "muse_eeg_delta_TP10_std": 0.4653,
  "muse_eeg_gamma_TP9_mean": 1.1289, "muse_eeg_gamma_TP9_std": 1.3622,
  "muse_eeg_gamma_AF7_mean": -0.0242, "muse_eeg_gamma_AF7_std": -0.3731,
  "muse_eeg_gamma_AF8_mean": 0.9602, "muse_eeg_gamma_AF8_std": 0.6628,
  "muse_eeg_gamma_TP10_mean": -0.1205, "muse_eeg_gamma_TP10_std": -0.1352,
  "wrist_hr_hr_mean": -0.8727, "wrist_hr_hr_std": -0.5818,
  "wrist_eda_eda_mean": -1.2197, "wrist_eda_eda_std": -0.1933,
  "wrist_temp_temp_mean": 0.0259, "wrist_temp_temp_std": 0.9228,
  "wrist_bvp_bvp_mean": -0.2412, "wrist_bvp_bvp_std": 0.1241,
  "alpha_beta_ratio_TP9": -0.8167, "theta_beta_ratio_TP9": -1.5564,
  "eda_hr_interaction": -1.1164,
  "muse_eeg_alpha_TP9_mean_lag1": -0.3918, "muse_eeg_alpha_TP9_mean_lag2": 0.1078,
  "muse_eeg_alpha_TP9_std_lag1": -0.2649, "muse_eeg_alpha_TP9_std_lag2": -0.0002,
  "muse_eeg_alpha_AF7_mean_lag1": -0.3019, "muse_eeg_alpha_AF7_mean_lag2": -0.5438,
  "muse_eeg_alpha_AF7_std_lag1": 0.1391, "muse_eeg_alpha_AF7_std_lag2": -0.1131,
  "muse_eeg_alpha_AF8_mean_lag1": -0.0489, "muse_eeg_alpha_AF8_mean_lag2": -0.3854,
  "muse_eeg_alpha_AF8_std_lag1": -0.0266, "muse_eeg_alpha_AF8_std_lag2": -0.3416,
  "muse_eeg_alpha_TP10_mean_lag1": -0.6913, "muse_eeg_alpha_TP10_mean_lag2": 0.7281,
  "muse_eeg_alpha_TP10_std_lag1": -0.2151, "muse_eeg_alpha_TP10_std_lag2": 0.0357,
  "muse_eeg_beta_TP9_mean_lag1": 0.0962, "muse_eeg_beta_TP9_mean_lag2": -0.2892,
  "muse_eeg_beta_TP9_std_lag1": -0.1831, "muse_eeg_beta_TP9_std_lag2": 0.2104,
  "muse_eeg_beta_AF7_mean_lag1": -0.7637, "muse_eeg_beta_AF7_mean_lag2": 0.0374,
  "muse_eeg_beta_AF7_std_lag1": 0.0591, "muse_eeg_beta_AF7_std_lag2": -0.2723,
  "muse_eeg_beta_AF8_mean_lag1": -0.332, "muse_eeg_beta_AF8_mean_lag2": 0.6423,
  "muse_eeg_beta_AF8_std_lag1": -0.0033, "muse_eeg_beta_AF8_std_lag2": -0.6374,
  "muse_eeg_beta_TP10_mean_lag1": -0.7637, "muse_eeg_beta_TP10_mean_lag2": 0.111,
  "muse_eeg_beta_TP10_std_lag1": 0.407, "muse_eeg_beta_TP10_std_lag2": 0.0127,
  "muse_eeg_theta_TP9_mean_lag1": -0.7311, "muse_eeg_theta_TP9_mean_lag2": -0.5482,
  "muse_eeg_theta_TP9_std_lag1": 0.3289, "muse_eeg_theta_TP9_std_lag2": 0.8282,
  "muse_eeg_theta_AF7_mean_lag1": -0.6538, "muse_eeg_theta_AF7_mean_lag2": -0.8511,
  "muse_eeg_theta_AF7_std_lag1": 0.6388, "muse_eeg_theta_AF7_std_lag2": -0.1646,
  "muse_eeg_theta_AF8_mean_lag1": -0.3373, "muse_eeg_theta_AF8_mean_lag2": -0.7303,
  "muse_eeg_theta_AF8_std_lag1": 0.7204, "muse_eeg_theta_AF8_std_lag2": -0.4279,
  "muse_eeg_theta_TP10_mean_lag1": -0.8072, "muse_eeg_theta_TP10_mean_lag2": -0.379,
  "muse_eeg_theta_TP10_std_lag1": 0.1338, "muse_eeg_theta_TP10_std_lag2": 0.2716,
  "muse_eeg_delta_TP9_mean_lag1": -0.8223, "muse_eeg_delta_TP9_mean_lag2": -0.7559,
  "muse_eeg_delta_TP9_std_lag1": 0.2374, "muse_eeg_delta_TP9_std_lag2": 0.3634,
  "muse_eeg_delta_AF7_mean_lag1": -1.1567, "muse_eeg_delta_AF7_mean_lag2": -0.8903,
  "muse_eeg_delta_AF7_std_lag1": 1.0691, "muse_eeg_delta_AF7_std_lag2": 0.035,
  "muse_eeg_delta_AF8_mean_lag1": -0.5159, "muse_eeg_delta_AF8_mean_lag2": -0.5638,
  "muse_eeg_delta_AF8_std_lag1": 1.0293, "muse_eeg_delta_AF8_std_lag2": -0.4084,
  "muse_eeg_delta_TP10_mean_lag1": -0.5894, "muse_eeg_delta_TP10_mean_lag2": -0.3213,
  "muse_eeg_delta_TP10_std_lag1": 0.1642, "muse_eeg_delta_TP10_std_lag2": 0.1674,
  "muse_eeg_gamma_TP9_mean_lag1": 1.589, "muse_eeg_gamma_TP9_mean_lag2": 0.6825,
  "muse_eeg_gamma_TP9_std_lag1": -0.8485, "muse_eeg_gamma_TP9_std_lag2": 1.2305,
  "muse_eeg_gamma_AF7_mean_lag1": -0.8097, "muse_eeg_gamma_AF7_mean_lag2": 1.4403,
  "muse_eeg_gamma_AF7_std_lag1": -0.2686, "muse_eeg_gamma_AF7_std_lag2": -0.1736,
  "muse_eeg_gamma_AF8_mean_lag1": -0.0357, "muse_eeg_gamma_AF8_mean_lag2": 2.1793,
  "muse_eeg_gamma_AF8_std_lag1": -0.3266, "muse_eeg_gamma_AF8_std_lag2": 0.0031,
  "muse_eeg_gamma_TP10_mean_lag1": -0.1366, "muse_eeg_gamma_TP10_mean_lag2": 0.3947,
  "muse_eeg_gamma_TP10_std_lag1": -0.3763, "muse_eeg_gamma_TP10_std_lag2": -0.079,
  "wrist_hr_hr_mean_lag1": -0.8857, "wrist_hr_hr_mean_lag2": -1.0968,
  "wrist_hr_hr_std_lag1": -0.5412, "wrist_hr_hr_std_lag2": -0.5361,
  "wrist_eda_eda_mean_lag1": -0.9554, "wrist_eda_eda_mean_lag2": -1.0143,
  "wrist_eda_eda_std_lag1": -0.5888, "wrist_eda_eda_std_lag2": -0.4662,
  "wrist_temp_temp_mean_lag1": -0.2529, "wrist_temp_temp_mean_lag2": -0.5213,
  "wrist_temp_temp_std_lag1": -0.4416, "wrist_temp_temp_std_lag2": -0.0594,
  "wrist_bvp_bvp_mean_lag1": -0.1137, "wrist_bvp_bvp_mean_lag2": -0.1472,
  "wrist_bvp_bvp_std_lag1": -0.3859, "wrist_bvp_bvp_std_lag2": -0.6591,
  "alpha_beta_ratio_TP9_lag1": -0.7222, "alpha_beta_ratio_TP9_lag2": 0.457,
  "theta_beta_ratio_TP9_lag1": -1.3701, "theta_beta_ratio_TP9_lag2": -0.6535,
  "eda_hr_interaction_lag1": -0.9389, "eda_hr_interaction_lag2": -0.9899
}'''

# ──────────────────────────────────────────────────────────────────────────────
# Session state — persists the payload text across button clicks
# ──────────────────────────────────────────────────────────────────────────────
if "json_input" not in st.session_state:
    st.session_state["json_input"] = default_json
if "sample_meta" not in st.session_state:
    st.session_state["sample_meta"] = None   # (participant_id, true_label, window_num)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Input Stream")

# ── Random sample loader ───────────────────────────────────────────────────
holdout_df = load_holdout_data()

if st.sidebar.button("🎲 Load Random Sample from Held-Out Set", use_container_width=True):
    sample_row = holdout_df[feature_cols].dropna(how='all').sample(1)
    # Grab metadata for display
    idx = sample_row.index[0]
    pid        = holdout_df.loc[idx, 'participant_id']
    true_label = holdout_df.loc[idx, 'fatigue_level']
    # Convert to clean JSON
    st.session_state["json_input"] = json.dumps(
        {k: round(float(v), 4) for k, v in sample_row.iloc[0].items()},
        indent=2
    )
    st.session_state["sample_meta"] = (pid, true_label)

# Show true label badge if a sample has been loaded
if st.session_state["sample_meta"]:
    pid, true_label = st.session_state["sample_meta"]
    badge_colour = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}.get(true_label, "⚪")
    st.sidebar.info(
        f"👷 **Worker: Participant {pid}**  \n"
        f"{badge_colour} Ground Truth: **{true_label} Fatigue**"
    )
    st.sidebar.caption("Prediction below — compare with ground truth above.")

st.sidebar.markdown("---")
json_input = st.sidebar.text_area(
    "Live JSON Payload (v2 pipeline features)",
    key="json_input",
    height=380,
)

try:
    data_dict = json.loads(json_input)
    st.sidebar.success(f"✅ Valid payload — {len(data_dict)} features received.")
except json.JSONDecodeError as e:
    st.sidebar.error(f"Invalid JSON: {e}")
    st.stop()

# Load Causal Weights — check v2 dir first, then fall back to original outputs/
causal_weights = {}
for path in [os.path.join(OUTPUTS_DIR, 'causal_weights.json'),
             os.path.join(os.path.dirname(OUTPUTS_DIR), 'causal_weights.json')]:
    if os.path.exists(path):
        with open(path) as f:
            causal_weights = json.load(f)
        break

def is_cross_link(key: str) -> bool:
    """Return True if the causal key connects two *different* variables.

    Key format: "<SOURCE>_t-<N>-><TARGET>_t"
    Splits on '_t-' to extract source and on '->' then strips trailing '_t' for target.
    This is explicit and safe against variable names that contain the substring '_t'.
    """
    source = key.split('_t-')[0]                    # e.g. "Alpha_TP9"
    target = key.split('->')[1].rsplit('_t', 1)[0]  # e.g. "HeartRate"
    return source != target

st.sidebar.markdown("---")
st.sidebar.header("🧠 Tigramite Causal AI Engine")
if causal_weights:
    st.sidebar.write("Top extracted physiological pathways driving fatigue:")
    top_pathways = sorted(causal_weights.items(), key=lambda x: abs(x[1]), reverse=True)
    cross_links = [p for p in top_pathways if is_cross_link(p[0])]
    for path, weight in cross_links[:5]:
        st.sidebar.write(f"🔗 `{path}`: {weight:.2f}")
else:
    st.sidebar.warning("No causal weights found. Run `causal_analysis.py` first.")

# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────
fatigue_state, risk_score, pred_proba = analyze_fatigue(data_dict)

# ──────────────────────────────────────────────────────────────────────────────
# UI Layout
# ──────────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    # ── Current Classification (the model's discrete verdict) ─────────────────
    st.subheader("Current Classification")
    state_colours = {'Low': 'alert-low', 'Medium': 'alert-med', 'High': 'alert-high'}
    state_icons   = {'Low': '', 'Medium': '', 'High': ''}
    st.markdown(
        f'<div class="alert-card {state_colours[fatigue_state]}">'
        f'{state_icons[fatigue_state]} <b>Fatigue Level: {fatigue_state.upper()}</b><br>'
        f'<small>Model\'s discrete verdict — drives the intervention on the right.</small>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Fatigue Proximity Score (continuous, reflects uncertainty) ─────────────
    st.subheader("Fatigue Proximity Score")
    st.caption("How close is this worker to the danger zone? (0 = deeply safe, 100 = maximum risk). "
               "Even a Low classification can show an elevated score if the model has partial doubt.")
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = risk_score,
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33.3],  'color': "lightgreen"},
                {'range': [33.3, 66.6], 'color': "orange"},
                {'range': [66.6, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': 'black', 'width': 3},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(t=5, b=0, l=5, r=5)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Class probability breakdown ────────────────────────────────────────────
    st.markdown("### Class Probabilities")
    conf_df = pd.DataFrame({
        'Fatigue Class': ['Low', 'Medium', 'High'],
        'Probability': [f"{p*100:.1f}%" for p in pred_proba]
    })
    st.dataframe(conf_df, hide_index=True, use_container_width=True)

    # Key sensor readings from the payload (z-scored values)
    st.markdown("### Key Telemetry (z-scored)")
    st.caption("Values are per-participant z-scores: 0 = worker's average, +1 = 1 std above their normal.")
    hr   = data_dict.get('wrist_hr_hr_mean', np.nan)
    eda  = data_dict.get('wrist_eda_eda_mean', np.nan)
    beta = data_dict.get('muse_eeg_beta_TP9_mean', np.nan)
    alpha= data_dict.get('muse_eeg_alpha_TP9_mean', np.nan)
    ab_ratio = data_dict.get('alpha_beta_ratio_TP9', np.nan)
    tb_ratio = data_dict.get('theta_beta_ratio_TP9', np.nan)
    temp = data_dict.get('wrist_temp_temp_mean', np.nan)
    st.metric("HR Mean (z-score)",           f"{hr:+.2f}"    if not np.isnan(hr)    else "N/A")
    st.metric("EDA Mean (z-score)",           f"{eda:+.2f}"   if not np.isnan(eda)   else "N/A")
    st.metric("Alpha/Beta Ratio (z-score)",   f"{ab_ratio:+.2f}" if not np.isnan(ab_ratio) else "N/A")
    st.metric("Theta/Beta Ratio (z-score)",   f"{tb_ratio:+.2f}" if not np.isnan(tb_ratio) else "N/A")
    st.metric("Skin Temp (z-score)",          f"{temp:+.2f}"  if not np.isnan(temp)  else "N/A")
    st.metric("Beta EEG — TP9 (z-score)",     f"{beta:+.2f}"  if not np.isnan(beta)  else "N/A")

with col2:
    st.subheader("Causal AI Interventions")

    if fatigue_state == 'Low':
        st.markdown('<div class="alert-card alert-low">✅ <b>Status: Optimal</b>. Worker is fully alert and focused. No intervention needed.</div>', unsafe_allow_html=True)

    else:
        # Root-cause routing using per-participant z-scored features (threshold > 1.0 means 1 std above normal)
        if not np.isnan(tb_ratio) and tb_ratio > 1.0:
            # High Theta/Beta ratio → Drowsiness / Microsleep risk
            st.markdown('<div class="alert-card alert-high">🚨 <b>Status: High Risk (Drowsiness / Microsleep Risk)</b></div>', unsafe_allow_html=True)
            st.markdown("#### 🔍 Root-Cause Analysis")
            st.write(f"Elevated Theta/Beta Ratio (z-score: **{tb_ratio:+.2f}**) — indicating high low-frequency drowsiness waves dominating over alertness waves.")
            st.write("**Mathematically Proven Causal Pathways (PCMCI):**")
            pathways = []
            for k, v in causal_weights.items():
                if ("Theta" in k or "Delta" in k or "Alpha" in k) and is_cross_link(k):
                    pathways.append(f"- `{k}` (Strength: {v:.2f})")
            if pathways:
                st.markdown("\n".join(pathways))
            st.markdown("#### 💊 HR Prescription")
            st.markdown('<div class="intervention">⚠️ Immediate Break. Worker shows neurological signs of micro-sleeps. Remove from hazardous task immediately and enforce a 20-minute rest.</div>', unsafe_allow_html=True)

        elif (not np.isnan(beta) and beta > 1.0) or (not np.isnan(ab_ratio) and ab_ratio < -1.0):
            # High Beta or Low Alpha/Beta ratio → Cognitive Forcing / Overload
            st.markdown('<div class="alert-card alert-med">⚠️ <b>Status: Elevated Risk (Cognitive Overload)</b></div>', unsafe_allow_html=True)
            st.markdown("#### 🔍 Root-Cause Analysis")
            st.write(f"Elevated Beta EEG (z-score: **{beta:+.2f}**) with low Alpha/Beta ratio (z-score: **{ab_ratio:+.2f}**) — sustained high-frequency brainwave activity indicating mental overload and stress.")
            st.write("**Mathematically Proven Causal Pathways (PCMCI):**")
            pathways = []
            for k, v in causal_weights.items():
                if ("Beta" in k or "Gamma" in k or "Alpha/Beta" in k) and is_cross_link(k):
                    pathways.append(f"- `{k}` (Strength: {v:.2f})")
            if pathways:
                st.markdown("\n".join(pathways))
            st.markdown("#### 💊 HR Prescription")
            st.markdown('<div class="intervention">Rotate Task. Move this worker to a physically active but cognitively simple task for 30 minutes to allow the frontal lobe to recover.</div>', unsafe_allow_html=True)

        elif not np.isnan(eda) and eda > 1.0 and not np.isnan(hr) and hr > 1.0:
            # High EDA + High HR → Physical Stress
            st.markdown('<div class="alert-card alert-med">⚠️ <b>Status: Elevated Risk (Acute Physical Stress)</b></div>', unsafe_allow_html=True)
            st.markdown("#### 🔍 Root-Cause Analysis")
            st.write(f"Elevated EDA (z-score: **{eda:+.2f}**) combined with elevated Heart Rate (**{hr:+.1f}**) — indicating acute sympathetic arousal.")
            st.write("**Mathematically Proven Causal Pathways (PCMCI):**")
            pathways = []
            for k, v in causal_weights.items():
                if ("EDA" in k or "HeartRate" in k or "SkinTemp" in k) and is_cross_link(k):
                    pathways.append(f"- `{k}` (Strength: {v:.2f})")
            if pathways:
                st.markdown("\n".join(pathways))
            st.markdown("#### 💊 HR Prescription")
            st.markdown('<div class="intervention">Hydration Break. Instruct the worker to take a 10-minute break in the cool-down area.</div>', unsafe_allow_html=True)

        else:
            # General / compounding fatigue
            if fatigue_state == 'High':
                st.markdown('<div class="alert-card alert-high">🚨 <b>Status: High Risk (Compounding Fatigue)</b></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-card alert-med">⚠️ <b>Status: Elevated Risk (General Wear)</b></div>', unsafe_allow_html=True)
            
            st.markdown("#### 🔍 Root-Cause Analysis")
            st.write("Mixed physiological shifts without a single acute spike — consistent with gradual compounding fatigue over the shift.")
            
            # Dynamically identify the highest elevated metrics for this specific worker to explain the compounding fatigue
            z_values = {
                'Beta': beta, 'Alpha': alpha, 'Theta/Beta': tb_ratio, 
                'Alpha/Beta': ab_ratio, 'EDA': eda, 'HeartRate': hr, 'SkinTemp': temp
            }
            valid_z = {k: v for k, v in z_values.items() if not np.isnan(v)}
            
            if valid_z:
                top_sensors = sorted(valid_z.items(), key=lambda x: x[1], reverse=True)[:2]
                sensor_names = [s[0] for s in top_sensors]
                st.write(f"**Primary Contributing Sensors:** {', '.join(sensor_names)}")
                st.write("**Mathematically Proven Causal Pathways (PCMCI):**")
                
                pathways = []
                for k, v in causal_weights.items():
                    if is_cross_link(k) and any(sensor in k for sensor in sensor_names):
                        pathways.append(f"- `{k}` (Strength: {v:.2f})")
                        
                if pathways:
                    st.markdown("\n".join(pathways))
                else:
                    st.write("  - `[Fatigue (t-1)] -> [Fatigue (t)]`")
            else:
                st.write("**Causal Pathway:** `[Fatigue (t-1)] -> [Fatigue (t)]`")
                
            st.markdown("#### 💊 HR Prescription")
            if fatigue_state == 'High':
                st.markdown('<div class="intervention">⚠️ Immediate Task Rotation. Worker is experiencing severe compounding fatigue. Move to a lower-risk station.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="intervention">Monitor closely. Remind worker of proper ergonomic posture.</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"*XGBoost v2 — {len(feature_cols)} features | Per-participant z-scored | Causal AI via PCMCI (Tigramite)*")
