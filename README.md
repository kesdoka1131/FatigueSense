# FatigueSense

A machine-learning pipeline for **real-time driver fatigue detection** using physiological sensor data. The system combines XGBoost classification, causal inference (PC algorithm via Tigramite), and an interactive Streamlit dashboard for HR-level monitoring.

---

## Project Structure

```
fatigue-pipeline-v2/
│
├── data_pipeline.py            # Data ingestion, cleaning, windowed feature extraction
├── train_model.py              # XGBoost model training with Optuna hyperparameter tuning
├── causal_analysis.py          # Causal graph discovery (PC algorithm + CMIknn)
├── evaluate_holdout.py         # Hold-out test set evaluation & metrics
│
├── dashboard/
│   ├── app.py                  # Streamlit dashboard UI (main entry point)
│   └── hr_dashboard_backend.py # Backend helpers for the HR dashboard
│
├── tests/
│   ├── test_pipeline.py        # Unit tests for the data pipeline
│   ├── test_inference.py       # Unit tests for model inference
│   └── test_dashboard.py       # Unit tests for the dashboard backend
│
├── outputs/                    # Generated model artifacts & results
│   ├── xgboost_fatigue_model.json
│   ├── imputer.pkl
│   ├── scaler.pkl
│   ├── feature_cols.json
│   ├── causal_weights.json
│   ├── causal_graph.png
│   ├── confusion_matrix.png
│   ├── holdout_confusion_matrix.png
│   ├── feature_importance.png
│   ├── processed_features.csv
│   ├── processed_features_windowed.csv
│   ├── training_features.csv
│   ├── unseen_test_data.csv
│   └── XGBoost-v2/            # Versioned model snapshot
│
├── fatigueset/                 # Raw dataset (gitignored)
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/kesdoka1131/FatigueSense.git
cd FatigueSense

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** Place the `fatigueset/` dataset directory in the project root. It is gitignored due to its size.

---

## Running the Pipeline

### 1. Data Pipeline — Feature Extraction

```bash
python data_pipeline.py
```

Processes raw sensor data from `fatigueset/`, applies windowed feature extraction, and saves processed features to `outputs/`.

### 2. Model Training

```bash
python train_model.py
```

Trains an XGBoost classifier with Optuna-based hyperparameter tuning. Saves the model, scaler, imputer, and feature columns to `outputs/`.

### 3. Causal Analysis

```bash
python causal_analysis.py
```

Runs the PC causal discovery algorithm with CMIknn independence tests. Generates the causal graph image and causal weights JSON in `outputs/`.

### 4. Hold-out Evaluation

```bash
python evaluate_holdout.py
```

Evaluates the trained model on a truly unseen test set and generates the hold-out confusion matrix.

### 5. Dashboard

```bash
streamlit run dashboard/app.py
```

Launches the interactive Streamlit dashboard for real-time fatigue monitoring and HR-level insights.

---

## Running Tests

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_pipeline.py
pytest tests/test_inference.py
pytest tests/test_dashboard.py
```

---

## Tech Stack

| Component          | Technology                     |
| ------------------ | ------------------------------ |
| ML Model           | XGBoost                        |
| Hyperparameter Tuning | Optuna                      |
| Causal Inference   | Tigramite (PC + CMIknn)        |
| Dashboard          | Streamlit                      |
| Data Processing    | Pandas, NumPy, Scikit-learn    |
| Visualization      | Matplotlib, Seaborn, Plotly    |
| Testing            | Pytest                         |

---

*This README is a temporary placeholder — a more detailed version is coming soon.*
