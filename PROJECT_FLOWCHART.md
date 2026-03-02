# Project Flowchart

```mermaid
flowchart TD
    A["FatigueSet Raw Data<br/>12 participants x 3 sessions"] --> B["data_pipeline.py"]
    B --> C["Load sensor CSVs per session<br/>EEG bands + HR + EDA + Temp + BVP"]
    C --> D["Resample to 60-second windows<br/>mean + std per signal"]
    D --> E["Engineer features<br/>alpha/beta, theta/beta, EDA x HR"]
    E --> F["Add lag features<br/>t-1 and t-2"]
    F --> G["processed_features_windowed.csv<br/>outputs/XGBoost-v2"]

    G --> H["train_model.py"]
    H --> I["Per-participant z-score normalization"]
    I --> J["Split data<br/>train: participants 1-9<br/>holdout: participants 10-12"]
    J --> K["Impute missing values<br/>median imputer"]
    K --> L["LOSO CV + Optuna tuning<br/>XGBClassifier multi:softprob"]
    L --> M["Final model trained on participants 1-9"]
    M --> N["Saved artifacts<br/>xgboost_fatigue_model.json<br/>scaler.pkl<br/>imputer.pkl<br/>feature_cols.json<br/>unseen_test_data.csv<br/>feature_importance.png<br/>confusion_matrix.png"]

    G --> O["causal_analysis.py"]
    O --> P["Select 12 variables<br/>sensor summaries + engineered features + fatigue"]
    P --> Q["PCMCI causal discovery<br/>tau_max = 2, ParCorr"]
    Q --> R["Saved causal outputs<br/>causal_weights.json<br/>causal_graph.png"]

    N --> S["evaluate_holdout.py"]
    S --> T["Evaluate participants 10-12"]
    T --> U["holdout_confusion_matrix.png<br/>held-out metrics"]

    N --> V["dashboard/app.py"]
    R --> V
    V --> W["Input feature payload<br/>or random held-out sample"]
    W --> X["Impute -> Scale -> Predict"]
    X --> Y["Outputs<br/>fatigue class: Low / Medium / High<br/>risk score<br/>causal context for explanation"]
```

## Notes

- Current project output directory: `outputs/XGBoost-v2`
- Main dataset artifact: `processed_features_windowed.csv`
- Model type: `xgb.XGBClassifier` with 3 fatigue classes
- Causal module is generated offline via PCMCI and stored separately from the classifier artifacts
