import pandas as pd
import numpy as np
import os
from pathlib import Path

# Paths
DATA_DIR = Path('fatigueset')
OUTPUT_DIR = Path('outputs/XGBoost-v2')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / 'processed_features_windowed.csv'

def process_participant(participant_id, metadata_df):
    """Processes all 3 sessions for a single participant into time windows."""
    participant_row = metadata_df[metadata_df['participant_id'] == int(participant_id)].iloc[0]
    
    session_to_label = {
        f"{int(participant_row['low_session']):02d}": 'Low',
        f"{int(participant_row['medium_session']):02d}": 'Medium',
        f"{int(participant_row['high_session']):02d}": 'High'
    }
    
    print(f"Processing Participant {participant_id}")
    
    all_sessions_features = []
    
    for session_id, label in session_to_label.items():
        session_path = DATA_DIR / participant_id / session_id
        if not session_path.exists():
            continue
            
        print(f"  Session {session_id} ({label})")
        
        target_files = {
            # ── Headband (Muse) ──
            'muse_eeg_alpha': 'forehead_eeg_alpha_abs.csv',
            'muse_eeg_beta':  'forehead_eeg_beta_abs.csv',
            'muse_eeg_theta': 'forehead_eeg_theta_abs.csv',   # NEW
            'muse_eeg_delta': 'forehead_eeg_delta_abs.csv',   # NEW
            'muse_eeg_gamma': 'forehead_eeg_gamma_abs.csv',   # NEW
            # ── Wristband ──
            'wrist_hr':   'wrist_hr.csv',
            'wrist_eda':  'wrist_eda.csv',
            'wrist_temp': 'wrist_skin_temperature.csv',       # NEW
            'wrist_bvp':  'wrist_bvp.csv',                    # NEW
        }
        
        # Load and resample each file
        resampled_dfs = []
        for key, filename in target_files.items():
            filepath = session_path / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    if 'timestamp' in df.columns:
                        # Convert ms epoch to datetime object
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        
                        # Resample to 60-second windows and compute mean and std
                        # Only applying to numeric columns
                        resampled = df.select_dtypes(include=[np.number]).resample('60s').agg(['mean', 'std'])
                        
                        # Flatten MultiIndex columns: e.g., 'TP9', 'mean' -> 'muse_eeg_alpha_TP9_mean'
                        resampled.columns = [f"{key}_{col[0]}_{col[1]}" for col in resampled.columns]
                        resampled_dfs.append(resampled)
                except Exception as e:
                    pass

        if not resampled_dfs:
            continue
            
        # Merge all sensor dataframes on their time index for this session
        session_df = pd.concat(resampled_dfs, axis=1)
        
        # Drop windows that are entirely NaN (e.g. no sensors were recording)
        session_df.dropna(how='all', inplace=True)
        
        # We forward-fill short gaps if one sensor dropped out momentarily, 
        # up to 2 minutes. The rest we leave as NaN for the model's imputer.
        session_df.ffill(limit=2, inplace=True)
        
        # --- Cross-Sensor Interaction Features (literature-backed) ---
        # Alpha/Beta ratio — canonical drowsiness/fatigue index
        if 'muse_eeg_alpha_TP9_mean' in session_df.columns and 'muse_eeg_beta_TP9_mean' in session_df.columns:
            session_df['alpha_beta_ratio_TP9'] = (
                session_df['muse_eeg_alpha_TP9_mean'] / (session_df['muse_eeg_beta_TP9_mean'] + 1e-8)
            )
        # Theta/Beta ratio — sustained attention / fatigue marker
        if 'muse_eeg_theta_TP9_mean' in session_df.columns and 'muse_eeg_beta_TP9_mean' in session_df.columns:
            session_df['theta_beta_ratio_TP9'] = (
                session_df['muse_eeg_theta_TP9_mean'] / (session_df['muse_eeg_beta_TP9_mean'] + 1e-8)
            )
        # EDA × HR interaction — sympathetic arousal indicator
        if 'wrist_eda_eda_mean' in session_df.columns and 'wrist_hr_hr_mean' in session_df.columns:
            session_df['eda_hr_interaction'] = (
                session_df['wrist_eda_eda_mean'] * session_df['wrist_hr_hr_mean']
            )
        # ----------------------------------------------------------
        
        # --- Advanced Feature Engineering: Time-Lagged Features ---
        # Add values from the previous 1 and 2 minutes to give XGBoost sequential context
        numeric_cols = session_df.columns.tolist()
        for col in numeric_cols:
            session_df[f"{col}_lag1"] = session_df[col].shift(1)
            session_df[f"{col}_lag2"] = session_df[col].shift(2)
        # ------------------------------------------------------------
        
        # Add metadata columns
        session_df['participant_id'] = participant_id
        session_df['session_id'] = session_id
        session_df['fatigue_level'] = label
        
        # We can also keep the window timestamp as a column if needed
        session_df.reset_index(inplace=True)
        
        all_sessions_features.append(session_df)
        
    if all_sessions_features:
        return pd.concat(all_sessions_features, ignore_index=True)
    else:
        return pd.DataFrame()

def main():
    metadata_path = DATA_DIR / 'metadata.csv'
    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found.")
        return
        
    metadata_df = pd.read_csv(metadata_path)
    
    all_data = []
    for i in range(1, 13):
        participant_id = f"{i:02d}"
        participant_df = process_participant(participant_id, metadata_df)
        if not participant_df.empty:
            all_data.append(participant_df)
        
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved features to {OUTPUT_FILE}")
    print(f"Total rows (1-minute windows): {len(final_df)}")
    print(f"Total columns: {len(final_df.columns)}")

if __name__ == '__main__':
    main()
