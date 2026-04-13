import sys
import datetime
import json
import os

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

def append_to_log(log_path, logs):
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('\n' + '\n'.join(logs) + '\n')

def main():
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    print("1. Loading Artifacts and Data...")

    try:
        le = joblib.load('label_encoder.pkl')
    except FileNotFoundError:
        raise FileNotFoundError("CRITICAL: label_encoder.pkl not found!")
        
    benign_idx = list(le.classes_).index('BENIGN')

    FEATURES_V2 = [
        'Flow Bytes/s', 
        'Total Length of Fwd Packets', 
        'Flow Packets/s', 
        'Flow IAT Mean', 
        'Flow Duration', 
        'Total Backward Packets',
        'SYN Flag Count',
        'ACK Flag Count'
    ]

    print("Loading valid_processed.parquet...")
    df_val = pd.read_parquet('val_processed.parquet')
    X_val = df_val[FEATURES_V2]
    y_val = df_val['Label'].values

    model_path = 'pipeline_checkpoint.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Run w5_train_final.py first.")

    print(f"Loading {model_path} via joblib...")
    model = joblib.load(model_path)

    print("2. Predicting probabilities and performing Threshold Sweep...")
    y_probs = model.predict_proba(X_val)
    y_prob_benign = y_probs[:, benign_idx]

    actual_benign = (y_val == benign_idx)
    actual_attack = (y_val != benign_idx)

    total_benign = actual_benign.sum()
    total_attack = actual_attack.sum()
    
    TARGET_MAX_FPR_PCT = 0.10

    # Sweep thresholds to find the boundary of FPR vs Missed Attacks
    thresholds_to_test = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

    sweep_results = []
    log_sweep_lines = []

    for t in thresholds_to_test:
        # Predict BENIGN if P_BENIGN >= threshold
        pred_benign = (y_prob_benign >= t)

        false_positives = (~pred_benign & actual_benign).sum()  # Real Benign flagged as Attack
        missed_attacks = (pred_benign & actual_attack).sum()    # Real Attack flagged as Benign

        fpr_pct = (false_positives / max(total_benign, 1)) * 100.0

        sweep_results.append({
            "threshold": t,
            "fpr": float(fpr_pct),
            "missed": int(missed_attacks)
        })

        log_sweep_lines.append(f"  {t:.3f}: FPR={fpr_pct:.3f}%  missed={missed_attacks}")

    # Select the optimal threshold (where FPR <= 0.10% and minimizing missed attacks)
    valid_thresholds = [r for r in sweep_results if r["fpr"] <= TARGET_MAX_FPR_PCT]

    if valid_thresholds:
        # Sort by missed attacks ascending (best performance overall while passing FPR gate)
        valid_thresholds.sort(key=lambda x: x["missed"])
        best = valid_thresholds[0]
        target_met = True
    else:
        # Fallback to the one with the lowest FPR if none met the target
        sweep_results.sort(key=lambda x: x["fpr"])
        best = sweep_results[0]
        target_met = False

    print(f"\nBest Selected Threshold : {best['threshold']}")
    print(f"Achieved FPR            : {best['fpr']:.3f}%")
    print(f"Missed Attacks          : {best['missed']}")
    
    print("\n3. Saving configuration and thresholds...")
    with open("threshold_analysis.json", "w") as f:
        json.dump(sweep_results, f, indent=2)

    config = {
        "model_path": "xgboost_final.onnx", # To be used downstream by ONNX exporter
        "threshold":  best["threshold"],
        "achieved_fpr_pct": best["fpr"],
        "achieved_missed_attacks": best["missed"],
        "target_met": target_met,
        "generated_at": timestamp
    }
    with open("deploy_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n4. Logging W5_THRESHOLD to LOG_TRAINING.TXT...")
    status_line = "STATUS: SUCCESS — Target met." if target_met else "STATUS: FAILED — FPR target not met. Escalate to retraining with expanded feature set."

    log_lines = [
        "══════════════════════════════════════════════════════════",
        "STAGE: W5_THRESHOLD",
        "══════════════════════════════════════════════════════════",
        f"timestamp         : {timestamp}",
        f"model_loaded      : {model_path}",
        "threshold_sweep   :"
    ]
    log_lines.extend(log_sweep_lines)
    log_lines.extend([
        f"recommended_thresh: {best['threshold']}",
        "artifacts_saved   : deploy_config.json, threshold_analysis.json",
        status_line
    ])

    append_to_log('LOG_TRAINING.TXT', log_lines)

    if not target_met:
        raise SystemExit("\nPIPELINE BLOCKED: FPR target not met. Retrain required. Will not proceed to Week 6 ONNX export.")
        
    print("\nSUCCESS: Phase 3 Threshold configuration complete! Ready for ONNX deployment.")

if __name__ == '__main__':
    main()
