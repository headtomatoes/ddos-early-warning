import sys
import platform
import socket
import datetime
import hashlib
import glob
import gc
import json
import os

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
import xgboost as xgb

def hash_file(filepath):
    if not os.path.exists(filepath):
        return 'N/A'
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def append_to_log(log_path, logs):
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('\n' + '\n'.join(logs) + '\n')

def main():
    timestamp_start = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    print("1. Loading Artifacts and Data...")
    
    # 2. Load, Don't Fit
    try:
        le = joblib.load('label_encoder.pkl')
        print(f"Loaded LabelEncoder successfully. Classes: {le.classes_}")
    except FileNotFoundError:
        raise FileNotFoundError("CRITICAL: label_encoder.pkl not found! Ensure w3_preprocess.py ran successfully.")
        
    benign_idx = list(le.classes_).index('BENIGN')
    
    # 1. Feature Updates (FEATURES_V2)
    FEATURES_V2 = [
        'Flow Bytes/s', 
        'Total Length of Fwd Packets', 
        'Flow Packets/s', 
        'Flow IAT Mean', 
        'Flow Duration', 
        'Total Backward Packets',
        'SYN Flag Count',  # <--- Added
        'ACK Flag Count'   # <--- Added
    ]

    print("Loading valid and train parquet datasets...")
    df_train = pd.read_parquet('train_processed.parquet')
    df_val = pd.read_parquet('val_processed.parquet')
    
    X_train = df_train[FEATURES_V2]
    y_train = df_train['Label']
    
    X_val = df_val[FEATURES_V2]
    y_val = df_val['Label']
    
    del df_train
    del df_val
    gc.collect()

    print("\n2. Computing Sample Weights...")
    # Calculate sample weights to handle dramatic class imbalances
    weights_train = compute_sample_weight('balanced', y_train)

    EXPERIMENTS = [
        {"stage": "W4_TRAIN", "name": "baseline", "max_depth": 4, "n_estimators": 100, "weight_clip": None},
        {"stage": "W5_OPTIMIZE", "name": "optimize", "max_depth": 6, "n_estimators": 150, "weight_clip": None},
        {"stage": "W5_TUNE", "name": "tune", "max_depth": 5, "n_estimators": 100, "weight_clip": 100.0}
    ]

    best_benign_f1 = -1.0
    best_cfg = None
    
    # Logs buffer for combined schema
    log_lines = []

    print("\n3. Running Sequential Experiment Loop (Automated Checkpointing)...")
    for i, cfg in enumerate(EXPERIMENTS):
        print(f"--- Experiment {i+1}: {cfg['name'].upper()} ---")
        
        # Apply weight clipping if configured
        current_weights = weights_train.copy()
        if cfg['weight_clip'] is not None:
            current_weights = np.clip(current_weights, 0, cfg['weight_clip'])
            
        model = xgb.XGBClassifier(
            max_depth=cfg['max_depth'],
            n_estimators=cfg['n_estimators'],
            objective='multi:softprob',
            num_class=len(le.classes_),
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        
        print(f"Training...")
        model.fit(X_train, y_train, sample_weight=current_weights)
        
        print("Evaluating...")
        y_pred = model.predict(X_val)
        
        macro_f1 = f1_score(y_val, y_pred, average='macro')
        
        # Get metrics for BENIGN specifically
        f1_scores = f1_score(y_val, y_pred, average=None)
        benign_f1 = f1_scores[benign_idx]

        # Evaluate FPR (False Alarms vs Real Benign Traffic)
        cm = confusion_matrix(y_val, y_pred)
        # Real Benign is row enign_idx
        # Predicted Benign is col enign_idx
        real_benign_count = np.sum(cm[benign_idx, :])
        false_alarms = real_benign_count - cm[benign_idx, benign_idx]
        fpr_percent = (false_alarms / max(real_benign_count, 1)) * 100
        
        # Determine Checkpoint State
        status_msg = ""
        model_saved = False
        if benign_f1 > best_benign_f1:
            best_benign_f1 = benign_f1
            best_cfg = cfg
            print(f"CHECKPOINT: New best BENIGN-F1 ({benign_f1:.4f}). Record config for pipeline build.")
            status_msg = "SUCCESS — new best checkpoint"
            model_saved = True
        else:
            print(f"ROLLBACK: Config regressed BENIGN-F1 ({benign_f1:.4f} <= {best_benign_f1:.4f}). Discarded.")
            status_msg = "FAILED — BENIGN-F1 regressed. Reverted to previous checkpoint."
            
        # Construct log block for this pipeline stage
        log_lines.extend([
            "══════════════════════════════════════════════════════════",
            f"STAGE: {cfg['stage']} ({cfg['name']})",
            "══════════════════════════════════════════════════════════",
            f"timestamp         : {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
            f"config_applied    : max_depth={cfg['max_depth']}, n_estimators={cfg['n_estimators']}, weight_clip={cfg['weight_clip']}",
            f"macro_f1          : {macro_f1:.4f}",
            f"benign_f1         : {benign_f1:.4f}",
            f"fpr               : {fpr_percent:.3f}% ({false_alarms} false alarms)",
            f"STATUS: {status_msg}"
        ])
        
        print(f"Macro-F1: {macro_f1:.4f} | BENIGN-F1: {benign_f1:.4f} | FPR: {fpr_percent:.3f}%\n")
        
        # Cleanup memory before next experiment
        del model
        del current_weights
        gc.collect()

    print("\n4. Building final Sklearn Pipeline for ONNX Export...")
    final_weights = weights_train.copy()
    if best_cfg['weight_clip'] is not None:
        final_weights = np.clip(final_weights, 0, best_cfg['weight_clip'])

    pipeline = Pipeline([
        ("log1p",  FunctionTransformer(np.log1p, validate=True)),
        ("scaler", MinMaxScaler()),
        ("model",  xgb.XGBClassifier(
            max_depth=best_cfg['max_depth'],
            n_estimators=best_cfg['n_estimators'],
            objective='multi:softprob',
            num_class=len(le.classes_),
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        ))
    ])

    print("Training final pipeline on full training data...")
    pipeline.fit(X_train, y_train, model__sample_weight=final_weights)

    best_model_path = "pipeline_checkpoint.pkl"
    joblib.dump(pipeline, best_model_path)
    print(f"Saved final pipeline to {best_model_path}")

    # Log the pipeline export
    model_size_kb = os.path.getsize(best_model_path) / 1024.0 if os.path.exists(best_model_path) else 0
    log_lines.extend([
        "══════════════════════════════════════════════════════════",
        f"STAGE: PIPELINE_EXPORT",
        "══════════════════════════════════════════════════════════",
        f"timestamp         : {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"best_config       : {best_cfg['name']}",
        f"model_size_kb     : {model_size_kb:.1f}",
        f"artifacts_saved   : {best_model_path}",
        f"STATUS: SUCCESS — PIPELINE READY FOR EXPORT"
    ])

    print("5. Appending to LOG_TRAINING.TXT...")
    append_to_log('LOG_TRAINING.TXT', log_lines)
    
    print("\nSUCCESS: Phase 2 Training Complete. Final pipeline saved as 'pipeline_checkpoint.pkl'")

if __name__ == '__main__':
    main()
