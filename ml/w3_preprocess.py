import sys
import platform
import socket
import datetime
import hashlib
import glob
import gc
import json

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def hash_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        # read chunks
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def append_to_log(log_path, metadata):
    # Format according to the template
    lines = [
        "══════════════════════════════════════════════════════════",
        "STAGE: W3_PREPROCESS",
        "══════════════════════════════════════════════════════════",
        f"timestamp_start   : {metadata['timestamp_start']}",
        f"timestamp_end     : {metadata['timestamp_end']}",
        f"python_version    : {metadata['python_version']}",
        f"host_machine      : {metadata['host_machine']}",
        f"source_files      : {metadata['source_files_count']} CSV files",
        f"total_rows_raw    : {metadata['total_rows_raw']}",
        f"features_selected : [{', '.join(metadata['features_selected'])}]",
        f"rows_after_drop   : {metadata['rows_after_drop']}",
        f"null_imputed_cols : {metadata['null_imputed_cols']}",
        f"impute_strategy   : per_class_median (fitted on train only)",
        f"train_rows        : {metadata['train_rows']}  BENIGN: {metadata.get('train_benign', 0)}",
        f"val_rows          : {metadata['val_rows']}  BENIGN: {metadata.get('val_benign', 0)}",
        f"test_rows         : {metadata['test_rows']}  BENIGN: {metadata.get('test_benign', 0)}",
        f"scale_pos_weight  : {metadata.get('scale_pos_weight', 'N/A')}",
        f"transforms_applied: None (Deferred to Pipeline in W4/W5)",
        "artifacts_saved   :"
    ]
    for art, hsh in metadata.get('artifacts', []):
        lines.append(f"  {art.ljust(22)} sha256: {hsh if hsh else 'N/A'}")
    lines.append("STATUS: SUCCESS\n")

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

def main():
    timestamp_start = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    print("1. Finding CSV files...")
    csv_files = glob.glob('01-12/*.csv') + glob.glob('03-11/*.csv')

    if not csv_files:
        raise FileNotFoundError("ERROR: No CSV files found!")

    print(f"Found {len(csv_files)} files. Starting...")

    # Added SYN and ACK flag counts according to ablation plan
    FEATURES = [
        'Flow Bytes/s', 
        'Total Length of Fwd Packets', 
        'Flow Packets/s', 
        'Flow IAT Mean', 
        'Flow Duration', 
        'Total Backward Packets',
        'SYN Flag Count',
        'ACK Flag Count',
        'Label'
    ]

    df_list = []
    
    for file in csv_files:
        try:
            peek_df = pd.read_csv(file, nrows=0)
            raw_cols = peek_df.columns.tolist()
            actual_usecols = [c for c in raw_cols if c.strip() in FEATURES]
            if not actual_usecols:
                continue
            temp_df = pd.read_csv(file, engine='pyarrow', usecols=actual_usecols)
        except Exception as e:
            continue
        
        temp_df.columns = temp_df.columns.str.strip()
        
        if 'Label' in temp_df.columns:
            temp_df['Label'] = temp_df['Label'].astype(str).str.strip().str.upper()
            # Merge rare classes BEFORE encoding
            temp_df['Label'] = temp_df['Label'].replace(['WEBDDOS', 'UDPLAG', 'UDP-LAG'], 'RARE_ATTACK')
            
        temp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        for col in temp_df.columns:
            if col != 'Label':
                temp_df[col] = pd.to_numeric(temp_df[col], downcast='float')
                
        df_list.append(temp_df)
        del temp_df
        del peek_df
        gc.collect()

    print("\n3. Concatenating Master DataFrame...")
    df = pd.concat(df_list, ignore_index=True)
    del df_list
    gc.collect()

    total_rows_raw = len(df)
    
    print("\n4. Label Encoding...")
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    
    benign_idx = next((i for i, c in enumerate(le.classes_) if c == 'BENIGN'), None)
    
    # Save label encoder right away
    joblib.dump(le, 'label_encoder.pkl')

    print("\n5. Stratified Splitting Data...")
    X_train, X_temp = train_test_split(df, test_size=0.30, stratify=df['Label'], random_state=42)
    X_val, X_test = train_test_split(X_temp, test_size=0.50, stratify=X_temp['Label'], random_state=42)
    
    del df 
    del X_temp
    gc.collect()

    print("\n6. Vectorized Per-Class Imputation...")
    impute_cols = ['Flow Bytes/s', 'Flow Packets/s']
    null_imputed_str = []
    
    # Store class medians globally for inference/pipeline
    saved_medians = {}

    for col in impute_cols:
        if col in X_train.columns:
            nulls_train = X_train[col].isna().sum()
            nulls_val = X_val[col].isna().sum()
            nulls_test = X_test[col].isna().sum()
            total_nulls = nulls_train + nulls_val + nulls_test
            null_imputed_str.append(f"{col}={total_nulls}")
            
            # Vectorized O(1) Imputation
            class_medians = X_train.groupby('Label', observed=True)[col].median()
            
            X_train[col] = X_train[col].fillna(X_train['Label'].map(class_medians))
            X_val[col]   = X_val[col].fillna(X_val['Label'].map(class_medians))
            X_test[col]  = X_test[col].fillna(X_test['Label'].map(class_medians))
            
            saved_medians[col] = class_medians.to_dict()

    joblib.dump(saved_medians, 'class_medians.pkl')

    print("\n7. Saving to Parquet (Raw Unscaled Data)...")
    # Dropped the slow log1p and MinMax processing -> Defer to Phase 2 (ONNX Embedding!)
    X_train.to_parquet('train_processed.parquet', index=False)
    X_val.to_parquet('val_processed.parquet', index=False)
    X_test.to_parquet('test_holdout_processed.parquet', index=False)
    
    train_benign = (X_train['Label'] == benign_idx).sum() if benign_idx is not None else 0
    scale_pos_weight = (len(X_train) - train_benign) / max(train_benign, 1)

    print("\n8. Logging to LOG_TRAINING.TXT...")
    timestamp_end = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    
    metadata = {
        'timestamp_start': timestamp_start,
        'timestamp_end': timestamp_end,
        'python_version': platform.python_version(),
        'host_machine': socket.gethostname(),
        'source_files_count': len(csv_files),
        'total_rows_raw': total_rows_raw,
        'features_selected': FEATURES,
        'rows_after_drop': total_rows_raw,
        'null_imputed_cols': ", ".join(null_imputed_str),
        'train_rows': len(X_train),
        'train_benign': train_benign,
        'val_rows': len(X_val),
        'val_benign': (X_val['Label'] == benign_idx).sum() if benign_idx is not None else 0,
        'test_rows': len(X_test),
        'test_benign': (X_test['Label'] == benign_idx).sum() if benign_idx is not None else 0,
        'scale_pos_weight': round(scale_pos_weight, 2),
        'artifacts': [
            ('label_encoder.pkl', hash_file('label_encoder.pkl') if glob.glob('label_encoder.pkl') else None),
            ('class_medians.pkl', hash_file('class_medians.pkl') if glob.glob('class_medians.pkl') else None),
            ('train_processed.parquet', hash_file('train_processed.parquet') if glob.glob('train_processed.parquet') else None),
            ('val_processed.parquet', hash_file('val_processed.parquet') if glob.glob('val_processed.parquet') else None),
            ('test_holdout_processed.parquet', hash_file('test_holdout_processed.parquet') if glob.glob('test_holdout_processed.parquet') else None)
        ]
    }
    
    append_to_log('LOG_TRAINING.TXT', metadata)
    
    print("\nSUCCESS: Phase 1 Pipeline Complete!")

if __name__ == '__main__':
    main()