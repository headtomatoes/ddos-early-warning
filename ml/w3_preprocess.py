import datetime
import gc
import glob
import hashlib
import os
import platform
import socket

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import LabelEncoder


CHUNK_SIZE = 150_000
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

IMPUTE_COLS = ['Flow Bytes/s', 'Flow Packets/s']
RARE_LABELS = ['WEBDDOS', 'UDPLAG', 'UDP-LAG']

# Requested expanded feature set: existing 8 + additional features.
FEATURES_V3 = [
    'Flow Bytes/s',
    'Total Length of Fwd Packets',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow Duration',
    'Total Backward Packets',
    'SYN Flag Count',
    'ACK Flag Count',
    'Protocol',
    'Destination Port',
    'Flow IAT Std',
    'Flow IAT Max',
    'Total Fwd Packets',
    'Total Length of Bwd Packets',
    'Down/Up Ratio',
    'Packet Length Std',
    'Packet Length Variance',
    'Average Packet Size',
    'Bwd Packet Length Std',
    'Fwd Packet Length Std',
    'RST Flag Count',
    'PSH Flag Count',
    'URG Flag Count',
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    'Active Std',
    'Idle Std',
    'Active Mean',
    'Idle Mean',
    'Inbound',
    'Subflow Fwd Packets',
    'Subflow Bwd Packets',
    'Bwd Packets/s',
    'Fwd Packets/s',
]

# Explicitly excluded fields to avoid leakage/memorization.
SAFE_EXCLUSIONS = [
    'Source IP',
    'Destination IP',
    'Flow ID',
    'Timestamp',
    'Unnamed: 0',
    'SimillarHTTP',
    'Fwd Header Length.1',
]

TMP_TRAIN = 'train_processed_tmp.parquet'
TMP_VAL = 'val_processed_tmp.parquet'
TMP_TEST = 'test_holdout_processed_tmp.parquet'

OUT_TRAIN = 'train_processed.parquet'
OUT_VAL = 'val_processed.parquet'
OUT_TEST = 'test_holdout_processed.parquet'

FEATURES_DTYPE = np.float32
LABEL_DTYPE = np.int32
OUTPUT_SCHEMA = pa.schema([(col, pa.float32()) for col in FEATURES_V3] + [('Label', pa.int32())])


def hash_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def append_to_log(log_path, metadata):
    lines = [
        '══════════════════════════════════════════════════════════',
        'STAGE: W3_PREPROCESS',
        '══════════════════════════════════════════════════════════',
        f"timestamp_start   : {metadata['timestamp_start']}",
        f"timestamp_end     : {metadata['timestamp_end']}",
        f"python_version    : {metadata['python_version']}",
        f"host_machine      : {metadata['host_machine']}",
        f"source_files      : {metadata['source_files_count']} CSV files",
        f"total_rows_raw    : {metadata['total_rows_raw']}",
        f"features_selected : [{', '.join(metadata['features_selected'])}]",
        f"safe_exclusions   : [{', '.join(metadata['safe_exclusions'])}]",
        f"rows_after_drop   : {metadata['rows_after_drop']}",
        f"null_imputed_cols : {metadata['null_imputed_cols']}",
        'impute_strategy   : train_global_median (fitted on train only; label-agnostic)',
        f"train_rows        : {metadata['train_rows']}  BENIGN: {metadata.get('train_benign', 0)}",
        f"val_rows          : {metadata['val_rows']}  BENIGN: {metadata.get('val_benign', 0)}",
        f"test_rows         : {metadata['test_rows']}  BENIGN: {metadata.get('test_benign', 0)}",
        f"scale_pos_weight  : {metadata.get('scale_pos_weight', 'N/A')}",
        'transforms_applied: None (Deferred to Pipeline in W4/W5)',
        'artifacts_saved   :',
    ]
    for art, hsh in metadata.get('artifacts', []):
        lines.append(f"  {art.ljust(22)} sha256: {hsh if hsh else 'N/A'}")
    lines.append('STATUS: SUCCESS\n')

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def normalize_labels(series):
    s = series.astype(str).str.strip().str.upper()
    return s.replace(RARE_LABELS, 'RARE_ATTACK')


def get_column_map(file_path):
    header_df = pd.read_csv(file_path, nrows=0)
    raw_cols = header_df.columns.tolist()
    return {c.strip(): c for c in raw_cols}


def safe_remove(path):
    if os.path.exists(path):
        os.remove(path)


def enforce_output_dtypes(df):
    for col in FEATURES_V3:
        df[col] = df[col].astype(FEATURES_DTYPE)
    df['Label'] = df['Label'].astype(LABEL_DTYPE)
    return df


def split_chunk_stratified(chunk, rng):
    labels = chunk['Label'].to_numpy()
    split_codes = np.empty(len(chunk), dtype=np.uint8)

    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        r = rng.random(len(idx))
        split_codes[idx] = np.where(r < TRAIN_RATIO, 0, np.where(r < TRAIN_RATIO + VAL_RATIO, 1, 2))

    chunk['_split'] = split_codes
    train_df = chunk.loc[chunk['_split'] == 0].drop(columns=['_split'])
    val_df = chunk.loc[chunk['_split'] == 1].drop(columns=['_split'])
    test_df = chunk.loc[chunk['_split'] == 2].drop(columns=['_split'])
    return train_df, val_df, test_df


def write_chunk(df, path, writers):
    if df.empty:
        return
    table = pa.Table.from_pandas(df, schema=OUTPUT_SCHEMA, preserve_index=False)
    if path not in writers:
        writers[path] = pq.ParquetWriter(path, OUTPUT_SCHEMA, compression='snappy')
    writers[path].write_table(table)


def close_writers(writers):
    for writer in writers.values():
        writer.close()


def impute_temp_to_final(temp_path, final_path, medians):
    safe_remove(final_path)

    pf = pq.ParquetFile(temp_path)
    writer = pq.ParquetWriter(final_path, OUTPUT_SCHEMA, compression='snappy')
    for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
        df_batch = batch.to_pandas()
        for col, median in medians.items():
            if col in df_batch.columns:
                df_batch[col] = df_batch[col].fillna(median)
        df_batch = enforce_output_dtypes(df_batch)
        table = pa.Table.from_pandas(df_batch, schema=OUTPUT_SCHEMA, preserve_index=False)
        writer.write_table(table)
    writer.close()


def main():
    timestamp_start = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    print('1. Finding CSV files...')
    csv_files = glob.glob('01-12/*.csv') + glob.glob('03-11/*.csv')

    if not csv_files:
        raise FileNotFoundError('ERROR: No CSV files found!')

    print(f'Found {len(csv_files)} files. Starting...')

    required_cols = FEATURES_V3 + ['Label']

    print('\n2. Validating required columns and scanning label space...')
    total_rows_raw = 0
    label_values = set()

    file_column_maps = {}
    for file_path in csv_files:
        col_map = get_column_map(file_path)
        file_column_maps[file_path] = col_map
        missing = [c for c in required_cols if c not in col_map]
        if missing:
            raise ValueError(f'Missing required columns in {file_path}: {missing}')

        label_raw_col = col_map['Label']
        for chunk in pd.read_csv(file_path, usecols=[label_raw_col], chunksize=CHUNK_SIZE):
            chunk.columns = chunk.columns.str.strip()
            labels = normalize_labels(chunk['Label'])
            label_values.update(labels.unique().tolist())
            total_rows_raw += len(chunk)

    le = LabelEncoder()
    le.fit(sorted(label_values))
    joblib.dump(le, 'label_encoder.pkl')

    benign_idx = next((i for i, c in enumerate(le.classes_) if c == 'BENIGN'), None)
    label_to_idx = {c: int(i) for i, c in enumerate(le.classes_)}

    print('\n3. Chunked preprocessing and split writing (RAM-safe)...')
    for path in [TMP_TRAIN, TMP_VAL, TMP_TEST, OUT_TRAIN, OUT_VAL, OUT_TEST]:
        safe_remove(path)

    rng = np.random.default_rng(RANDOM_SEED)
    writers = {}

    split_rows = {'train': 0, 'val': 0, 'test': 0}
    split_benign = {'train': 0, 'val': 0, 'test': 0}
    null_counts = {col: 0 for col in IMPUTE_COLS}

    for file_path in csv_files:
        col_map = file_column_maps[file_path]
        usecols_raw = [col_map[c] for c in required_cols]

        for chunk in pd.read_csv(file_path, usecols=usecols_raw, chunksize=CHUNK_SIZE):
            chunk.columns = chunk.columns.str.strip()
            chunk['Label'] = normalize_labels(chunk['Label']).map(label_to_idx)
            chunk = chunk[chunk['Label'].notna()]
            if chunk.empty:
                continue

            chunk['Label'] = chunk['Label'].astype(np.int32)

            for col in FEATURES_V3:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')

            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Clip negative values to 0 to prevent log1p failure in w5
            for col in FEATURES_V3:
                chunk[col] = chunk[col].clip(lower=0.0)

            for col in IMPUTE_COLS:
                null_counts[col] += int(chunk[col].isna().sum())

            chunk = chunk[FEATURES_V3 + ['Label']]
            chunk = enforce_output_dtypes(chunk)
            train_df, val_df, test_df = split_chunk_stratified(chunk, rng)

            for split_name, split_df, path in [
                ('train', train_df, TMP_TRAIN),
                ('val', val_df, TMP_VAL),
                ('test', test_df, TMP_TEST),
            ]:
                if split_df.empty:
                    continue
                split_rows[split_name] += len(split_df)
                if benign_idx is not None:
                    split_benign[split_name] += int((split_df['Label'] == benign_idx).sum())
                write_chunk(split_df, path, writers)

            del chunk
            del train_df
            del val_df
            del test_df

        gc.collect()

    close_writers(writers)

    print('\n4. Computing train-only medians for imputation...')
    saved_medians = {'strategy': 'train_global_median', 'values': {}}
    for col in IMPUTE_COLS:
        series = pd.read_parquet(TMP_TRAIN, columns=[col])[col]
        saved_medians['values'][col] = float(series.median())
        del series
        gc.collect()

    joblib.dump(saved_medians, 'class_medians.pkl')

    print('\n5. Applying imputation and writing final parquet artifacts...')
    impute_temp_to_final(TMP_TRAIN, OUT_TRAIN, saved_medians['values'])
    impute_temp_to_final(TMP_VAL, OUT_VAL, saved_medians['values'])
    impute_temp_to_final(TMP_TEST, OUT_TEST, saved_medians['values'])

    for path in [TMP_TRAIN, TMP_VAL, TMP_TEST]:
        safe_remove(path)

    train_benign = split_benign['train'] if benign_idx is not None else 0
    scale_pos_weight = (split_rows['train'] - train_benign) / max(train_benign, 1)

    print('\n6. Logging to LOG_TRAINING.TXT...')
    timestamp_end = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    null_imputed_str = ', '.join(f'{k}={v}' for k, v in null_counts.items())
    metadata = {
        'timestamp_start': timestamp_start,
        'timestamp_end': timestamp_end,
        'python_version': platform.python_version(),
        'host_machine': socket.gethostname(),
        'source_files_count': len(csv_files),
        'total_rows_raw': total_rows_raw,
        'features_selected': FEATURES_V3 + ['Label'],
        'safe_exclusions': SAFE_EXCLUSIONS,
        'rows_after_drop': split_rows['train'] + split_rows['val'] + split_rows['test'],
        'null_imputed_cols': null_imputed_str,
        'train_rows': split_rows['train'],
        'train_benign': train_benign,
        'val_rows': split_rows['val'],
        'val_benign': split_benign['val'] if benign_idx is not None else 0,
        'test_rows': split_rows['test'],
        'test_benign': split_benign['test'] if benign_idx is not None else 0,
        'scale_pos_weight': round(scale_pos_weight, 2),
        'artifacts': [
            ('label_encoder.pkl', hash_file('label_encoder.pkl') if os.path.exists('label_encoder.pkl') else None),
            ('class_medians.pkl', hash_file('class_medians.pkl') if os.path.exists('class_medians.pkl') else None),
            ('train_processed.parquet', hash_file(OUT_TRAIN) if os.path.exists(OUT_TRAIN) else None),
            ('val_processed.parquet', hash_file(OUT_VAL) if os.path.exists(OUT_VAL) else None),
            ('test_holdout_processed.parquet', hash_file(OUT_TEST) if os.path.exists(OUT_TEST) else None),
        ],
    }

    append_to_log('LOG_TRAINING.TXT', metadata)

    print('\nSUCCESS: Phase 1 Pipeline Complete!')


if __name__ == '__main__':
    main()
