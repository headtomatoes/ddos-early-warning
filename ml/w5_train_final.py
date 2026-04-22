import datetime
import gc
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight


def append_to_log(log_path, logs):
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('\n' + '\n'.join(logs) + '\n')


def compute_validation_metrics(y_true, y_pred, benign_idx):
    macro_f1 = float(f1_score(y_true, y_pred, average='macro'))
    f1_scores = f1_score(y_true, y_pred, average=None)
    benign_f1 = float(f1_scores[benign_idx]) if benign_idx < len(f1_scores) else 0.0

    cm = confusion_matrix(y_true, y_pred)
    real_benign_count = int(np.sum(cm[benign_idx, :]))
    true_benign = int(cm[benign_idx, benign_idx])
    false_alarms = max(real_benign_count - true_benign, 0)
    fpr_percent = (false_alarms / max(real_benign_count, 1)) * 100.0

    return macro_f1, benign_f1, fpr_percent, false_alarms


def build_pipeline(cfg, num_classes):
    return Pipeline([
        ('log1p', FunctionTransformer(np.log1p, validate=True)),
        ('scaler', MinMaxScaler()),
        ('model', xgb.XGBClassifier(
            max_depth=cfg['max_depth'],
            n_estimators=cfg['n_estimators'],
            learning_rate=cfg['learning_rate'],
            min_child_weight=20,
            gamma=2.0,
            reg_lambda=10.0,
            subsample=0.8,
            colsample_bytree=0.8,
            max_delta_step=1,
            objective='multi:softprob',
            num_class=num_classes,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        ))
    ])


def main():
    print('1. Loading Artifacts and Data...')

    try:
        le = joblib.load('label_encoder.pkl')
        print(f'Loaded LabelEncoder successfully. Classes: {le.classes_}')
    except FileNotFoundError:
        raise FileNotFoundError('CRITICAL: label_encoder.pkl not found! Ensure w3_preprocess.py ran successfully.')

    benign_idx = list(le.classes_).index('BENIGN')

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
        'Fwd Packets/s'
    ]

    print('Loading train and validation parquet datasets...')
    df_train = pd.read_parquet('train_processed.parquet', columns=FEATURES_V3 + ['Label'])
    df_val = pd.read_parquet('val_processed.parquet', columns=FEATURES_V3 + ['Label'])

    X_train = df_train[FEATURES_V3]
    y_train = df_train['Label']

    X_val = df_val[FEATURES_V3]
    y_val = df_val['Label']
    y_val_np = y_val.to_numpy()

    del df_train
    del df_val
    gc.collect()

    min_feature_value = min(float(np.nanmin(X_train.to_numpy())), float(np.nanmin(X_val.to_numpy())))
    if min_feature_value <= -1.0:
        raise ValueError(
            f'CRITICAL: Found feature values <= -1 ({min_feature_value:.6f}). '
            'This breaks log1p. Inspect preprocessing before training.'
        )

    print('\n2. Computing Sample Weights...')
    weights_train = compute_sample_weight('balanced', y_train)
    benign_weight_boost = 2.0
    weights_train = weights_train * np.where(y_train.to_numpy() == benign_idx, benign_weight_boost, 1.0)

    experiments = [
        {'stage': 'W4_TRAIN', 'name': 'baseline', 'max_depth': 4, 'n_estimators': 100, 'learning_rate': 0.05, 'weight_clip': 50.0},
        {'stage': 'W5_OPTIMIZE', 'name': 'optimize', 'max_depth': 5, 'n_estimators': 150, 'learning_rate': 0.05, 'weight_clip': 50.0},
        {'stage': 'W5_TUNE', 'name': 'tune', 'max_depth': 5, 'n_estimators': 100, 'learning_rate': 0.03, 'weight_clip': 50.0}
    ]

    target_max_fpr_pct = 0.10

    best_benign_f1 = -1.0
    best_fpr_percent = float('inf')
    best_cfg = None
    best_pipeline = None
    best_metrics = None

    log_lines = []

    print('\n3. Running Sequential Experiment Loop (Architecture-Consistent Checkpointing)...')
    for i, cfg in enumerate(experiments):
        print(f"--- Experiment {i + 1}: {cfg['name'].upper()} ---")

        current_weights = weights_train.copy()
        if cfg['weight_clip'] is not None:
            current_weights = np.clip(current_weights, 0, cfg['weight_clip'])

        pipeline = build_pipeline(cfg, len(le.classes_))

        print('Training full deployment pipeline...')
        pipeline.fit(X_train, y_train, model__sample_weight=current_weights)

        print('Evaluating on validation split...')
        y_pred = pipeline.predict(X_val)
        macro_f1, benign_f1, fpr_percent, false_alarms = compute_validation_metrics(y_val, y_pred, benign_idx)

        y_prob_benign = pipeline.predict_proba(X_val)[:, benign_idx]
        actual_benign = (y_val_np == benign_idx)
        benign_zero_prob_count = int((y_prob_benign[actual_benign] == 0.0).sum())
        benign_zero_prob_pct = (benign_zero_prob_count / max(int(actual_benign.sum()), 1)) * 100.0

        is_fpr_pass = fpr_percent <= target_max_fpr_pct
        best_is_fpr_pass = best_metrics is not None and best_metrics['fpr_percent'] <= target_max_fpr_pct

        if is_fpr_pass and not best_is_fpr_pass:
            is_better = True
        elif is_fpr_pass and best_is_fpr_pass:
            is_better = (
                (benign_f1 > best_benign_f1 + 1e-9)
                or (abs(benign_f1 - best_benign_f1) <= 1e-9 and fpr_percent < best_fpr_percent)
            )
        elif (not is_fpr_pass) and (not best_is_fpr_pass):
            is_better = (
                (fpr_percent < best_fpr_percent - 1e-9)
                or (abs(fpr_percent - best_fpr_percent) <= 1e-9 and benign_f1 > best_benign_f1)
            )
        else:
            is_better = False

        if is_better:
            best_benign_f1 = benign_f1
            best_fpr_percent = fpr_percent
            best_cfg = cfg
            best_pipeline = pipeline
            best_metrics = {
                'macro_f1': macro_f1,
                'benign_f1': benign_f1,
                'fpr_percent': fpr_percent,
                'false_alarms': false_alarms,
                'benign_zero_prob_count': benign_zero_prob_count,
                'benign_zero_prob_pct': benign_zero_prob_pct
            }
            status_msg = 'SUCCESS - new best checkpoint'
            print(f'CHECKPOINT: New best BENIGN-F1 ({benign_f1:.4f}). Saved fitted pipeline in memory.')
        else:
            status_msg = 'FAILED - BENIGN-F1 regressed. Reverted to previous checkpoint.'
            print(f'ROLLBACK: Config regressed BENIGN-F1 ({benign_f1:.4f} <= {best_benign_f1:.4f}). Discarded.')
            del pipeline

        log_lines.extend([
            '══════════════════════════════════════════════════════════',
            f"STAGE: {cfg['stage']} ({cfg['name']})",
            '══════════════════════════════════════════════════════════',
            f"timestamp         : {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
            (
                'config_applied    : '
                f"max_depth={cfg['max_depth']}, n_estimators={cfg['n_estimators']}, "
                f"learning_rate={cfg['learning_rate']}, weight_clip={cfg['weight_clip']}, "
                'min_child_weight=20, gamma=2.0, reg_lambda=10.0, '
                f'subsample=0.8, colsample_bytree=0.8, max_delta_step=1, benign_weight_boost={benign_weight_boost}'
            ),
            f'fpr_gate_target    : <= {target_max_fpr_pct:.3f}%',
            f'fpr_gate_pass      : {is_fpr_pass}',
            f'macro_f1          : {macro_f1:.4f}',
            f'benign_f1         : {benign_f1:.4f}',
            f'fpr               : {fpr_percent:.3f}% ({false_alarms} false alarms)',
            f'benign_p0_pct     : {benign_zero_prob_pct:.3f}% ({benign_zero_prob_count} samples)',
            f'STATUS: {status_msg}'
        ])

        print(
            f'Macro-F1: {macro_f1:.4f} | '
            f'BENIGN-F1: {benign_f1:.4f} | '
            f'FPR: {fpr_percent:.3f}% | '
            f'BENIGN P0: {benign_zero_prob_pct:.3f}%\n'
        )

        del current_weights
        gc.collect()

    if best_pipeline is None or best_cfg is None or best_metrics is None:
        raise RuntimeError('No valid experiment checkpoint produced. Training aborted.')

    print('\n4. Saving exact best fitted pipeline artifact...')
    best_model_path = 'pipeline_checkpoint.pkl'
    joblib.dump(best_pipeline, best_model_path)
    print(f'Saved final pipeline to {best_model_path}')

    print('5. Verifying saved checkpoint metrics on validation split...')
    reloaded_pipeline = joblib.load(best_model_path)
    y_ckpt_pred = reloaded_pipeline.predict(X_val)
    macro_ckpt, benign_ckpt, fpr_ckpt, false_alarms_ckpt = compute_validation_metrics(y_val, y_ckpt_pred, benign_idx)

    y_ckpt_prob_benign = reloaded_pipeline.predict_proba(X_val)[:, benign_idx]
    actual_benign = (y_val_np == benign_idx)
    benign_zero_prob_ckpt = int((y_ckpt_prob_benign[actual_benign] == 0.0).sum())
    benign_zero_prob_ckpt_pct = (benign_zero_prob_ckpt / max(int(actual_benign.sum()), 1)) * 100.0

    tol = 1e-6
    if (
        abs(benign_ckpt - best_metrics['benign_f1']) > tol
        or abs(fpr_ckpt - best_metrics['fpr_percent']) > tol
    ):
        raise RuntimeError(
            'Checkpoint mismatch: exported pipeline metrics diverged from experiment metrics. '
            f"Expected BENIGN-F1={best_metrics['benign_f1']:.6f}, FPR={best_metrics['fpr_percent']:.6f}; "
            f'got BENIGN-F1={benign_ckpt:.6f}, FPR={fpr_ckpt:.6f}'
        )

    model_size_kb = os.path.getsize(best_model_path) / 1024.0 if os.path.exists(best_model_path) else 0.0
    timestamp_now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    log_lines.extend([
        '══════════════════════════════════════════════════════════',
        'STAGE: PIPELINE_EXPORT',
        '══════════════════════════════════════════════════════════',
        f'timestamp         : {timestamp_now}',
        f"best_config       : {best_cfg['name']}",
        f"model_size_kb     : {model_size_kb:.1f}",
        f'artifacts_saved   : {best_model_path}',
        f'checkpoint_macro_f1: {macro_ckpt:.4f}',
        f'checkpoint_benign_f1: {benign_ckpt:.4f}',
        f'checkpoint_fpr    : {fpr_ckpt:.3f}% ({false_alarms_ckpt} false alarms)',
        f'checkpoint_benign_p0_pct: {benign_zero_prob_ckpt_pct:.3f}% ({benign_zero_prob_ckpt} samples)',
        'STATUS: SUCCESS - PIPELINE READY FOR EXPORT'
    ])

    print('6. Appending to LOG_TRAINING.TXT...')
    append_to_log('LOG_TRAINING.TXT', log_lines)

    print("\nSUCCESS: Phase 2 Training Complete. Final pipeline saved as 'pipeline_checkpoint.pkl'")


if __name__ == '__main__':
    main()
