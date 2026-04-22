import datetime
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def append_to_log(log_path, logs):
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('\n' + '\n'.join(logs) + '\n')


def get_model_classes(model):
    if hasattr(model, 'named_steps') and 'model' in model.named_steps and hasattr(model.named_steps['model'], 'classes_'):
        return np.asarray(model.named_steps['model'].classes_)
    if hasattr(model, 'classes_'):
        return np.asarray(model.classes_)
    raise ValueError('Unable to resolve model classes_ from checkpoint.')


def benign_fpr_from_predictions(y_true, y_pred, benign_idx):
    cm = confusion_matrix(y_true, y_pred)
    real_benign = int(np.sum(cm[benign_idx, :]))
    true_benign = int(cm[benign_idx, benign_idx])
    false_alarms = max(real_benign - true_benign, 0)
    fpr_pct = (false_alarms / max(real_benign, 1)) * 100.0
    return fpr_pct, false_alarms, real_benign


def main():
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    print('1. Loading Artifacts and Data...')

    try:
        le = joblib.load('label_encoder.pkl')
    except FileNotFoundError:
        raise FileNotFoundError('CRITICAL: label_encoder.pkl not found!')

    benign_idx = list(le.classes_).index('BENIGN')

    features_v3 = [
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

    print('Loading val_processed.parquet...')
    df_val = pd.read_parquet('val_processed.parquet', columns=features_v3 + ['Label'])
    x_val = df_val[features_v3]
    y_val = df_val['Label'].to_numpy()

    model_path = 'pipeline_checkpoint.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file {model_path} not found. Run w5_train_final.py first.')

    print(f'Loading {model_path} via joblib...')
    model = joblib.load(model_path)

    model_classes = get_model_classes(model)
    benign_positions = np.where(model_classes == benign_idx)[0]
    if benign_positions.size != 1:
        raise ValueError(
            f'Expected one BENIGN class index ({benign_idx}) in model classes, got {benign_positions.tolist()} '
            f'with model classes {model_classes.tolist()}'
        )
    benign_prob_col = int(benign_positions[0])

    print('2. Predicting probabilities and performing Threshold Sweep...')
    y_probs = model.predict_proba(x_val)
    y_prob_benign = y_probs[:, benign_prob_col]

    y_pred_argmax = model.predict(x_val)
    argmax_fpr_pct, argmax_false_alarms, total_benign = benign_fpr_from_predictions(y_val, y_pred_argmax, benign_idx)

    actual_benign = (y_val == benign_idx)
    actual_attack = ~actual_benign
    total_attack = int(actual_attack.sum())

    benign_prob_zero_count = int((y_prob_benign[actual_benign] == 0.0).sum())
    benign_prob_zero_pct = (benign_prob_zero_count / max(int(actual_benign.sum()), 1)) * 100.0

    target_max_fpr_pct = 0.10
    target_max_missed_rate_pct = 1.00

    thresholds_to_test = [
        0.000001, 0.000005, 0.000010, 0.000050, 0.000100, 0.000500,
        0.001, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.300, 0.400, 0.500
    ]

    sweep_results = []
    log_sweep_lines = []

    for threshold in thresholds_to_test:
        pred_benign = (y_prob_benign >= threshold)

        false_alarms = int((~pred_benign & actual_benign).sum())
        missed_attacks = int((pred_benign & actual_attack).sum())
        fpr_pct = (false_alarms / max(int(actual_benign.sum()), 1)) * 100.0
        missed_rate_pct = (missed_attacks / max(total_attack, 1)) * 100.0

        sweep_results.append({
            'threshold': float(threshold),
            'fpr': float(fpr_pct),
            'missed': missed_attacks,
            'missed_rate_pct': float(missed_rate_pct)
        })

        log_sweep_lines.append(
            f'  {threshold:.6f}: FPR={fpr_pct:.3f}%  missed={missed_attacks} ({missed_rate_pct:.3f}%)'
        )

    valid_thresholds = [
        row for row in sweep_results
        if row['fpr'] <= target_max_fpr_pct and row['missed_rate_pct'] <= target_max_missed_rate_pct
    ]

    threshold_floor_exceeded = benign_prob_zero_pct > target_max_fpr_pct
    failure_reason = ''

    if threshold_floor_exceeded:
        sweep_results.sort(key=lambda row: row['fpr'])
        best = sweep_results[0]
        target_met = False
        failure_reason = (
            f'BENIGN probability floor is {benign_prob_zero_pct:.3f}% from P(BENIGN)=0 samples. '
            f'This exceeds target {target_max_fpr_pct:.3f}%, so threshold tuning alone cannot pass.'
        )
    elif valid_thresholds:
        valid_thresholds.sort(key=lambda row: row['missed'])
        best = valid_thresholds[0]
        target_met = True
    else:
        sweep_results.sort(key=lambda row: (row['fpr'], row['missed_rate_pct']))
        best = sweep_results[0]
        target_met = False
        failure_reason = (
            'No tested threshold met both gates '
            f'(FPR <= {target_max_fpr_pct:.3f}% and missed_rate <= {target_max_missed_rate_pct:.3f}%). '
            'Retraining required.'
        )

    print(f'\nModel BENIGN probability column : {benign_prob_col}')
    print(f'Argmax BENIGN FPR              : {argmax_fpr_pct:.3f}% ({argmax_false_alarms} false alarms)')
    print(f'BENIGN P(BENIGN)=0 floor       : {benign_prob_zero_pct:.3f}% ({benign_prob_zero_count} samples)')
    print(f'Target Missed-Attack Rate Gate : <= {target_max_missed_rate_pct:.3f}%')
    print(f'Best Selected Threshold        : {best["threshold"]}')
    print(f'Achieved FPR                   : {best["fpr"]:.3f}%')
    print(f'Missed Attacks                 : {best["missed"]} ({best["missed_rate_pct"]:.3f}%)')

    print('\n3. Saving configuration and thresholds...')
    with open('threshold_analysis.json', 'w') as f:
        json.dump(sweep_results, f, indent=2)

    config = {
        'model_path': 'xgboost_final.onnx',
        'threshold': best['threshold'],
        'achieved_fpr_pct': best['fpr'],
        'achieved_missed_attacks': best['missed'],
        'achieved_missed_rate_pct': best['missed_rate_pct'],
        'target_max_fpr_pct': target_max_fpr_pct,
        'target_max_missed_rate_pct': target_max_missed_rate_pct,
        'target_met': target_met,
        'argmax_fpr_pct': argmax_fpr_pct,
        'benign_prob_zero_count': benign_prob_zero_count,
        'benign_prob_zero_pct': benign_prob_zero_pct,
        'failure_reason': failure_reason,
        'generated_at': timestamp
    }
    with open('deploy_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print('\n4. Logging W5_THRESHOLD to LOG_TRAINING.TXT...')
    if target_met:
        status_line = 'STATUS: SUCCESS - Target met.'
    else:
        status_line = f'STATUS: FAILED - {failure_reason}'

    log_lines = [
        '══════════════════════════════════════════════════════════',
        'STAGE: W5_THRESHOLD',
        '══════════════════════════════════════════════════════════',
        f'timestamp         : {timestamp}',
        f'model_loaded      : {model_path}',
        f'benign_prob_col   : {benign_prob_col}',
        f'argmax_fpr        : {argmax_fpr_pct:.3f}% ({argmax_false_alarms} false alarms)',
        f'benign_p0_floor   : {benign_prob_zero_pct:.3f}% ({benign_prob_zero_count} samples)',
        f'target_max_fpr    : {target_max_fpr_pct:.3f}%',
        f'target_max_missed : {target_max_missed_rate_pct:.3f}%',
        'threshold_sweep   :'
    ]
    log_lines.extend(log_sweep_lines)
    log_lines.extend([
        f'recommended_thresh: {best["threshold"]}',
        'artifacts_saved   : deploy_config.json, threshold_analysis.json',
        status_line
    ])

    append_to_log('LOG_TRAINING.TXT', log_lines)

    if not target_met:
        raise SystemExit(
            '\nPIPELINE BLOCKED: FPR target not met. ' +
            ('Threshold tuning cannot fix this checkpoint. ' if threshold_floor_exceeded else '') +
            'Retrain required before Week 6 ONNX export.'
        )

    print('\nSUCCESS: Phase 3 Threshold configuration complete! Ready for ONNX deployment.')


if __name__ == '__main__':
    main()
