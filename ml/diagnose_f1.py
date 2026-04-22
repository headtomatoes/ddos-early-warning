import joblib
import pandas as pd
from sklearn.metrics import classification_report

FEATURES_V3 = [
    'Flow Bytes/s', 'Total Length of Fwd Packets', 'Flow Packets/s', 'Flow IAT Mean', 'Flow Duration', 
    'Total Backward Packets', 'SYN Flag Count', 'ACK Flag Count', 'Protocol', 'Destination Port', 
    'Flow IAT Std', 'Flow IAT Max', 'Total Fwd Packets', 'Total Length of Bwd Packets', 'Down/Up Ratio', 
    'Packet Length Std', 'Packet Length Variance', 'Average Packet Size', 'Bwd Packet Length Std', 
    'Fwd Packet Length Std', 'RST Flag Count', 'PSH Flag Count', 'URG Flag Count', 'Init_Win_bytes_forward', 
    'Init_Win_bytes_backward', 'Active Std', 'Idle Std', 'Active Mean', 'Idle Mean', 'Inbound', 
    'Subflow Fwd Packets', 'Subflow Bwd Packets', 'Bwd Packets/s', 'Fwd Packets/s'
]

def main():
    print("Loading data and model...")
    df_val = pd.read_parquet('val_processed.parquet', columns=FEATURES_V3 + ['Label'])
    x_val = df_val[FEATURES_V3]
    y_val = df_val['Label']

    le = joblib.load('label_encoder.pkl')
    model = joblib.load('pipeline_checkpoint.pkl')

    print("Predicting...")
    y_pred = model.predict(x_val)

    print("\n" + "="*50)
    print("Classification Report:")
    print("="*50)
    report = classification_report(y_val, y_pred, target_names=le.classes_, zero_division=0)
    print(report)

if __name__ == '__main__':
    main()
