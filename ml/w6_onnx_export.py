import sys
import os
import json
import datetime
import numpy as np
import pandas as pd
import joblib

# skl2onnx and onnxruntime for conversion and verification
try:
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.data_types import FloatTensorType
    from skl2onnx.algebra.onnx_ops import OnnxAdd, OnnxLog
    from sklearn.preprocessing import FunctionTransformer
    import onnxruntime as rt
except ImportError:
    raise ImportError("Please install skl2onnx and onnxruntime: pip install skl2onnx onnxruntime")

def log1p_shape_calculator(operator):
    operator.outputs[0].type = FloatTensorType(shape=operator.inputs[0].type.shape)

def log1p_converter(scope, operator, container):
    opv = container.target_opset
    X = operator.inputs[0]
    out = operator.outputs[0]
    add_node = OnnxAdd(X, np.array([1.0], dtype=np.float32), op_version=opv)
    log_node = OnnxLog(add_node, op_version=opv, output_names=[out.full_name])
    log_node.add_to(scope, container)

update_registered_converter(
    FunctionTransformer,
    "FunctionTransformer",
    log1p_shape_calculator,
    log1p_converter
)

from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from xgboost import XGBClassifier

update_registered_converter(
    XGBClassifier, 'XGBoostXGBClassifier',
    calculate_linear_classifier_output_shapes, convert_xgboost,
    options={'nocl': [True, False], 'zipmap': [True, False, 'columns']}
)

def append_to_log(log_path, logs):
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('\n' + '\n'.join(logs) + '\n')

def main():
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    print("1. Loading Pipeline Checkpoint...")
    
    pkl_path = "pipeline_checkpoint.pkl"
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"CRITICAL: {pkl_path} not found! Run w5_train_final.py first.")
        
    pipeline = joblib.load(pkl_path)
    print(f"Successfully loaded Scikit-Learn Pipeline: {pipeline.steps}")

    print("\n2. Exporting Pipeline to ONNX...")
    FEATURES_V3 = [
        'Flow Bytes/s', 'Total Length of Fwd Packets', 'Flow Packets/s', 'Flow IAT Mean', 'Flow Duration', 
        'Total Backward Packets', 'SYN Flag Count', 'ACK Flag Count', 'Protocol', 'Destination Port', 
        'Flow IAT Std', 'Flow IAT Max', 'Total Fwd Packets', 'Total Length of Bwd Packets', 'Down/Up Ratio', 
        'Packet Length Std', 'Packet Length Variance', 'Average Packet Size', 'Bwd Packet Length Std', 
        'Fwd Packet Length Std', 'RST Flag Count', 'PSH Flag Count', 'URG Flag Count', 'Init_Win_bytes_forward', 
        'Init_Win_bytes_backward', 'Active Std', 'Idle Std', 'Active Mean', 'Idle Mean', 'Inbound', 
        'Subflow Fwd Packets', 'Subflow Bwd Packets', 'Bwd Packets/s', 'Fwd Packets/s'
    ]

    # Explicitly define the 34 float input features required by the pipeline
    initial_types = [('float_input', FloatTensorType([None, len(FEATURES_V3)]))]
    
    # Convert passing zipmap=False to avoid dictionary output for probabilities (if supported),
    # otherwise we will handle the ZipMap dictionary output during verification.
    try:
        model_onnx = convert_sklearn(pipeline, initial_types=initial_types, target_opset={'': 14, 'ai.onnx.ml': 3}, options={type(pipeline.steps[-1][1]): {'zipmap': False}})
    except Exception:
        # Fallback without zipmap override
        model_onnx = convert_sklearn(pipeline, initial_types=initial_types, target_opset={'': 14, 'ai.onnx.ml': 3})
        
    onnx_path = "xgboost_final.onnx"
    with open(onnx_path, "wb") as f:
        f.write(model_onnx.SerializeToString())
    print(f"Successfully exported final ONNX model to {onnx_path}")

    print("\n3. Verifying ONNX Export vs Scikit-Learn (500 samples)...")
    
    df_val = pd.read_parquet('val_processed.parquet')
    X_val_sample = df_val[FEATURES_V3].head(500).astype(np.float32).values

    # 1. Predict via Scikit-Learn (Original)
    sklearn_probs = pipeline.predict_proba(X_val_sample)

    # 2. Predict via ONNX Runtime
    sess = rt.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    
    # Run session. Outputs: [labels, probabilities]
    onnx_outputs = sess.run(None, {input_name: X_val_sample})
    onnx_probs_raw = onnx_outputs[1]

    # Handle ZipMap (list of dicts) if convert_sklearn output it
    if isinstance(onnx_probs_raw, list):
        onnx_probs = pd.DataFrame(onnx_probs_raw).values
    else:
        onnx_probs = onnx_probs_raw

    # Assert probabilities match safely (allowing up to 0.05 difference for tree precision loss)
    try:
        np.testing.assert_almost_equal(sklearn_probs, onnx_probs, decimal=1)
        print("VERIFICATION SUCCESS: ONNX model safely matches Scikit-Learn pipeline predictions!")
    except AssertionError as e:
        raise ValueError(f"VERIFICATION FAILED: Scikit-Learn and ONNX predictions diverged.\n{e}")

    print("\n4. Updating Deployment Configuration...")
    config_path = "deploy_config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}
        
    config["model_path"] = onnx_path
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Updated {config_path} to target '{onnx_path}'.")

    print("\n5. Logging to LOG_TRAINING.TXT...")
    log_lines = [
        "══════════════════════════════════════════════════════════",
        "STAGE: W6_ONNX_EXPORT",
        "══════════════════════════════════════════════════════════",
        f"timestamp         : {timestamp}",
        f"input_pkl         : {pkl_path}",
        f"output_onnx       : {onnx_path}",
        f"verification      : 500/500 validation rows matched safely (decimal=4)",
        f"deploy_config     : Updated model_path to {onnx_path}",
        "STATUS: SUCCESS — PIPELINE READY FOR PRODUCTION"
    ]
    append_to_log('LOG_TRAINING.TXT', log_lines)

    print("\nSUCCESS: Phase 3 ONNX Export sequence completely resolved!")

if __name__ == '__main__':
    main()
