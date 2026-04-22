# DDOS Early Warning - Current State Summary

Date: 2026-04-20
Workspace: c:\Project\ddos-early-warning\ml

## 1) Executive Snapshot

This workspace is in a clean pre-run state after artifact cleanup.

What is complete in code right now:
- Preprocessing has been refactored to chunked, RAM-aware processing with train-only median imputation (label-agnostic).
- Week 5 training now uses the same expanded feature set as preprocessing.
- Week 5 thresholding now uses the same expanded feature set and enforces dual gates (FPR and missed-attack rate).
- Pipeline runner executes W3 -> W5 train -> W5 threshold, while ONNX export is intentionally disabled.
- Static diagnostics report no current errors in the core scripts.

What is not yet complete operationally:
- No current run artifacts exist, because outputs were intentionally cleaned before rerun.
- A full successful rerun after the latest schema-fix patch in preprocessing has not yet been recorded in this clean state.

## 2) Artifact and Log Inventory (Current On-Disk State)

Checked now in ml folder:
- LOG_TRAINING.TXT: absent
- label_encoder.pkl: absent
- class_medians.pkl: absent
- train_processed.parquet: absent
- val_processed.parquet: absent
- test_holdout_processed.parquet: absent
- pipeline_checkpoint.pkl: absent
- deploy_config.json: absent
- threshold_analysis.json: absent
- xgboost_final.onnx: absent

Interpretation:
- This is a clean slate, suitable for a full deterministic rerun.
- Any next run will generate fresh outputs from scratch.

## 3) Source Control Delta (Important Recent Changes)

Observed net changes relative to tracked history:
- run_pipeline.py modified to keep ONNX export commented out.
- w3_preprocess.py significantly updated (feature expansion, leakage removal, chunked flow, schema stability fix).
- w5_train_final.py significantly updated (artifact consistency, feature sync, FPR-gated selection logic).
- w5_threshold.py significantly updated (feature sync, diagnostics, dual-gate threshold logic).
- LOG_TRAINING.TXT, deploy_config.json, threshold_analysis.json were removed as part of cleanup.

## 4) Detailed File-by-File State

## 4.1 w3_preprocess.py

Status:
- Uses chunked CSV ingestion for lower peak memory pressure.
- Uses expanded FEATURES_V3 set.
- Uses explicit SAFE_EXCLUSIONS list for leakage/memorization-prone fields.
- Uses train-only global medians for imputation strategy.
- Writes temporary split parquet files, then writes final imputed parquet files.

Critical fixes already present:
- Label leakage removed from imputation path:
  - No val/test label-dependent median mapping is used.
  - Imputation medians are computed from train split only and applied across splits.

- Parquet schema mismatch fix applied:
  - Added fixed output schema (all features float32 + label int32).
  - Added enforce_output_dtypes() for chunk-level dtype normalization.
  - write_chunk() now writes with a fixed schema.
  - impute_temp_to_final() writes with the same fixed schema.

Why this matters:
- Prevents the observed crash where some chunks inferred float64 and others float32, causing pyarrow write_table schema mismatch.

Expected outputs of a successful W3 run:
- label_encoder.pkl
- class_medians.pkl
- train_processed.parquet
- val_processed.parquet
- test_holdout_processed.parquet
- LOG_TRAINING.TXT (appended)

## 4.2 w5_train_final.py

Status:
- Reads train/val from parquet using FEATURES_V3 (synchronized with preprocessing and threshold scripts).
- Trains a full deployment pipeline: log1p -> MinMaxScaler -> XGBClassifier.
- Uses balanced sample weights with additional benign_weight_boost.

Selection and checkpoint behavior:
- Runs three experiments (baseline, optimize, tune).
- Applies FPR-aware selection logic:
  - Prioritizes configs that pass FPR target gate.
  - Uses benign_f1/fpr tie-break logic.
- Saves exact best fitted pipeline (pipeline_checkpoint.pkl).
- Reloads checkpoint and re-evaluates to verify saved artifact metrics match selected experiment metrics.

Why this matters:
- Prevents mismatch between evaluated model and exported artifact.
- Keeps evaluation aligned with deployment object.

## 4.3 w5_threshold.py

Status:
- Reads validation parquet using FEATURES_V3 (synchronized with W3 and W5 train).
- Loads pipeline_checkpoint.pkl.
- Resolves BENIGN class probability column safely from model classes.

Diagnostics and gating behavior:
- Computes argmax FPR baseline.
- Computes BENIGN zero-probability floor diagnostics.
- Performs threshold sweep over extended low-to-high threshold range.
- Computes both:
  - FPR (% of benign flagged as attack)
  - missed_rate_pct (% of attacks missed)
- Enforces dual gates:
  - target_max_fpr_pct = 0.10
  - target_max_missed_rate_pct = 1.00
- Writes threshold_analysis.json and deploy_config.json when run.
- Raises SystemExit and blocks ONNX path if gates are not met.

Why this matters:
- Prevents false success cases where FPR looks low but attack miss-rate is operationally unacceptable.

## 4.4 run_pipeline.py

Execution order currently:
1. w3_preprocess.py
2. w5_train_final.py
3. w5_threshold.py
4. w6_onnx_export.py is intentionally commented out

Operational implication:
- ONNX export is gated manually and should remain disabled until threshold gates pass.

## 5) Validation and Health Checks

Current static checks:
- No errors found in:
  - w3_preprocess.py
  - w5_train_final.py
  - w5_threshold.py
  - run_pipeline.py

Current runtime state:
- Last observed runtime blocker (pyarrow schema mismatch) has been patched in preprocessing.
- Full end-to-end rerun after this specific patch is still pending in the clean artifact state.

## 6) Feature Set Alignment (Current Truth)

Feature alignment is now consistent across all three stages:
- W3 preprocess uses FEATURES_V3
- W5 train uses FEATURES_V3
- W5 threshold uses FEATURES_V3

FEATURES_V3 includes:
- Flow Bytes/s
- Total Length of Fwd Packets
- Flow Packets/s
- Flow IAT Mean
- Flow Duration
- Total Backward Packets
- SYN Flag Count
- ACK Flag Count
- Protocol
- Destination Port
- Flow IAT Std
- Flow IAT Max
- Total Fwd Packets
- Total Length of Bwd Packets
- Down/Up Ratio
- Packet Length Std
- Packet Length Variance
- Average Packet Size
- Bwd Packet Length Std
- Fwd Packet Length Std
- RST Flag Count
- PSH Flag Count
- URG Flag Count
- Init_Win_bytes_forward
- Init_Win_bytes_backward
- Active Std
- Idle Std
- Active Mean
- Idle Mean
- Inbound
- Subflow Fwd Packets
- Subflow Bwd Packets
- Bwd Packets/s
- Fwd Packets/s

## 7) Strict Action Board

### NOW

1. Execute a clean end-to-end rerun from [ml/run_pipeline.py](run_pipeline.py).
2. Verify all stage outputs regenerate: encoder, medians, processed parquet splits, checkpoint, threshold outputs.
3. Confirm threshold gate result (pass/fail) and capture the new block in LOG_TRAINING.TXT.

### NEXT

1. If threshold dual-gate fails, tune the model (class weighting/hyperparameters/features) and rerun Week 5.
2. Compare frontier trade-off: minimum FPR at <= 1.0% missed-rate, and minimum missed-rate at <= 0.10% FPR.
3. Update model-card style documentation with final chosen operating point and failure reason if still blocked.

### LATER

1. Re-enable [ml/w6_onnx_export.py](w6_onnx_export.py) only after threshold gate passes.
2. Implement controller/XDP/Grafana integration workflows (currently scaffold-only in repo root folders).
3. Run live-system validation and benchmarking phases from the revised weeks 7-12 plan.

## 8) TODO Status (One-Pass Rewrite)

- [x] Remove label leakage in preprocess.
- [x] Unify training and export estimator.
- [x] Harden threshold diagnostics and gating.
- [x] Run syntax and consistency checks.
- [x] Provide step-by-step execution guide.

Operational follow-up checklist (new):

- [ ] Run a fresh full pipeline after latest preprocessing schema patch.
- [ ] Validate threshold gate outcome on regenerated artifacts.
- [ ] Decide whether ONNX export can be re-enabled.

## 9) Correct Execution Guide (From Current Clean State)

From PowerShell in c:\Project\ddos-early-warning\ml:

1. Activate environment.

1. .\ddos_env\Scripts\Activate.ps1

1. Optional: reset log file for a fresh run record.

1. Remove-Item .\LOG_TRAINING.TXT -Force -ErrorAction SilentlyContinue

1. Run full pipeline.

1. python .\run_pipeline.py

1. Verify generated artifacts after completion.

1. Get-ChildItem .\label_encoder.pkl, .\class_medians.pkl, .\train_processed.parquet, .\val_processed.parquet, .\test_holdout_processed.parquet, .\pipeline_checkpoint.pkl, .\deploy_config.json, .\threshold_analysis.json

1. If threshold stage fails (expected possible outcome):

1. Keep ONNX export disabled.
1. Use deploy_config.json + threshold_analysis.json + LOG_TRAINING.TXT to drive next retraining cycle.

## 10) Risks and Watch Items

- Data volume is very large; W3 remains the longest and most I/O-heavy stage.
- The schema bug should be resolved, but first post-fix full run is the critical confirmation.
- FPR and missed-rate dual constraints are strict; threshold-only tuning may still fail if model separation is insufficient.

## 11) Bottom Line

Code-level architecture and safety controls are materially stronger and better aligned than the earlier state. The project is ready for a clean end-to-end rerun to confirm runtime success and verify whether the current model satisfies deployment gates.
