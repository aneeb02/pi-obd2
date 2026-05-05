#artifcats loading and lstm inference
import os
import json
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from rules_engine import generate_overheat_reasons
from preprocess import (
    BACKBONE_FEATURES,
    validate_input_df,
    preprocess_obd_df,
    scale_obd,
    create_sequences,
)

class OverheatArtifacts:
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = artifacts_dir
        self.model = load_model(os.path.join(artifacts_dir, "lstm_autoencoder_best.keras"))
        self.scaler = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))

        thresholds_path = os.path.join(artifacts_dir, "allthresholds.npy")
        thresholds = np.load(thresholds_path, allow_pickle=True).item()

        self.p95 = float(thresholds["p95"])
        self.p99 = float(thresholds["p99"])


# ---------- Core logic ----------

def recon_error_mse(X: np.ndarray, model) -> float:
    """
    X shape: (1, seq_len, 7)
    """
    Xhat = model.predict(X, verbose=0)
    return float(np.mean((X - Xhat) ** 2))


def score_from_percentiles(error: float, p95: float, p99: float, relax_factor: float = 2.5) -> float:
    """
    Relaxed scoring for demo:
    - Expand anomaly boundary by multiplying p99
    """
    upper = p99 * relax_factor

    if error <= p95:
        return 0.0
    if error >= upper:
        return 1.0

    return (error - p95) / (upper - p95)

def risk_from_score(score: float) -> str:
    if score < 0.5:
        return "LOW"
    if score < 0.8:
        return "MEDIUM"
    return "HIGH"

def predict_overheat_from_readings(
    readings: list[dict],
    artifacts: OverheatArtifacts,
    seq_len: int = 50,
    hop: int = 1,
) -> dict:
    """
    End-to-end inference:
    JSON readings -> df -> validate -> preprocess -> scale -> create sequences ->
    LSTM errors -> max error -> score -> risk -> reasons
    """
    df = pd.DataFrame(readings)

    # 1) validate input
    validate_input_df(df, BACKBONE_FEATURES, min_rows=seq_len)

    # 2) preprocess (match training behavior)
    df_clean = preprocess_obd_df(df, BACKBONE_FEATURES)

    # 3) scale
    scaled = scale_obd(df_clean, artifacts.scaler)  # (N, 7)
    print("Scaled data shape:", scaled.shape)
    print("\n\n")
    print("Scaled min:", scaled.min())
    print("Scaled max:", scaled.max())

    # 4) sequences
    seqs = create_sequences(scaled, seq_len=seq_len, hop=hop)  # (W, 50, 7)
    if len(seqs) == 0:
        raise ValueError("Sequence creation returned 0 windows. Check seq_len/hop/input length.")

    # 5) LSTM inference on each window -> error list
    errors = []
    for s in seqs:
        X = np.expand_dims(s, axis=0)  # (1, 50, 7)
        errors.append(recon_error_mse(X, artifacts.model))

    max_error = float(np.max(errors))
    score = float(score_from_percentiles(max_error, artifacts.p95, artifacts.p99))
    risk = risk_from_score(score)

    # 6) explanations (OBD-only)
    reasons = generate_overheat_reasons(df_clean)

    return {
        "risk_level": risk,
        "obd_score": score,
        "max_reconstruction_error": max_error,
        "p95": artifacts.p95,
        "p99": artifacts.p99,
        "seq_len": seq_len,
        "hop": hop,
        "windows_analyzed": int(len(seqs)),
        "reasons": reasons,
        "feature_order": BACKBONE_FEATURES,
    }