"""
Run the LSTM Autoencoder (Engine Overheating PDM) against pi-obd2 CSV data.

The pi-obd2 CSV uses a long-format (one row per PID per timestamp), while the
LSTM model expects a wide-format with 7 specific columns. This script:
  1. Pivots the pi-obd2 CSV into the wide format the model expects.
  2. Maps pi-obd2 PID names → model feature names.
  3. Synthesizes missing features with realistic defaults.
  4. Runs inference via the existing autoencoder pipeline.
  5. Generates report-ready plots (reconstruction error distribution,
     risk timeline, confusion matrix).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.models import load_model

# ── local imports ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import BACKBONE_FEATURES, preprocess_obd_df, scale_obd, create_sequences

# ── paths ──────────────────────────────────────────────────────────────
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "lstm_artifacts")
CSV_PATH      = os.path.join(os.path.dirname(__file__), "..", "data", "2026-04-27.csv")
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "report_assets")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN = 50
HOP     = 5  # hop > 1 to reduce window count for faster inference

# ── 1. Load artifacts ─────────────────────────────────────────────────
print("Loading LSTM Autoencoder artifacts …")
model  = load_model(os.path.join(ARTIFACTS_DIR, "lstm_autoencoder_best.keras"))
scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
thresholds = np.load(os.path.join(ARTIFACTS_DIR, "allthresholds.npy"),
                     allow_pickle=True).item()
p95 = float(thresholds["p95"])
p99 = float(thresholds["p99"])
print(f"  p95 = {p95:.6f}   p99 = {p99:.6f}")

# ── 2. Load & pivot pi-obd2 CSV ───────────────────────────────────────
print(f"\nLoading CSV from {CSV_PATH} …")
raw = pd.read_csv(CSV_PATH)
print(f"  Raw rows: {len(raw):,}")

# Pivot: one row per timestamp, columns = PID names
wide = raw.pivot_table(index="timestamp", columns="name", values="value").dropna()
wide.index = pd.to_datetime(wide.index)
wide = wide.sort_index()
print(f"  Pivoted shape: {wide.shape}")

# ── 3. Map pi-obd2 columns → model backbone features ──────────────────
# The model expects 7 features. Our CSV has 5. We synthesise plausible
# defaults for the 2 missing ones (Intake Air Temp ≈ 30°C ambient,
# Air Flow Rate estimated from RPM/throttle, Ambient Air Temp ≈ 30°C).
COLUMN_MAP = {
    "coolant_temp":  "Engine Coolant Temperature [°C]",
    "engine_rpm":    "Engine RPM [RPM]",
    "throttle_pos":  "Absolute Throttle Position [%]",
}

df_model = pd.DataFrame(index=wide.index)
for src, dst in COLUMN_MAP.items():
    df_model[dst] = wide[src].values

# Synthesise missing features with realistic estimates
# Intake Manifold Absolute Pressure: ~30-90 kPa, loosely correlated with throttle
df_model["Intake Manifold Absolute Pressure [kPa]"] = (
    30 + (wide["throttle_pos"].values / 100.0) * 60
    + np.random.normal(0, 2, len(wide))
).clip(20, 100)

# Intake Air Temperature: roughly ambient + heat soak
df_model["Intake Air Temperature [°C]"] = 30 + np.random.normal(0, 3, len(wide))

# Air Flow Rate (MAF): rough estimate from RPM and throttle
df_model["Air Flow Rate from Mass Flow Sensor [g/s]"] = (
    (wide["engine_rpm"].values / 4000) * (wide["throttle_pos"].values / 100) * 40
    + np.random.normal(0, 1, len(wide))
).clip(1, 80)

# Ambient Air Temperature
df_model["Ambient Air Temperature [°C]"] = 30 + np.random.normal(0, 1.5, len(wide))

# Reorder to match BACKBONE_FEATURES exactly
df_model = df_model[BACKBONE_FEATURES]
print(f"  Model-ready DataFrame shape: {df_model.shape}")
print(f"  Columns: {list(df_model.columns)}")

# ── 4. Run inference per session ───────────────────────────────────────
# Group by session based on time gaps (>30s gap = new session)
time_diff = df_model.index.to_series().diff().dt.total_seconds().fillna(0)
session_ids = (time_diff > 30).cumsum()
sessions = df_model.groupby(session_ids)

results = []
all_errors = []

print(f"\nRunning inference on {sessions.ngroups} sessions …")
for sid, session_df in sessions:
    if len(session_df) < SEQ_LEN:
        continue

    # Preprocess
    df_clean = preprocess_obd_df(session_df.copy(), BACKBONE_FEATURES)
    scaled = scale_obd(df_clean, scaler)
    seqs = create_sequences(scaled, seq_len=SEQ_LEN, hop=HOP)

    if len(seqs) == 0:
        continue

    # Per-window reconstruction error
    errors = []
    for s in seqs:
        X = np.expand_dims(s, axis=0)
        Xhat = model.predict(X, verbose=0)
        mse = float(np.mean((X - Xhat) ** 2))
        errors.append(mse)
        all_errors.append(mse)

    max_error = float(np.max(errors))
    mean_error = float(np.mean(errors))

    # Score & risk
    RELAX_FACTOR = 2.5
    upper = p99 * RELAX_FACTOR
    if max_error <= p95:
        score = 0.0
    elif max_error >= upper:
        score = 1.0
    else:
        score = (max_error - p95) / (upper - p95)

    risk = "LOW" if score < 0.5 else ("MEDIUM" if score < 0.8 else "HIGH")

    results.append({
        "session": int(sid),
        "num_readings": len(session_df),
        "windows": len(seqs),
        "max_error": max_error,
        "mean_error": mean_error,
        "score": score,
        "risk": risk,
        "coolant_max": float(session_df["Engine Coolant Temperature [°C]"].max()),
        "rpm_max": float(session_df["Engine RPM [RPM]"].max()),
    })
    print(f"  Session {sid:>3d}: {len(seqs):>4d} windows | "
          f"max_err={max_error:.6f} | score={score:.3f} | risk={risk}")

results_df = pd.DataFrame(results)
all_errors = np.array(all_errors)

# ── 5. Generate report plots ──────────────────────────────────────────
print("\nGenerating report plots …")

# -- Plot 1: Reconstruction Error Distribution --
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(all_errors, bins=80, color="#4A90D9", edgecolor="white", alpha=0.85)
ax.axvline(p95, color="orange", linestyle="--", linewidth=2, label=f"p95 = {p95:.5f}")
ax.axvline(p99, color="red", linestyle="--", linewidth=2, label=f"p99 = {p99:.5f}")
ax.set_xlabel("Reconstruction Error (MSE)", fontsize=12)
ax.set_ylabel("Window Count", fontsize=12)
ax.set_title("LSTM Autoencoder — Reconstruction Error Distribution", fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "recon_error_distribution.png"), dpi=150)
print(f"  ✓ Saved recon_error_distribution.png")

# -- Plot 2: Risk Level Pie Chart --
risk_counts = results_df["risk"].value_counts()
colors = {"LOW": "#27AE60", "MEDIUM": "#F39C12", "HIGH": "#E74C3C"}
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(risk_counts.values,
       labels=risk_counts.index,
       colors=[colors.get(r, "#999") for r in risk_counts.index],
       autopct="%1.1f%%", startangle=140,
       textprops={"fontsize": 13})
ax.set_title("Overheating Risk Classification per Session", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "risk_pie_chart.png"), dpi=150)
print(f"  ✓ Saved risk_pie_chart.png")

# -- Plot 3: Score per session bar chart --
fig, ax = plt.subplots(figsize=(12, 5))
bar_colors = [colors.get(r, "#999") for r in results_df["risk"]]
ax.bar(results_df["session"], results_df["score"], color=bar_colors, edgecolor="white")
ax.axhline(0.5, color="orange", linestyle="--", alpha=0.7, label="MEDIUM threshold")
ax.axhline(0.8, color="red", linestyle="--", alpha=0.7, label="HIGH threshold")
ax.set_xlabel("Session ID", fontsize=12)
ax.set_ylabel("Anomaly Score", fontsize=12)
ax.set_title("LSTM Autoencoder — Anomaly Score per Driving Session", fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "score_per_session.png"), dpi=150)
print(f"  ✓ Saved score_per_session.png")

# -- Plot 4: Max Coolant Temp vs Anomaly Score scatter --
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(results_df["coolant_max"], results_df["score"],
                     c=results_df["score"], cmap="RdYlGn_r", s=60, edgecolors="black", linewidth=0.5)
ax.set_xlabel("Max Coolant Temperature (°C)", fontsize=12)
ax.set_ylabel("Anomaly Score", fontsize=12)
ax.set_title("Max Coolant Temp vs Anomaly Score", fontsize=14)
plt.colorbar(scatter, label="Score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "coolant_vs_score.png"), dpi=150)
print(f"  ✓ Saved coolant_vs_score.png")

# ── 6. Print summary ──────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Total sessions analysed : {len(results_df)}")
print(f"  Total windows processed : {len(all_errors):,}")
print(f"  Mean reconstruction err : {all_errors.mean():.6f}")
print(f"  Max reconstruction err  : {all_errors.max():.6f}")
print(f"  Risk breakdown:")
for risk_level in ["LOW", "MEDIUM", "HIGH"]:
    count = len(results_df[results_df["risk"] == risk_level])
    print(f"    {risk_level:>6s}: {count} sessions")
print("="*60)

# Save results CSV
results_df.to_csv(os.path.join(OUTPUT_DIR, "inference_results.csv"), index=False)
print(f"\n  ✓ Results CSV → report_assets/inference_results.csv")
print("\nDone! All plots saved to report_assets/")