"""
Anomaly Detection for OBD-II Vehicle Telemetry
───────────────────────────────────────────────
Uses Isolation Forest to detect anomalous driving patterns from
Pi OBD2 telemetry data. The model trains on real collected data and
is validated against synthetically injected anomalies.

Why Isolation Forest works here (and the LSTM didn't):
  - Trains directly on LOCAL data → no domain shift
  - Doesn't need temporal sequences → works on individual samples
  - Fast training and inference → runs in seconds on the Pi
  - Interpretable anomaly scores → easy to explain in a report
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

np.random.seed(42)
os.makedirs("report_assets", exist_ok=True)

# ─── 1. Load & Pivot Data ────────────────────────────────────────────
data_file = "data/2026-04-27.csv"
print(f"Loading data from {data_file}...")
df = pd.read_csv(data_file)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df_pivot = df.pivot_table(index="timestamp", columns="name", values="value").dropna()

features = ["engine_rpm", "coolant_temp", "throttle_pos", "engine_load", "vehicle_speed"]
X = df_pivot[features].copy()
print(f"  Total samples: {len(X):,}")
print(f"  Features: {features}")

# ─── 2. Train/Test Split ─────────────────────────────────────────────
# Split BEFORE fitting to avoid data leakage
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
print(f"\n  Train set: {len(X_train):,} samples")
print(f"  Test set:  {len(X_test):,} samples")

# ─── 3. Scale Features ───────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)        # fit on TRAIN only
X_test_scaled = scaler.transform(X_test)               # transform test

# ─── 4. Train Isolation Forest ────────────────────────────────────────
print("\nTraining Isolation Forest...")
model = IsolationForest(
    contamination=0.05,  # expect ~5% of real data to be edge-case outliers
    n_estimators=200,
    max_samples="auto",
    random_state=42,
)
model.fit(X_train_scaled)

# Get anomaly scores for test set (more negative = more anomalous)
test_scores = model.decision_function(X_test_scaled)
test_preds = model.predict(X_test_scaled)  # -1 = anomaly, 1 = normal

n_anomalies = (test_preds == -1).sum()
n_normal = (test_preds == 1).sum()
print(f"  Test set predictions: {n_normal} normal, {n_anomalies} anomalies ({n_anomalies/len(test_preds)*100:.1f}%)")

# ─── 5. Inject Synthetic Anomalies for Evaluation ────────────────────
# Since all our collected data is normal driving, we create synthetic
# anomalies that represent real failure modes to evaluate the model.
print("\nInjecting synthetic anomalies for evaluation...")

n_inject = 500  # number of synthetic anomaly samples

# Anomaly Type 1: Engine overheating (high coolant + low RPM = cooling failure)
overheat = pd.DataFrame({
    "engine_rpm":    np.random.uniform(700, 1000, n_inject),     # idle RPM
    "coolant_temp":  np.random.uniform(108, 125, n_inject),      # dangerously hot
    "throttle_pos":  np.random.uniform(2, 10, n_inject),         # low throttle
    "engine_load":   np.random.uniform(15, 35, n_inject),        # low load
    "vehicle_speed": np.random.uniform(0, 15, n_inject),         # crawling/stopped
})

# Anomaly Type 2: Aggressive driving (redline RPM + high throttle)
aggressive = pd.DataFrame({
    "engine_rpm":    np.random.uniform(5000, 7000, n_inject),    # way above our max of 4000
    "coolant_temp":  np.random.uniform(95, 105, n_inject),       # warm but not critical
    "throttle_pos":  np.random.uniform(85, 100, n_inject),       # full throttle
    "engine_load":   np.random.uniform(85, 100, n_inject),       # full load
    "vehicle_speed": np.random.uniform(120, 180, n_inject),      # speeding
})

# Anomaly Type 3: Sensor malfunction (impossible value combinations)
sensor_fault = pd.DataFrame({
    "engine_rpm":    np.random.uniform(0, 200, n_inject),        # near-stall
    "coolant_temp":  np.random.uniform(10, 40, n_inject),        # cold engine running?
    "throttle_pos":  np.random.uniform(50, 80, n_inject),        # high throttle but no RPM
    "engine_load":   np.random.uniform(0, 5, n_inject),          # zero load
    "vehicle_speed": np.random.uniform(60, 100, n_inject),       # moving fast with stalled engine?
})

synthetic_anomalies = pd.concat([overheat, aggressive, sensor_fault], ignore_index=True)
synthetic_scaled = scaler.transform(synthetic_anomalies[features])

# Predict on synthetic anomalies
synth_preds = model.predict(synthetic_scaled)
synth_scores = model.decision_function(synthetic_scaled)
synth_detected = (synth_preds == -1).sum()
print(f"  Synthetic anomalies: {len(synthetic_anomalies)} injected, {synth_detected} detected ({synth_detected/len(synthetic_anomalies)*100:.1f}%)")

# Break down by anomaly type
for name, start, end in [("Overheating", 0, n_inject), 
                          ("Aggressive", n_inject, 2*n_inject),
                          ("Sensor Fault", 2*n_inject, 3*n_inject)]:
    detected = (synth_preds[start:end] == -1).sum()
    print(f"    {name}: {detected}/{n_inject} detected ({detected/n_inject*100:.1f}%)")

# ─── 6. Build Combined Evaluation Set ────────────────────────────────
# Combine real test data (labeled normal) + synthetic anomalies (labeled anomaly)
# for proper classification metrics
print("\n--- Evaluation: Real Normal vs Synthetic Anomalies ---")

# Take a balanced sample of normal test data
n_normal_sample = len(synthetic_anomalies)
normal_sample_idx = np.random.choice(len(X_test_scaled), n_normal_sample, replace=False)
X_eval = np.vstack([X_test_scaled[normal_sample_idx], synthetic_scaled])
y_true = np.array([1] * n_normal_sample + [-1] * len(synthetic_anomalies))  # 1=normal, -1=anomaly
y_pred = model.predict(X_eval)

print(classification_report(
    y_true, y_pred,
    target_names=["Anomaly (-1)", "Normal (1)"],
    digits=3,
))

conf_matrix = confusion_matrix(y_true, y_pred)

# ─── 7. Generate Report Plots ────────────────────────────────────────
print("Generating report plots...")

# Plot 1: Confusion Matrix
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    conf_matrix, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Anomaly", "Normal"],
    yticklabels=["Anomaly", "Normal"],
    annot_kws={"size": 16},
)
ax.set_title("Confusion Matrix: Isolation Forest\n(Real Normal Data vs Synthetic Anomalies)", fontsize=13)
ax.set_ylabel("Actual", fontsize=12)
ax.set_xlabel("Predicted", fontsize=12)
plt.tight_layout()
plt.savefig("report_assets/confusion_matrix.png", dpi=150)
print("  ✓ confusion_matrix.png")

# Plot 2: RPM vs Coolant Temp scatter with anomalies
full_preds = model.predict(scaler.transform(X))
df_pivot["anomaly"] = full_preds

fig, ax = plt.subplots(figsize=(10, 6))
normal_data = df_pivot[df_pivot["anomaly"] == 1]
anomaly_data = df_pivot[df_pivot["anomaly"] == -1]
ax.scatter(normal_data["engine_rpm"], normal_data["coolant_temp"],
           c="#4A90D9", label=f"Normal ({len(normal_data):,})", alpha=0.4, s=8)
ax.scatter(anomaly_data["engine_rpm"], anomaly_data["coolant_temp"],
           c="#E74C3C", label=f"Anomaly ({len(anomaly_data):,})", alpha=0.7, s=20, edgecolors="black", linewidth=0.3)
# Also plot the synthetic anomalies in a different marker
ax.scatter(synthetic_anomalies["engine_rpm"], synthetic_anomalies["coolant_temp"],
           c="#F39C12", marker="x", label=f"Synthetic ({len(synthetic_anomalies):,})", alpha=0.5, s=15)
ax.set_title("Engine RPM vs Coolant Temperature\n(Anomalies Highlighted)", fontsize=14)
ax.set_xlabel("Engine RPM", fontsize=12)
ax.set_ylabel("Coolant Temp (°C)", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("report_assets/anomaly_scatter.png", dpi=150)
print("  ✓ anomaly_scatter.png")

# Plot 3: Anomaly Score Distribution
fig, ax = plt.subplots(figsize=(10, 5))
all_scores = model.decision_function(scaler.transform(X))
ax.hist(all_scores, bins=80, color="#4A90D9", edgecolor="white", alpha=0.7, label="Real Data")
ax.hist(synth_scores, bins=40, color="#E74C3C", edgecolor="white", alpha=0.6, label="Synthetic Anomalies")
ax.axvline(0, color="black", ls="--", lw=2, label="Decision Boundary")
ax.set_xlabel("Anomaly Score (negative = more anomalous)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Isolation Forest: Anomaly Score Distribution", fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("report_assets/anomaly_score_distribution.png", dpi=150)
print("  ✓ anomaly_score_distribution.png")

# Plot 4: Feature importance via anomaly score correlation
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for idx, feat in enumerate(features):
    ax = axes[idx // 3][idx % 3]
    ax.scatter(X[feat], all_scores, alpha=0.3, s=5, c="#4A90D9")
    ax.set_xlabel(feat, fontsize=11)
    ax.set_ylabel("Anomaly Score", fontsize=10)
    ax.axhline(0, color="red", ls="--", alpha=0.5)
    ax.set_title(f"{feat}", fontsize=12)
# Hide the 6th subplot (we have 5 features)
axes[1][2].set_visible(False)
fig.suptitle("Feature vs Anomaly Score (below red line = anomalous)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("report_assets/feature_vs_score.png", dpi=150)
print("  ✓ feature_vs_score.png")

# ─── 8. Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
anomaly_pct = (full_preds == -1).sum() / len(full_preds) * 100
print(f"  Model: Isolation Forest (200 trees, 5% contamination)")
print(f"  Train samples: {len(X_train):,}")
print(f"  Test samples:  {len(X_test):,}")
print(f"  Real data anomalies detected: {(full_preds == -1).sum():,} / {len(full_preds):,} ({anomaly_pct:.1f}%)")
print(f"  Synthetic anomalies detected: {synth_detected} / {len(synthetic_anomalies)} ({synth_detected/len(synthetic_anomalies)*100:.1f}%)")
print("=" * 60)
print("\nAll plots saved to report_assets/")
