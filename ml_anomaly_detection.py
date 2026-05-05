import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# 1. Load Data
data_file = 'data/2026-04-27.csv'
print(f"Loading data from {data_file}...")
df = pd.read_csv(data_file)

# 2. Preprocess Data
# Pivot the table so each row is a timestamp and columns are the different PIDs
df['timestamp'] = pd.to_datetime(df['timestamp'])
df_pivot = df.pivot_table(index='timestamp', columns='name', values='value').dropna()

# Select features for anomaly detection (Engine RPM and Coolant Temp are good indicators)
features = ['engine_rpm', 'coolant_temp', 'throttle_pos', 'engine_load']
X = df_pivot[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train Isolation Forest Model
print("Training Isolation Forest Model...")
# Isolation Forest is an unsupervised anomaly detection algorithm
model = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
model.fit(X_scaled)

# Predict anomalies (-1 for anomaly, 1 for normal)
predictions = model.predict(X_scaled)
df_pivot['anomaly'] = predictions

# For reporting metrics, let's create a synthetic ground truth 
# (assuming extreme high coolant temp or extreme RPM as true anomalies for evaluation purposes)
df_pivot['true_anomaly'] = np.where((df_pivot['coolant_temp'] > 100) | (df_pivot['engine_rpm'] > 4500), -1, 1)

# 4. Generate Performance Metrics
print("\n--- Model Performance Metrics ---")
print(classification_report(df_pivot['true_anomaly'], df_pivot['anomaly'], target_names=['Anomaly', 'Normal']))

conf_matrix = confusion_matrix(df_pivot['true_anomaly'], df_pivot['anomaly'])

# 5. Generate Plots for the Report
os.makedirs("report_assets", exist_ok=True)

# Plot 1: Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Anomaly', 'Normal'], yticklabels=['Anomaly', 'Normal'])
plt.title('Confusion Matrix: Anomaly Detection')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('report_assets/confusion_matrix.png')
print("Saved Confusion Matrix to report_assets/confusion_matrix.png")

# Plot 2: Scatter plot of RPM vs Coolant Temp highlighting anomalies
plt.figure(figsize=(10, 6))
normal_data = df_pivot[df_pivot['anomaly'] == 1]
anomaly_data = df_pivot[df_pivot['anomaly'] == -1]

plt.scatter(normal_data['engine_rpm'], normal_data['coolant_temp'], c='blue', label='Normal', alpha=0.5, s=10)
plt.scatter(anomaly_data['engine_rpm'], anomaly_data['coolant_temp'], c='red', label='Anomaly', alpha=0.7, s=20)
plt.title('Engine RPM vs Coolant Temperature (Anomalies Highlighted)')
plt.xlabel('Engine RPM')
plt.ylabel('Coolant Temp (°C)')
plt.legend()
plt.tight_layout()
plt.savefig('report_assets/anomaly_scatter.png')
print("Saved Anomaly Scatter Plot to report_assets/anomaly_scatter.png")

print("\nAll done! You can use the generated images in your report.")
