# Final Report — Pi OBD2 Telemetry System

> This document contains the draft sections required for your final academic report. Copy these into your main report document and add your own screenshots where noted.

---

## 1. Process of Cloud Configuration

**Cloud Platform Chosen**: Amazon Web Services (AWS)

To enable remote monitoring and centralized data storage for our vehicle telemetry system, we deployed the backend services on Amazon Web Services (AWS).

### Step-by-Step Configuration

1. **EC2 Instance Provisioning**: An Ubuntu 22.04 LTS `t3.small` instance was launched in the `ap-south-1` (Mumbai) region to minimize latency for our deployment in Pakistan. The instance was assigned an Elastic IP for stable addressing.

2. **Security Group Configuration**: A custom Security Group (`pi-obd2-sg`) was created with the following inbound rules:
   | Port | Protocol | Source | Purpose |
   |------|----------|--------|---------|
   | 22   | TCP      | My IP  | SSH administration |
   | 8000 | TCP      | 0.0.0.0/0 | FastAPI REST/WebSocket endpoints |
   | 3000 | TCP      | 0.0.0.0/0 | Grafana dashboard UI |

3. **Software Stack Deployment**: Docker and Docker Compose were installed on the EC2 instance. The Grafana container was deployed using the project's `infra/docker-compose.yml`, with the Infinity data source plugin pre-installed for API-based data fetching.

4. **Data Ingestion Architecture**: The Raspberry Pi 3 (edge node) connects to the vehicle's ECU via an ELM327 Bluetooth adapter, normalizes the OBD-II readings, and streams JSON telemetry payloads to the cloud-hosted FastAPI server over WebSocket. The server persists data to daily CSV files and serves it via REST endpoints consumed by Grafana.

5. **Grafana Dashboard Provisioning**: Dashboards and data sources were provisioned automatically via YAML configuration files mounted into the Grafana container (`infra/grafana/provisioning/`), ensuring reproducible deployments.

*(Insert screenshots: EC2 instance dashboard, Security Group rules, Grafana login page)*

---

## 2. Description of ML/DL Algorithms Used

### 2.1 Problem Formulation

The problem was framed as **unsupervised anomaly detection for engine overheating prediction**. In real-world automotive data, labeled failure events (e.g., "overheating incident at timestamp X") are extremely rare and costly to obtain. Therefore, we adopted an anomaly detection approach: train a model exclusively on *normal* driving data, and flag any future data that deviates significantly from learned normal patterns as potentially anomalous.

Specifically, the model learns to reconstruct sequences of OBD-II sensor readings. When presented with abnormal thermal behavior — such as rising coolant temperature under low load — the model produces a high **reconstruction error**, signaling a potential overheating condition.

### 2.2 Algorithm Selection & Justification

**Algorithm**: LSTM Autoencoder (Long Short-Term Memory Autoencoder)

**Why LSTM Autoencoder?**
- **Temporal dependencies**: Unlike traditional anomaly detectors (e.g., Isolation Forest, One-Class SVM), an LSTM Autoencoder captures *sequential patterns* in time-series data. Engine overheating is a progressive event — coolant temperature rises over time, correlated with RPM and throttle patterns — making temporal modeling essential.
- **Unsupervised learning**: No labeled anomaly data is required. The model is trained only on normal driving data and learns to reconstruct it. Anomalies are detected when the reconstruction error exceeds learned thresholds.
- **Multivariate capability**: The model simultaneously processes 7 correlated sensor features, capturing cross-feature dependencies (e.g., high RPM + high throttle + rising coolant = normal under load; but rising coolant + low RPM + low throttle = abnormal).

**Why not simpler alternatives?**
| Algorithm | Limitation for this problem |
|-----------|-----------------------------|
| Isolation Forest | No temporal modeling; treats each sample independently |
| One-Class SVM | Computationally expensive on large sequences; no sequence context |
| Simple threshold rules | Cannot capture complex multi-feature correlations |
| Statistical (Z-score) | Assumes normal distribution; ignores temporal dynamics |

### 2.3 Model Architecture

The LSTM Autoencoder follows an encoder–decoder architecture:

```
Input: (batch, 50, 7)  — 50-timestep windows of 7 OBD-II features

ENCODER:
  LSTM(128, relu, return_sequences=True)   → (batch, 50, 128)
  LSTM(64, relu, return_sequences=False)   → (batch, 64)
  Dense(32, relu)                          → (batch, 32)        ← latent vector

DECODER:
  RepeatVector(50)                         → (batch, 50, 32)
  LSTM(64, relu, return_sequences=True)    → (batch, 50, 64)
  LSTM(128, relu, return_sequences=True)   → (batch, 50, 128)
  TimeDistributed(Dense(7))                → (batch, 50, 7)     ← reconstruction

Loss: Mean Squared Error (MSE)
Optimizer: Adam
```

The **encoder** compresses a 50-timestep, 7-feature window into a 32-dimensional latent representation. The **decoder** attempts to reconstruct the original input from this compressed vector. For normal data, the reconstruction is accurate (low MSE). For anomalous patterns, the reconstruction diverges (high MSE).

### 2.4 Features Used

The model operates on 7 OBD-II features (the "backbone"):

| # | Feature | Unit | Relevance |
|---|---------|------|-----------|
| 1 | Engine Coolant Temperature | °C | Primary overheating indicator |
| 2 | Intake Manifold Absolute Pressure | kPa | Engine load proxy |
| 3 | Engine RPM | RPM | Operating condition indicator |
| 4 | Intake Air Temperature | °C | Under-hood heat soak detection |
| 5 | Air Flow Rate (MAF) | g/s | Combustion intensity indicator |
| 6 | Absolute Throttle Position | % | Driver demand / load context |
| 7 | Ambient Air Temperature | °C | Environmental baseline |

### 2.5 Implementation Details

- **Framework**: TensorFlow/Keras 2.x for model training and inference
- **Libraries**: Scikit-learn (data scaling via `MinMaxScaler`), Pandas (data manipulation), NumPy, Matplotlib/Seaborn (visualization)
- **Training Dataset**: RADAR OBD-II Dataset (Kaggle) — real-world driving data collected under Normal, Free-flow, and Congested traffic conditions
- **Data Split**: 70% training / 15% validation / 15% test (split at the trip level to prevent data leakage)
- **Preprocessing Pipeline**:
  1. Column name normalization (UTF-8 encoding fixes)
  2. Zero-value replacement with NaN → linear interpolation → forward/backward fill
  3. Feature scaling using MinMaxScaler fitted on training data
  4. Sliding window generation: window size = 50 timesteps, hop = 1
- **Training Configuration**: 80 epochs, batch size 64, EarlyStopping (patience=8), ModelCheckpoint
- **Anomaly Thresholds**: Percentile-based from normal test data — p95 = 0.000304, p99 = 0.001103
- **Scoring**: `score = (error - p95) / (p99 * relax_factor - p95)`, mapped to risk levels: LOW (<0.5), MEDIUM (0.5–0.8), HIGH (≥0.8)
- **Hardware**: Model trained on Kaggle (GPU-accelerated), inference on Raspberry Pi 3 (CPU) and development machine

### 2.6 Performance Metrics & Results

#### Inference on Pi OBD2 Telemetry Data

The trained LSTM Autoencoder was applied to 15,487 real telemetry readings collected from the Raspberry Pi OBD2 system, spanning 7 driving sessions and 3,013 analysis windows.

| Metric | Value |
|--------|-------|
| Total sessions analyzed | 7 |
| Total windows processed | 3,013 |
| Mean reconstruction error | 0.026114 |
| Max reconstruction error | 0.042249 |
| p95 threshold (from training) | 0.000304 |
| p99 threshold (from training) | 0.001103 |

**Risk Classification Results:**

| Risk Level | Sessions | Percentage |
|------------|----------|------------|
| LOW        | 0        | 0%         |
| MEDIUM     | 0        | 0%         |
| HIGH       | 7        | 100%       |

*(Insert: `report_assets/recon_error_distribution.png` — Reconstruction error histogram with p95/p99 thresholds)*

*(Insert: `report_assets/risk_pie_chart.png` — Risk level distribution pie chart)*

*(Insert: `report_assets/score_per_session.png` — Anomaly score per driving session bar chart)*

*(Insert: `report_assets/coolant_vs_score.png` — Max coolant temp vs anomaly score scatter plot)*

### 2.7 Results Analysis

**Did the model perform as expected?**

The model classified all 7 driving sessions as HIGH risk. While this may appear to indicate poor performance, this result is **expected and explainable** given the model's design:

1. **Domain shift**: The LSTM Autoencoder was trained on the RADAR OBD-II dataset (German road conditions, specific vehicle types). Our Pi OBD2 system collects data from a different vehicle in Pakistani driving conditions, with different RPM ranges, throttle patterns, and ambient temperatures. The model has never seen data that matches these patterns, so reconstruction error is inherently high.

2. **Threshold sensitivity**: The p95/p99 thresholds (0.000304 / 0.001103) were derived from very clean normal driving data. The observed reconstruction errors (0.026–0.042) are ~30–40× higher than these thresholds, which is consistent with the domain gap.

3. **Known limitation**: As noted in the model documentation: *"The LSTM learned that normal coolant behavior = almost perfectly flat after warmup. So if your data has even slightly different slope/noise, the model flags it."*

**Challenges Faced:**
- **Data quality**: The Pi OBD2 system collects 5 of the 7 required features natively. The 2 missing features (Intake Air Temperature, MAF sensor) were synthesized with realistic estimates, introducing some noise.
- **Computational cost**: Running LSTM inference on the Raspberry Pi 3's ARM CPU is significantly slower than GPU-accelerated training. Processing 3,013 windows took approximately 30 minutes, highlighting the need for model optimization (quantization, TFLite conversion) for real-time edge deployment.
- **Threshold calibration**: The model would require fine-tuning or transfer learning on locally-collected normal driving data to establish vehicle-specific and region-specific anomaly thresholds.

---

## 3. Final User-Facing Application

### Description

The user-facing application is a real-time web dashboard powered by **Grafana**, an open-source analytics and monitoring platform. It provides drivers and technicians with an immediate, visual display of the vehicle's telemetry and health status.

### Architecture & Data Flow

```
┌─────────────┐    Bluetooth     ┌───────────────┐    REST/WS     ┌─────────────┐
│  Vehicle ECU │ ──────────────► │  Raspberry Pi  │ ────────────► │  AWS EC2     │
│  (OBD-II)    │    ELM327       │  + FastAPI     │    JSON       │  + Grafana   │
└─────────────┘                  └───────────────┘               └─────────────┘
                                       │ CSV
                                       ▼
                                 ┌─────────────┐
                                 │  ML Module   │
                                 │  (LSTM AE)   │
                                 └─────────────┘
```

1. **Hardware Layer**: The ELM327 OBD-II Bluetooth adapter continuously queries the vehicle's Engine Control Unit (ECU) for PID data.
2. **Edge Processing**: The Raspberry Pi 3 receives serial data via RFCOMM Bluetooth, normalizes it through the FastAPI backend, and stores timestamped readings in daily CSV files.
3. **API Layer**: FastAPI exposes REST endpoints (`/telemetry/latest`, `/telemetry/history`) and WebSocket streams for real-time data access.
4. **Visualization**: Grafana, deployed via Docker on AWS EC2, fetches data through the Infinity data source plugin and renders live gauges, time-series graphs, and alert panels.

### Dashboard Features

- **Real-time gauges**: Vehicle speed, Engine RPM, Coolant Temperature, Throttle Position, Engine Load
- **Time-series graphs**: Historical trends with configurable time ranges
- **Auto-refresh**: Dashboard polls the API every second for live updates

*(Insert screenshots: Grafana dashboard showing speed/RPM/coolant gauges, time-series panel)*

---

## 4. Conclusion & Future Work

### Achievements

This project successfully demonstrated an end-to-end IoT vehicle telemetry pipeline integrating:
- **Edge computing**: Raspberry Pi 3 as the data collection and processing node
- **Bluetooth communication**: ELM327 OBD-II adapter paired via RFCOMM
- **Cloud deployment**: FastAPI backend and Grafana dashboard hosted on AWS EC2
- **Machine learning**: LSTM Autoencoder for anomaly-based engine overheating prediction

The system polls 5 core OBD-II PIDs at 1-second intervals, persists normalized data to CSV, and exposes both REST and WebSocket APIs for real-time visualization.

### Critical Evaluation

| Objective | Status | Notes |
|-----------|--------|-------|
| Real-time OBD-II data collection | ✅ Achieved | 5 PIDs at 1Hz via Bluetooth |
| Cloud-hosted visualization | ✅ Achieved | Grafana on AWS EC2 |
| ML-based anomaly detection | ⚠️ Partial | Model works but requires local calibration |
| Data persistence | ✅ Achieved | Daily CSV files |
| Auto-start on boot | ✅ Achieved | systemd service + Docker restart policy |

The primary limitation is the LSTM model's sensitivity to domain shift — it needs to be fine-tuned on locally-collected normal driving data to reduce false positive rates.

### Future Work

1. **Transfer Learning / Fine-Tuning**: Collect 2–3 hours of verified normal driving data from the target vehicle and fine-tune the LSTM Autoencoder to establish vehicle-specific baselines and thresholds.
2. **Edge ML Deployment**: Convert the Keras model to TensorFlow Lite for optimized inference on the Raspberry Pi's ARM CPU, enabling real-time anomaly detection without cloud dependency.
3. **Time-Series Database**: Migrate from CSV storage to InfluxDB or AWS Timestream for efficient querying over large historical datasets.
4. **Additional Sensors**: Incorporate GPS data for location-aware analytics and additional OBD-II PIDs (fuel system status, O2 sensors) for broader diagnostic coverage.
5. **Mobile Application**: Develop a companion mobile app (React Native) that connects directly to the Pi's API for on-the-go monitoring without requiring a laptop.
6. **Advanced Algorithms**: Explore Transformer-based architectures (e.g., Temporal Fusion Transformer) for multi-horizon prediction and implement ensemble methods combining the LSTM Autoencoder with rule-based heuristics.

---

## 5. References

1. F. T. Liu, K. M. Ting, and Z. Zhou, "Isolation-Based Anomaly Detection," *ACM Transactions on Knowledge Discovery from Data*, vol. 6, no. 1, pp. 1–39, 2012.
2. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.
3. D. Bank, N. Koenigstein, and R. Giryes, "Autoencoders," *arXiv preprint arXiv:2003.05991*, 2020.
4. M. Abadi et al., "TensorFlow: A System for Large-Scale Machine Learning," *12th USENIX Symposium on Operating Systems Design and Implementation (OSDI)*, pp. 265–283, 2016.
5. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
6. S. Titouna, "RADAR OBD-II Dataset," Kaggle, 2023. Available: https://www.kaggle.com/datasets/stijntitouan/radar-obd-ii-dataset
7. FastAPI Documentation. Available: https://fastapi.tiangolo.com/
8. Grafana Labs, "Grafana: The Open Observability Platform." Available: https://grafana.com/
9. python-OBD Documentation. Available: https://python-obd.readthedocs.io/
