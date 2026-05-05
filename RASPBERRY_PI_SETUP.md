# 🚗 Raspberry Pi 3 Deployment Guide — OBD2 Telemetry

Complete guide for deploying the Pi OBD2 Telemetry system on a Raspberry Pi 3 with a Bluetooth ELM327 OBD-II scanner.

---

## Prerequisites

| Item | Details |
|------|---------|
| **Raspberry Pi 3** | Running Raspberry Pi OS (Bookworm or Bullseye) |
| **ELM327 Bluetooth Adapter** | Plugged into the car's OBD-II port |
| **Internet connection** | Wi-Fi or Ethernet on the Pi (for initial setup) |
| **Car with OBD-II port** | Ignition must be ON for the scanner to power up |

---

## Quick Start (Automated)

If you want to get up and running fast, we provide scripts that handle everything:

```bash
# 1. Get the code onto your Pi (pick one method)
# Option A: Clone from GitHub
git clone <your-repo-url> ~/pi-obd2
# Option B: Copy from your computer
scp -r /path/to/pi-obd2 pi@<PI_IP>:~/pi-obd2

# 2. SSH into the Pi
ssh pi@<PI_IP>

# 3. Run the installer (installs Python, Docker, dependencies)
cd ~/pi-obd2
sudo ./scripts/install.sh

# 4. Re-login (required for Docker group permissions)
exit
ssh pi@<PI_IP>

# 5. Pair your Bluetooth OBD-II scanner
cd ~/pi-obd2
sudo ./scripts/setup-bluetooth.sh

# 6. Test the API
source .venv/bin/activate
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

# 7. (Optional) Enable auto-start on boot
sudo ./scripts/enable-service.sh
```

That's it! The rest of this document explains each step in detail.

---

## Step-by-Step Manual Setup

### Phase 1: Get the Code onto the Pi

**Option A — Clone from GitHub (recommended):**
```bash
ssh pi@<YOUR_PI_IP>
git clone <your-repo-url> ~/pi-obd2
cd ~/pi-obd2
```

**Option B — Copy via SCP (from your dev machine):**
```bash
scp -r /home/dev/Desktop/pi-obd2 pi@<YOUR_PI_IP>:~/pi-obd2
```

---

### Phase 2: Install System Dependencies

```bash
# Update the system
sudo apt update && sudo apt upgrade -y

# Install Python, Bluetooth tools, and utilities
sudo apt install -y python3 python3-venv python3-pip bluetooth bluez rfcomm git curl
```

---

### Phase 3: Set Up Python Environment

```bash
cd ~/pi-obd2

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install project dependencies
pip install --upgrade pip
pip install -r backend/requirements.txt
```

---

### Phase 4: Pair the ELM327 Bluetooth Scanner

> **Important**: Make sure the ELM327 adapter is plugged into the car's OBD-II port and the car ignition is turned ON. The adapter needs power from the car to be discoverable.

#### Automated Method

```bash
sudo ./scripts/setup-bluetooth.sh
```

This script will scan for devices, let you pick your ELM327, pair it, trust it, and bind it to `/dev/rfcomm0`.

#### Manual Method

1. **Start the Bluetooth control tool:**
   ```bash
   bluetoothctl
   ```

2. **Inside `bluetoothctl`, run these commands:**
   ```
   agent on
   default-agent
   scan on
   ```

3. **Wait for your scanner to appear.** It will show up as something like:
   ```
   [NEW] Device AA:BB:CC:DD:EE:FF OBDII
   ```
   Note the MAC address (`AA:BB:CC:DD:EE:FF`).

4. **Pair and trust:**
   ```
   scan off
   pair AA:BB:CC:DD:EE:FF
   ```
   > If prompted for a PIN, enter `1234` (most common) or `0000`.

   ```
   trust AA:BB:CC:DD:EE:FF
   exit
   ```

5. **Bind to a serial port:**
   ```bash
   sudo rfcomm bind 0 AA:BB:CC:DD:EE:FF
   ```

6. **Verify the device exists:**
   ```bash
   ls -la /dev/rfcomm0
   ```
   You should see the device file. This is the serial port your application will use.

---

### Phase 5: Configure the Environment

```bash
cd ~/pi-obd2

# Create your .env from the template
cp .env.example .env

# Edit with your settings
nano .env
```

The critical settings to verify:

```ini
OBD_PORT=/dev/rfcomm0
DATA_DIR=/home/pi/pi-obd2/data
```

Make sure `DATA_DIR` matches your actual project path and that `OBD_PORT` points to the RFCOMM device you created in Phase 4.

---

### Phase 6: Test the API

```bash
cd ~/pi-obd2
source .venv/bin/activate

# Start the API server
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

**Verify it's working:**

| Check | URL |
|-------|-----|
| Health endpoint | `http://<PI_IP>:8000/health` |
| Latest telemetry | `http://<PI_IP>:8000/telemetry/latest` |
| API docs (Swagger) | `http://<PI_IP>:8000/docs` |
| Live dashboard | `http://<PI_IP>:8000/` |

The health endpoint should return:
```json
{
  "status": "ok",
  "connected": true,
  "reconnect_count": 0,
  "poll_errors": 0,
  "uptime_seconds": 12.34
}
```

If `connected` is `true`, the Pi is successfully reading data from your car via Bluetooth! Press `Ctrl+C` to stop the test.

> **Note:** If `connected` shows `false`, the app will automatically fall into simulation mode and generate fake data. Check the troubleshooting section below.

---

### Phase 7: Install Docker & Start Grafana

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh

# Add your user to the docker group
sudo usermod -aG docker $USER

# IMPORTANT: Log out and back in for group change to take effect
exit
ssh pi@<PI_IP>

# Start Grafana
cd ~/pi-obd2/infra
docker compose up -d
```

Access Grafana at `http://<PI_IP>:3000`
- Default username: `admin`
- Default password: `admin`

The dashboard is pre-provisioned to pull data from the FastAPI backend.

---

### Phase 8: Enable Auto-Start on Boot (Recommended)

So the system starts automatically every time the Pi powers on:

```bash
sudo ./scripts/enable-service.sh
```

This creates a systemd service that:
1. Binds the Bluetooth RFCOMM device on boot
2. Starts the FastAPI OBD2 telemetry server
3. Auto-restarts if the service crashes

Grafana is already configured with `restart: always` in Docker Compose, so it will also auto-start.

**Useful service commands:**
```bash
sudo systemctl status obd2-api       # Check if running
sudo journalctl -u obd2-api -f       # Watch live logs
sudo systemctl restart obd2-api      # Restart
sudo systemctl stop obd2-api         # Stop
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Raspberry Pi 3                        │
│                                                         │
│  ┌──────────────┐    Bluetooth     ┌───────────────┐   │
│  │   ELM327     │◄───(RFCOMM)────►│  OBD Reader   │   │
│  │  (in car)    │    /dev/rfcomm0  │  (python-OBD) │   │
│  └──────────────┘                  └───────┬───────┘   │
│                                            │            │
│                                    ┌───────▼───────┐   │
│                                    │ PID Scheduler │   │
│                                    │  (async loop) │   │
│                                    └──┬─────────┬──┘   │
│                                       │         │       │
│                              ┌────────▼──┐ ┌───▼────┐  │
│                              │ CSV Store │ │FastAPI │  │
│                              │  (daily   │ │ REST + │  │
│                              │   files)  │ │   WS   │  │
│                              └───────────┘ └───┬────┘  │
│                                                │       │
│                                        ┌───────▼────┐  │
│                                        │  Grafana   │  │
│                                        │ (Docker)   │  │
│                                        │ port 3000  │  │
│                                        └────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Data flow:**
1. **ELM327** plugs into the car's OBD-II port and communicates via Bluetooth
2. **OBD Reader** connects via `/dev/rfcomm0` using the `python-OBD` library
3. **PID Scheduler** polls 5 PIDs every second (speed, RPM, coolant temp, throttle, engine load)
4. **CSV Store** saves each reading to daily CSV files in `data/`
5. **FastAPI** exposes REST endpoints + WebSocket for real-time streaming
6. **Grafana** visualizes the telemetry on a dashboard

---

## Troubleshooting

### Bluetooth won't pair

```bash
# Check if Bluetooth service is running
sudo systemctl status bluetooth

# Restart Bluetooth
sudo systemctl restart bluetooth

# Re-scan
bluetoothctl
> scan on
```

Make sure the car ignition is ON — the ELM327 gets its power from the OBD-II port.

### `/dev/rfcomm0` doesn't appear after reboot

The RFCOMM binding doesn't persist across reboots by default. That's why the systemd service (Phase 8) includes an `ExecStartPre` that re-binds it on every boot. If you haven't set up the service:

```bash
sudo rfcomm bind 0 <YOUR_MAC_ADDRESS>
```

### API shows `connected: false`

1. Check the RFCOMM device: `ls -la /dev/rfcomm0`
2. Check the `.env` file has `OBD_PORT=/dev/rfcomm0`
3. Verify the car ignition is ON
4. Try reconnecting:
   ```bash
   sudo rfcomm release 0
   sudo rfcomm bind 0 <MAC_ADDRESS>
   sudo systemctl restart obd2-api
   ```

### Grafana can't reach the API

The Docker container accesses the host via `host.docker.internal`. Verify:
```bash
# Check API is running
curl http://localhost:8000/health

# Check Docker networking
docker exec pi-obd2-grafana wget -qO- http://host.docker.internal:8000/health
```

### Docker permission denied

```bash
# Make sure your user is in the docker group
sudo usermod -aG docker $USER
# Then log out and back in
```

---

## File Structure

```
pi-obd2/
├── backend/
│   ├── app/
│   │   ├── api/routes/telemetry.py   # REST & WebSocket endpoints
│   │   ├── models/schemas.py         # Pydantic data models
│   │   ├── services/
│   │   │   ├── obd_reader.py         # Bluetooth OBD connection
│   │   │   ├── pid_scheduler.py      # Async polling loop
│   │   │   └── csv_store.py          # Daily CSV persistence
│   │   ├── static/index.html         # Live dashboard UI
│   │   ├── config.py                 # Pydantic settings
│   │   └── main.py                   # FastAPI app entry point
│   └── requirements.txt
├── infra/
│   ├── docker-compose.yml            # Grafana container
│   └── grafana/                      # Dashboard provisioning
├── ml/                               # LSTM Autoencoder model
│   ├── lstm_artifacts/               # Trained model weights
│   ├── inference.py                  # Prediction pipeline
│   └── run_inference_on_csv.py       # Batch analysis script
├── data/                             # Generated CSV telemetry
├── scripts/
│   ├── install.sh                    # Full system setup
│   ├── setup-bluetooth.sh            # Bluetooth pairing helper
│   └── enable-service.sh             # Systemd auto-start setup
├── .env.example                      # Environment template
└── README.md
```
