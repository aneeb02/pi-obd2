# Pi OBD2 Telemetry MVP

This project reads live OBD2 data from a Raspberry Pi connected to an ELM327 adapter, exposes telemetry via FastAPI, stores historical samples in CSV, and visualizes data in Grafana.

## MVP capabilities

- Poll core OBD2 PIDs (speed, RPM, coolant temp, throttle, engine load)
- Keep latest values in memory for low-latency API responses
- Append normalized telemetry samples to daily CSV files
- Expose REST + WebSocket telemetry APIs
- Render a starter Grafana dashboard backed by API endpoints

## Repo structure

- `backend/` FastAPI service, OBD polling, CSV persistence, tests
- `infra/grafana/` Grafana provisioning and dashboard JSON
- `data/` generated CSV telemetry output

## Quick start (development)

1. Create and activate a virtual environment:
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
2. Install dependencies:
  - `pip install -r backend/requirements.txt`
  - `pip install -r backend/requirements-dev.txt`
3. Configure environment:
  - `cp .env.example .env`
  - Update `OBD_PORT` if needed.
4. Run API:
  - `uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload`

Open:

- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

## Raspberry Pi runbook

### 1) Pair and bind ELM327 (Bluetooth)

1. Pair adapter via `bluetoothctl`.
2. Bind RFCOMM device:
  - `sudo rfcomm bind 0 <ADAPTER_MAC>`
3. Verify:
  - `ls /dev/rfcomm0`

Set `OBD_PORT=/dev/rfcomm0` in `.env`.

### 2) Boot and troubleshoot

- If no data appears, check adapter power and car ignition.
- Verify adapter connectivity with `python-OBD` in a Python shell.
- Check API logs for reconnect counters and timeout warnings.

## Next steps

- Replace CSV with a dedicated time-series database for scale.
- Add anomaly detection and LLM-assisted diagnostics over trip history.

