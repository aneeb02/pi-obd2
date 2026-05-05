#!/usr/bin/env bash
# ── Pi OBD2: Enable Systemd Auto-Start Service ──
# Creates and enables a systemd service so the OBD2 API starts
# automatically on every boot.
#
# Usage:  sudo ./scripts/enable-service.sh

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; exit 1; }
info() { echo -e "${CYAN}[i]${NC} $*"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REAL_USER="${SUDO_USER:-$USER}"
SERVICE_NAME="obd2-api"

[[ $EUID -ne 0 ]] && err "This script must be run as root (use sudo)."

# ── Read saved Bluetooth MAC ──
BT_MAC=""
if [[ -f "$PROJECT_DIR/.bt_mac" ]]; then
    BT_MAC="$(cat "$PROJECT_DIR/.bt_mac" | tr -d '[:space:]')"
    log "Found saved Bluetooth MAC: $BT_MAC"
else
    warn "No .bt_mac file found. Run setup-bluetooth.sh first, or enter MAC manually."
    read -rp "Enter ELM327 MAC address (or press Enter to skip RFCOMM bind): " BT_MAC
fi

# ── Build the ExecStartPre line ──
RFCOMM_LINE=""
if [[ -n "$BT_MAC" ]]; then
    RFCOMM_LINE="ExecStartPre=/bin/bash -c '/usr/bin/rfcomm release 0 2>/dev/null; /usr/bin/rfcomm bind 0 $BT_MAC 1'"
fi

# ── Create the systemd service ──
info "Creating /etc/systemd/system/${SERVICE_NAME}.service..."

cat > "/etc/systemd/system/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=Pi OBD2 Telemetry API
After=network.target bluetooth.target
Wants=bluetooth.target

[Service]
Type=simple
User=$REAL_USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/.venv/bin:/usr/local/bin:/usr/bin:/bin"
${RFCOMM_LINE}
ExecStart=$PROJECT_DIR/.venv/bin/uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

log "Service file created."

# ── Enable and start ──
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl start "$SERVICE_NAME"

log "Service '$SERVICE_NAME' enabled and started!"

echo ""
info "Useful commands:"
echo "  sudo systemctl status $SERVICE_NAME    # Check status"
echo "  sudo journalctl -u $SERVICE_NAME -f    # Follow live logs"
echo "  sudo systemctl restart $SERVICE_NAME   # Restart the service"
echo "  sudo systemctl stop $SERVICE_NAME      # Stop the service"
