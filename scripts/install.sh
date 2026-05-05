#!/usr/bin/env bash
# ── Pi OBD2: Full Installation Script ──
# Installs all system dependencies, Python venv, Docker, and Grafana
# on a fresh Raspberry Pi OS (Bookworm / Bullseye).
#
# Usage:  sudo ./scripts/install.sh

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
REAL_HOME="$(eval echo "~$REAL_USER")"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     Pi OBD2 Telemetry — Installation Script     ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Project: $PROJECT_DIR"
echo "║  User:    $REAL_USER"
echo "╚══════════════════════════════════════════════════╝"
echo ""

[[ $EUID -ne 0 ]] && err "This script must be run as root (use sudo)."

# ── Step 1: System packages ──
info "Step 1/5: Installing system packages..."
apt update -qq
apt install -y -qq \
    python3 python3-venv python3-pip \
    bluetooth bluez rfcomm \
    git curl \
    > /dev/null 2>&1
log "System packages installed."

# ── Step 2: Python virtual environment ──
info "Step 2/5: Setting up Python virtual environment..."
cd "$PROJECT_DIR"

if [[ ! -d ".venv" ]]; then
    sudo -u "$REAL_USER" python3 -m venv .venv
    log "Virtual environment created at .venv/"
else
    warn "Virtual environment already exists, skipping."
fi

sudo -u "$REAL_USER" .venv/bin/pip install --upgrade pip -q
sudo -u "$REAL_USER" .venv/bin/pip install -r backend/requirements.txt -q
log "Python dependencies installed."

# ── Step 3: Environment file ──
info "Step 3/5: Configuring environment..."
if [[ ! -f ".env" ]]; then
    # Adapt the example to this system
    sed "s|/home/pi/pi-obd2|$PROJECT_DIR|g" .env.example > .env
    chown "$REAL_USER":"$REAL_USER" .env
    log "Created .env (edit OBD_PORT and DATA_DIR if needed)."
else
    warn ".env already exists, not overwriting."
fi

# Ensure data directory exists
DATA_DIR="${PROJECT_DIR}/data"
mkdir -p "$DATA_DIR"
chown -R "$REAL_USER":"$REAL_USER" "$DATA_DIR"
log "Data directory ready at $DATA_DIR"

# ── Step 4: Docker (for Grafana) ──
info "Step 4/5: Installing Docker..."
if command -v docker >/dev/null 2>&1; then
    warn "Docker already installed, skipping."
else
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sh /tmp/get-docker.sh > /dev/null 2>&1
    rm -f /tmp/get-docker.sh
    log "Docker installed."
fi

# Add user to docker group
if ! groups "$REAL_USER" | grep -q docker; then
    usermod -aG docker "$REAL_USER"
    log "Added $REAL_USER to docker group (re-login required)."
else
    warn "$REAL_USER already in docker group."
fi

# ── Step 5: Grafana ──
info "Step 5/5: Starting Grafana via Docker Compose..."
cd "$PROJECT_DIR/infra"
if command -v docker compose >/dev/null 2>&1; then
    sudo -u "$REAL_USER" docker compose up -d 2>/dev/null || docker compose up -d
    log "Grafana running on port 3000."
else
    warn "Docker Compose not available yet. Run 'docker compose up -d' after re-login."
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║           Installation Complete! 🎉              ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║                                                  ║"
echo "║  Next steps:                                     ║"
echo "║  1. Re-login for docker group to take effect     ║"
echo "║  2. Pair your Bluetooth scanner:                 ║"
echo "║     sudo ./scripts/setup-bluetooth.sh            ║"
echo "║  3. Test the API:                                ║"
echo "║     source .venv/bin/activate                    ║"
echo "║     uvicorn backend.app.main:app --host 0.0.0.0  ║"
echo "║  4. (Optional) Enable auto-start on boot:        ║"
echo "║     sudo ./scripts/enable-service.sh             ║"
echo "║                                                  ║"
echo "╚══════════════════════════════════════════════════╝"
