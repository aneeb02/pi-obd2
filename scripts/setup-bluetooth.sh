#!/usr/bin/env bash
# ── Pi OBD2: Bluetooth ELM327 Scanner Setup ──
# This script automates pairing, trusting, and binding an ELM327
# Bluetooth OBD-II adapter on Raspberry Pi OS.
#
# Usage:
#   sudo ./scripts/setup-bluetooth.sh            # interactive scan
#   sudo ./scripts/setup-bluetooth.sh AA:BB:CC:DD:EE:FF   # known MAC

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; exit 1; }
info() { echo -e "${CYAN}[i]${NC} $*"; }

# ── Pre-flight checks ──
[[ $EUID -ne 0 ]] && err "This script must be run as root (use sudo)."

command -v bluetoothctl >/dev/null 2>&1 || err "bluetoothctl not found. Install: sudo apt install bluez"
command -v rfcomm       >/dev/null 2>&1 || err "rfcomm not found. Install: sudo apt install bluez rfcomm"

# ── Ensure Bluetooth service is running ──
if ! systemctl is-active --quiet bluetooth; then
    warn "Bluetooth service not running, starting it..."
    systemctl start bluetooth
    sleep 2
fi

# ── Determine MAC address ──
MAC_ADDRESS="${1:-}"

if [[ -z "$MAC_ADDRESS" ]]; then
    info "No MAC address provided. Scanning for Bluetooth devices..."
    info "Make sure your ELM327 adapter is plugged in and powered on."
    echo ""

    # Scan for 10 seconds
    timeout 12 bluetoothctl scan on &
    SCAN_PID=$!
    sleep 10
    kill "$SCAN_PID" 2>/dev/null || true
    wait "$SCAN_PID" 2>/dev/null || true

    echo ""
    info "Scan complete. Listing discovered devices:"
    echo ""
    bluetoothctl devices | nl -ba
    echo ""

    read -rp "Enter the MAC address of your ELM327 (e.g. AA:BB:CC:DD:EE:FF): " MAC_ADDRESS
    [[ -z "$MAC_ADDRESS" ]] && err "No MAC address entered."
fi

# Validate MAC format
if ! [[ "$MAC_ADDRESS" =~ ^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$ ]]; then
    err "Invalid MAC address format: $MAC_ADDRESS"
fi

MAC_ADDRESS="${MAC_ADDRESS^^}"  # uppercase
log "Using MAC address: $MAC_ADDRESS"

# ── Pair ──
info "Pairing with $MAC_ADDRESS..."
bluetoothctl pair "$MAC_ADDRESS" 2>/dev/null && log "Paired successfully." || warn "Already paired or pairing failed (continuing...)"

# ── Trust ──
info "Trusting $MAC_ADDRESS..."
bluetoothctl trust "$MAC_ADDRESS" 2>/dev/null && log "Trusted successfully." || warn "Already trusted."

# ── Unbind any existing rfcomm0 ──
if [[ -e /dev/rfcomm0 ]]; then
    warn "/dev/rfcomm0 already exists, releasing it first..."
    rfcomm release 0 2>/dev/null || true
    sleep 1
fi

# ── Bind RFCOMM ──
info "Binding RFCOMM channel 0 to $MAC_ADDRESS..."
rfcomm bind 0 "$MAC_ADDRESS" 1

if [[ -e /dev/rfcomm0 ]]; then
    log "Success! /dev/rfcomm0 is ready."
else
    err "/dev/rfcomm0 was not created. Check adapter power and try again."
fi

# ── Write MAC to a reference file for the systemd service ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
echo "$MAC_ADDRESS" > "$PROJECT_DIR/.bt_mac"
log "Saved MAC address to $PROJECT_DIR/.bt_mac"

echo ""
log "Bluetooth setup complete!"
info "Your OBD-II scanner is available at /dev/rfcomm0"
info "Next: update OBD_PORT=/dev/rfcomm0 in your .env file and start the API."
