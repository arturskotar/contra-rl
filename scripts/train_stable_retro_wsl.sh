#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv-retro-wsl}"
CONFIG="${CONFIG:-./configs/ppo_stable_retro.yaml}"
ROM="${ROM:-./roms/Contra.nes}"
RUN_NAME="${RUN_NAME:-retro_long_$(date +%Y%m%d_%H%M%S)}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2000000}"
N_ENVS="${N_ENVS:-8}"
DEVICE="${DEVICE:-cuda}"
METRICS_LOG_FREQ="${METRICS_LOG_FREQ:-1000}"
TB_PORT="${TB_PORT:-6006}"
START_TENSORBOARD="${START_TENSORBOARD:-1}"
INSTALL_DEPS="${INSTALL_DEPS:-auto}"
TORCH_INSTALL_COMMAND="${TORCH_INSTALL_COMMAND:-python -m pip install torch torchvision torchaudio}"

log() {
  printf "\n[%s] %s\n" "$(date +%H:%M:%S)" "$*"
}

python_import_ok() {
  local module="$1"
  python - "$module" <<'PY'
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)
PY
}

port_is_free() {
  python - "$TB_PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
try:
    sock.bind(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
sys.exit(0)
PY
}

if ! grep -qiE "(microsoft|wsl)" /proc/version 2>/dev/null; then
  log "Warning: this script is intended for WSL/Linux."
fi

if [ ! -d "$VENV_DIR" ]; then
  log "Creating virtual environment: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  INSTALL_DEPS=1
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

log "Python: $(python --version)"
log "Project: $ROOT_DIR"
log "Run name: $RUN_NAME"
log "Config: $CONFIG"
log "ROM: $ROM"
log "Timesteps: $TOTAL_TIMESTEPS"
log "Parallel envs: $N_ENVS"
log "Device: $DEVICE"

if [ ! -f "$ROM" ]; then
  log "Missing ROM: $ROM"
  exit 1
fi

if [ ! -f "$CONFIG" ]; then
  log "Missing config: $CONFIG"
  exit 1
fi

if [ "$INSTALL_DEPS" = "auto" ]; then
  if python_import_ok stable_retro && python_import_ok torch && python_import_ok stable_baselines3; then
    INSTALL_DEPS=0
  else
    INSTALL_DEPS=1
  fi
fi

if [ "$INSTALL_DEPS" = "1" ]; then
  log "Installing/updating Python dependencies"
  python -m pip install --upgrade pip setuptools wheel
  log "Installing PyTorch with: $TORCH_INSTALL_COMMAND"
  bash -lc "$TORCH_INSTALL_COMMAND"
  python -m pip install -e ".[dev,stable-retro]"
fi

log "Checking CUDA"
python - "$DEVICE" <<'PY'
import sys

import torch

device = sys.argv[1]
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

if device == "cuda" and not torch.cuda.is_available():
    raise SystemExit("DEVICE=cuda was requested, but PyTorch cannot see CUDA.")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  log "nvidia-smi"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader
fi

log "Importing ROM into project Stable Retro integration if needed"
python -m contra_rl.cli retro-import-rom \
  --rom "$ROM" \
  --integration-path ./integrations \
  --game Contra-Nes

log "Checking Stable Retro integration"
python -m contra_rl.cli retro-status \
  --game Contra-Nes \
  --integration-path ./integrations

mkdir -p runs

if [ "$START_TENSORBOARD" = "1" ]; then
  if port_is_free; then
    TB_LOG="runs/tensorboard_${TB_PORT}.log"
    log "Starting TensorBoard: http://localhost:${TB_PORT}"
    nohup python -m tensorboard.main \
      --logdir ./runs \
      --host 0.0.0.0 \
      --port "$TB_PORT" \
      > "$TB_LOG" 2>&1 &
    echo "$!" > "runs/tensorboard_${TB_PORT}.pid"
    log "TensorBoard pid: $(cat "runs/tensorboard_${TB_PORT}.pid")"
    log "TensorBoard log: $TB_LOG"
  else
    log "TensorBoard port ${TB_PORT} is already in use; assuming it is already running."
  fi
fi

log "Starting training"
python -m contra_rl.cli train \
  --config "$CONFIG" \
  --rom "$ROM" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --n-envs "$N_ENVS" \
  --run-name "$RUN_NAME" \
  --device "$DEVICE" \
  --metrics-log-freq "$METRICS_LOG_FREQ"

log "Training finished"
log "Run directory: runs/${RUN_NAME}"
log "TensorBoard: http://localhost:${TB_PORT}"
