#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_PY="$ROOT_DIR/ppg_web/app.py"
PY="$ROOT_DIR/.venv/bin/python"
PORT="${PORT:-5001}"

# Stop any existing server for this app
pkill -f "$APP_PY" >/dev/null 2>&1 || true

# Best-effort: free the port if something else is bound to it
if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
fi

if [ ! -x "$PY" ]; then
  echo "ERROR: venv python not found/executable: $PY" >&2
  echo "Create venv first: python -m venv .venv && .venv/bin/pip install -r ppg_web/requirements.txt" >&2
  exit 1
fi

# Start server in background
nohup env PORT="$PORT" "$PY" "$APP_PY" > "$ROOT_DIR/web.log" 2>&1 &

# Wait until the server responds
for i in {1..40}; do
  if curl -s -o /dev/null "http://127.0.0.1:${PORT}/"; then
    break
  fi
  sleep 0.2
done

URL="http://127.0.0.1:${PORT}/"
echo "Web running: $URL"

# Open browser (best-effort)
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL" >/dev/null 2>&1 || true
fi
