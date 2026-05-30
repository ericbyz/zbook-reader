#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
port=9056
if lsof -ti:$port >/dev/null 2>&1; then
  echo "Port $port in use, killing..."
  lsof -ti:$port | xargs kill -9 2>/dev/null || true
fi
echo "Starting ZBook at http://localhost:$port"
python3 -m web.server $port
