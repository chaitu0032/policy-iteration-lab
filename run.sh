#!/usr/bin/env bash
# Start (or restart) Policy Iteration Lab on http://localhost:${PORT:-8765}
set -euo pipefail
PORT="${PORT:-8765}"
cd "$(dirname "$0")"
# Free the port if an old / stuck server still holds it.
if lsof -ti tcp:"$PORT" >/dev/null 2>&1; then
  echo "Stopping old server on port $PORT"
  lsof -ti tcp:"$PORT" | xargs kill -9 || true
  sleep 1
fi
echo "Policy Iteration Lab -> http://localhost:$PORT   (Ctrl+C to stop)"
exec python3 -m uvicorn backend.app:app --host 127.0.0.1 --port "$PORT"
