#!/usr/bin/env bash
# Publish the enhancer without an exposed RunPod port: starts a free Cloudflare quick tunnel
# to the local server (default 7860) and prints the public https URL. Jupyter keeps 8888.
# The URL is public, so set ENHANCE_AUTH="user:pass" before starting the server.
set -euo pipefail
WORKSPACE="${ENHANCE_WORKSPACE:-/workspace}"
PORT="${ENHANCE_PORT:-7860}"
BIN="$WORKSPACE/bin/cloudflared"
if [[ ! -x "$BIN" ]]; then
  mkdir -p "$WORKSPACE/bin"
  curl -L --retry 3 -o "$BIN" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x "$BIN"
fi
echo "tunnelling http://localhost:$PORT ... the public URL appears below (ends in trycloudflare.com)"
exec "$BIN" tunnel --no-autoupdate --url "http://localhost:$PORT" 2>&1 | grep --line-buffered -E 'trycloudflare.com|error|ERR'
