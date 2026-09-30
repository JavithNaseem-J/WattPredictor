#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-10000}"
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    echo "PORT must be an integer from 1 to 65535" >&2
    exit 1
fi

envsubst '${PORT}' < /app/deploy/nginx.conf.template > /tmp/wattpredictor-nginx.conf
nginx -t -c /tmp/wattpredictor-nginx.conf

streamlit run /app/app.py --server.port=8501 --server.address=127.0.0.1 --server.headless=true &
streamlit_pid=$!
uvicorn WattPredictor.api.main:app --host 127.0.0.1 --port 8000 &
api_pid=$!
nginx -c /tmp/wattpredictor-nginx.conf -g 'daemon off;' &
nginx_pid=$!

cleanup() {
    trap - EXIT TERM INT
    kill "$streamlit_pid" "$api_pid" "$nginx_pid" 2>/dev/null || true
    wait "$streamlit_pid" "$api_pid" "$nginx_pid" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

# Any stopped server ends the container so Render can replace it.
set +e
wait -n "$streamlit_pid" "$api_pid" "$nginx_pid"
exit 1
