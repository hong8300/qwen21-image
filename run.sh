#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv が見つかりません。uv をインストールしてから再実行してください。" >&2
    exit 1
fi

# Listen on all interfaces so other machines on the LAN can connect.
exec uv run python app.py --host 0.0.0.0 "$@"
