#!/usr/bin/env sh

# Preserve the familiar command while delegating security-sensitive download
# and checksum handling to the cross-platform Python implementation.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "${SCRIPT_DIR}/download_models.py" "$@"
