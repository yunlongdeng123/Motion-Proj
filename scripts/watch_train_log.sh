#!/usr/bin/env bash
# Convenience wrapper for following the latest formal training log.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/train_motionproj_oneclick.sh" tail "$@"
