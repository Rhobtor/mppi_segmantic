#!/usr/bin/env bash
# Variante sin el morro: conserva el 63 % superior, con topics independientes.
set -euo pipefail

TOPIC="${1:-/camera/rgb/image_raw}"
shift || true
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

exec bash "$SCRIPT_DIR/run_unity_segmentation.sh" "$TOPIC" \
    --crop-keep-ratio 0.63 \
    --mask-topic /segmentation_cropped/mask \
    --overlay-topic /segmentation_cropped/overlay \
    "$@"
