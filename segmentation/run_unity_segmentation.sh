#!/usr/bin/env bash
# Segmentación V2.2 mit_b0 sobre la cámara RGB del robot en Unity (vía ros_tcp_endpoint).
# Uso: ./run_unity_segmentation.sh [topic] [args extra de eval_zed.py]
#   TRT=0 ./run_unity_segmentation.sh   -> usa PyTorch en lugar de TensorRT
set -euo pipefail

TOPIC="${1:-/camera/rgb/image_raw}"
shift || true
REPO="$(cd "$(dirname "$0")" && pwd)"

if [[ "${TRT:-1}" == "1" ]]; then
    SCRIPT=(python V2_2/eval_zed_tensorrt.py
            --engine checkpoints/V2_2/mit_b0/epoch-300_mit_b0_fp32_480.engine)
else
    SCRIPT=(python V2_2/eval_zed.py)
fi

exec docker run --rm -it \
    --gpus all --ipc=host --network=host \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
    -v "$REPO":/workspace/RGBT_Transformer_Github \
    -w /workspace/RGBT_Transformer_Github \
    rgbt-transformer:humble-cu124 \
    "${SCRIPT[@]}" \
    --topic "$TOPIC" \
    --device cuda \
    --overlay-every 1 \
    --overlay-max-width 640 \
    "$@"
