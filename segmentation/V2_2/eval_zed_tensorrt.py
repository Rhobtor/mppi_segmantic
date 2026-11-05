#!/usr/bin/env python3
"""Run the ROS 2 ZED subscriber using a TensorRT engine."""

import logging
from pathlib import Path
import time

import numpy as np
import torch

from config import config
from dataloader.RGBXDataset import RGBXDataset
from eval_zed import build_parser, resolve_device, run_ros_node
from tensorrt_backend import TensorRTEngine


def main():
    parser = build_parser(include_checkpoint=False)
    parser.description = "Subscribe to a ROS 2 image topic and segment with TensorRT."
    parser.add_argument(
        "--engine",
        type=Path,
        required=True,
        help="GPU-specific .engine file created by export_tensorrt.py.",
    )
    args = parser.parse_args()
    if args.size <= 0:
        raise ValueError("--size must be greater than zero.")
    if args.metrics_interval <= 0:
        raise ValueError("--metrics-interval must be greater than zero.")
    if args.overlay_every <= 0:
        raise ValueError("--overlay-every must be greater than zero.")
    if args.overlay_max_width < 0:
        raise ValueError("--overlay-max-width must not be negative.")
    if not args.topic or not args.mask_topic or not args.overlay_topic:
        raise ValueError("Input and output topics must not be empty.")
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    engine_path = args.engine.expanduser().resolve()
    if not engine_path.is_file():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise ValueError("TensorRT requires --device cuda (or --device auto with CUDA).")

    LOGGER = logging.getLogger("rgbt_tensorrt")
    LOGGER.info("Loading TensorRT engine %s on %s", engine_path, device)
    started = time.perf_counter()
    engine = TensorRTEngine(engine_path, device)
    LOGGER.info("TensorRT engine loaded in %.2fs", time.perf_counter() - started)
    expected_shape = (1, 3, args.size, args.size)
    if engine.input_shapes["rgb"] != expected_shape:
        raise ValueError(
            f"--size {args.size} does not match engine input shape "
            f"{engine.input_shapes['rgb']}."
        )
    if engine.input_shapes["modal_x"] != expected_shape:
        raise ValueError(
            f"Engine second input shape {engine.input_shapes['modal_x']} "
            f"does not match {expected_shape}."
        )
    LOGGER.info("TensorRT input shape: %s; device: %s", expected_shape, device)
    colors = np.asarray(
        RGBXDataset.get_class_colors()[: config.num_classes], dtype=np.uint8
    )
    run_ros_node(args, engine, device, colors)


if __name__ == "__main__":
    main()
