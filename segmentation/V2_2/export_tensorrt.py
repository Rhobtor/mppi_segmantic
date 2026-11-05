#!/usr/bin/env python3
"""Export the V2.2 mit_b0 checkpoint to a GPU-specific TensorRT engine."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import time

import torch
import torch.nn as nn

from config import config
from models.builder import EncoderDecoder
from tensorrt_backend import TensorRTEngine, build_engine_from_onnx
from utils.pyt_utils import load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a V2.2 mit_b0 checkpoint into a TensorRT engine."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(config.repo_dir)
        / "checkpoints"
        / "V2_2"
        / "mit_b0"
        / "epoch-300.pth",
        help="Source PyTorch checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output TensorRT engine path (defaults next to the checkpoint).",
    )
    parser.add_argument(
        "--onnx-output",
        type=Path,
        help="Intermediate ONNX path (defaults next to the engine).",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16"),
        default="fp32",
        help="Engine precision; FP32 is the accuracy-reference mode.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=480,
        help="Fixed square input size; must match the inference command.",
    )
    parser.add_argument(
        "--workspace-gib",
        type=float,
        default=2.0,
        help="TensorRT workspace budget in GiB (default: 2).",
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=50,
        help="Number of synchronized iterations per backend (default: 50).",
    )
    return parser.parse_args()


def load_checkpoint_model(checkpoint_path, device):
    config.backbone = "mit_b0"
    model = EncoderDecoder(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    state = torch.load(checkpoint_path, map_location="cpu")
    if "model" in state:
        state = state["model"]
    elif "state_dict" in state:
        state = state["state_dict"]
    elif "module" in state:
        state = state["module"]
    load_model(model, state)
    model.to(device).eval()
    return model


def benchmark_inference(run_inference, inputs, device, warmup, iterations):
    for _ in range(warmup):
        run_inference(*inputs)
    torch.cuda.synchronize(device)

    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        run_inference(*inputs)
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - started) * 1000)

    samples = torch.tensor(samples, dtype=torch.float64)
    return {
        "mean_ms": samples.mean().item(),
        "p50_ms": torch.quantile(samples, 0.50).item(),
        "p95_ms": torch.quantile(samples, 0.95).item(),
        "fps": 1000.0 / samples.mean().item(),
    }


def main():
    args = parse_args()
    if args.size <= 0 or args.workspace_gib <= 0 or args.benchmark_iterations <= 0:
        raise ValueError(
            "--size, --workspace-gib, and --benchmark-iterations must be greater than zero."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "TensorRT export requires the NVIDIA GPU to be exposed to Docker; "
            "run docker with --gpus all."
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    engine_path = (
        args.output.expanduser().resolve()
        if args.output
        else checkpoint_path.with_name(
            f"{checkpoint_path.stem}_mit_b0_{args.precision}_{args.size}.engine"
        )
    )
    onnx_path = (
        args.onnx_output.expanduser().resolve()
        if args.onnx_output
        else engine_path.with_suffix(".onnx")
    )
    device = torch.device("cuda")
    torch.manual_seed(12345)

    LOGGER = logging.getLogger("tensorrt_export")
    LOGGER.info("Checkpoint: %s", checkpoint_path)
    LOGGER.info("GPU: %s", torch.cuda.get_device_name(device))
    model = load_checkpoint_model(checkpoint_path, device)
    rgb = torch.randn(1, 3, args.size, args.size, device=device)
    modal_x = torch.randn_like(rgb)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (rgb, modal_x),
            str(onnx_path),
            input_names=["rgb", "modal_x"],
            output_names=["logits"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    LOGGER.info("ONNX exported to %s", onnx_path)

    started = time.perf_counter()
    build_engine_from_onnx(
        onnx_path, engine_path, args.precision, args.workspace_gib
    )
    LOGGER.info("TensorRT engine built in %.1fs: %s", time.perf_counter() - started, engine_path)

    engine = TensorRTEngine(engine_path, device)
    with torch.inference_mode():
        reference = model(rgb, modal_x).float()
        actual = engine(rgb, modal_x).float()
        torch.cuda.synchronize(device)
    max_abs_error = (reference - actual).abs().max().item()
    mean_abs_error = (reference - actual).abs().mean().item()
    pixel_agreement = (
        reference.argmax(dim=1) == actual.argmax(dim=1)
    ).float().mean().item()
    LOGGER.info(
        "Validation on synthetic input: pixel_agreement=%.5f%% "
        "mean_abs_logit_error=%.7g max_abs_logit_error=%.7g",
        pixel_agreement * 100,
        mean_abs_error,
        max_abs_error,
    )
    LOGGER.warning(
        "Synthetic-input agreement is a numerical smoke test, not a substitute "
        "for evaluating mIoU on labeled RGB-T data."
    )
    benchmark_iters = args.benchmark_iterations
    pytorch_benchmark = benchmark_inference(
        model, (rgb, modal_x), device, warmup=10, iterations=benchmark_iters
    )
    tensorrt_benchmark = benchmark_inference(
        engine, (rgb, modal_x), device, warmup=10, iterations=benchmark_iters
    )
    LOGGER.info(
        "Synchronized inference benchmark (%d iterations): "
        "PyTorch mean=%.2fms p50=%.2fms p95=%.2fms %.2f FPS; "
        "TensorRT mean=%.2fms p50=%.2fms p95=%.2fms %.2f FPS; "
        "speedup=%.2fx",
        benchmark_iters,
        pytorch_benchmark["mean_ms"],
        pytorch_benchmark["p50_ms"],
        pytorch_benchmark["p95_ms"],
        pytorch_benchmark["fps"],
        tensorrt_benchmark["mean_ms"],
        tensorrt_benchmark["p50_ms"],
        tensorrt_benchmark["p95_ms"],
        tensorrt_benchmark["fps"],
        pytorch_benchmark["mean_ms"] / tensorrt_benchmark["mean_ms"],
    )

    import tensorrt as trt

    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    metadata = {
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "engine": engine_path.name,
        "onnx": onnx_path.name,
        "precision": args.precision,
        "input_shape": [1, 3, args.size, args.size],
        "thermal_input_shape": [1, 3, args.size, args.size],
        "gpu": torch.cuda.get_device_name(device),
        "tensorrt_version": trt.__version__,
        "torch_version": torch.__version__,
        "pixel_agreement_on_random_input": pixel_agreement,
        "mean_abs_logit_error_on_random_input": mean_abs_error,
        "max_abs_logit_error_on_random_input": max_abs_error,
        "pytorch_benchmark": pytorch_benchmark,
        "tensorrt_benchmark": tensorrt_benchmark,
    }
    engine_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    LOGGER.info("Engine metadata written to %s", engine_path.with_suffix(".json"))


if __name__ == "__main__":
    main()
