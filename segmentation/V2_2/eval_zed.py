#!/usr/bin/env python3
"""ROS 2 RGB image subscriber for live V2.2 mit_b0 segmentation."""

import argparse
import array
from collections import deque
import logging
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import cv2
import numpy as np
import torch
import torch.nn as nn

from config import config
from dataloader.RGBXDataset import RGBXDataset
from models.builder import EncoderDecoder
from utils.pyt_utils import load_model


DEFAULT_CHECKPOINT = (
    Path(config.repo_dir) / "checkpoints" / "V2_2" / "mit_b0" / "epoch-300.pth"
)
DEFAULT_IMAGE_TOPIC = "/zed/zed_node/rgb/image_rect_color"
LOGGER = logging.getLogger("rgbt_segmentation")


def crop_keep_ratio(value):
    ratio = float(value)
    if not 0 < ratio <= 1:
        raise argparse.ArgumentTypeError("--crop-keep-ratio must be in (0, 1].")
    return ratio


def crop_rgb_image(rgb, keep_ratio):
    if not 0 < keep_ratio <= 1:
        raise ValueError("--crop-keep-ratio must be in (0, 1].")
    height = max(1, int(rgb.shape[0] * keep_ratio))
    return rgb[:height]


def build_parser(include_checkpoint=True):
    parser = argparse.ArgumentParser(
        description="Subscribe to a ROS 2 RGB image topic and segment each frame."
    )
    if include_checkpoint:
        parser.add_argument(
            "--checkpoint",
            type=Path,
            default=DEFAULT_CHECKPOINT,
            help="V2.2 mit_b0 checkpoint (default: %(default)s)",
        )
    parser.add_argument(
        "--topic",
        default=DEFAULT_IMAGE_TOPIC,
        help=f"Input sensor_msgs/Image topic (default: {DEFAULT_IMAGE_TOPIC})",
    )
    parser.add_argument(
        "--mask-topic",
        default="/segmentation/mask",
        help="Output mono8 topic containing class IDs.",
    )
    parser.add_argument(
        "--overlay-topic",
        default="/segmentation/overlay",
        help="Output rgb8 topic containing the segmentation overlay.",
    )
    parser.add_argument(
        "--crop-keep-ratio",
        type=crop_keep_ratio,
        default=1.0,
        help=(
            "Keep this fraction of the image height from the top before inference "
            "(0 < ratio <= 1; default: 1, no crop). Outputs use cropped dimensions."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, or cuda:N (default: auto).",
    )
    parser.add_argument(
        "--x-mode",
        choices=("rgb-gray", "zeros"),
        default="rgb-gray",
        help=(
            "Synthetic second modality required by the RGB-T network: "
            "grayscale converted from RGB or zeros."
        ),
    )
    parser.add_argument(
        "--size",
        type=int,
        default=480,
        help="Square model input size (default: 480).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="Optional directory for per-frame raw masks and overlays.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show a local OpenCV preview (requires a display server).",
    )
    parser.add_argument(
        "--metrics-interval",
        type=float,
        default=5.0,
        help="Seconds between health and performance logs (default: 5).",
    )
    parser.add_argument(
        "--overlay-max-width",
        type=int,
        default=0,
        help=(
            "Limit the published overlay width, preserving aspect ratio. "
            "0 keeps the original resolution; masks and saved images are unchanged."
        ),
    )
    parser.add_argument(
        "--overlay-every",
        type=int,
        default=1,
        help=(
            "Publish the visualization overlay once every N processed frames. "
            "The class mask is still published for every frame (default: 1)."
        ),
    )
    return parser


def parse_args():
    return build_parser().parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch reports CUDA unavailable.")
    return device


def make_model(checkpoint, device):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    config.backbone = "mit_b0"
    model = EncoderDecoder(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    checkpoint_state = torch.load(checkpoint, map_location="cpu")
    if "model" in checkpoint_state:
        checkpoint_state = checkpoint_state["model"]
    elif "state_dict" in checkpoint_state:
        checkpoint_state = checkpoint_state["state_dict"]
    elif "module" in checkpoint_state:
        checkpoint_state = checkpoint_state["module"]
    load_model(model, checkpoint_state)
    model.to(device)
    model.eval()
    return model


def decode_rgb_image(message):
    encodings = {
        "rgb8": (3, "rgb"),
        "bgr8": (3, "bgr"),
        "rgba8": (4, "rgba"),
        "bgra8": (4, "bgra"),
        "mono8": (1, "mono"),
    }
    encoding = message.encoding.lower()
    if encoding not in encodings:
        raise ValueError(f"Unsupported image encoding {message.encoding!r}.")

    channels, color_order = encodings[encoding]
    row_bytes = message.width * channels
    if message.step < row_bytes:
        raise ValueError(
            f"Image step {message.step} is smaller than row size {row_bytes}."
        )
    data = np.frombuffer(message.data, dtype=np.uint8)
    required_bytes = message.step * message.height
    if data.size < required_bytes:
        raise ValueError(
            f"Image data has {data.size} bytes; expected at least {required_bytes}."
        )

    rows = data[:required_bytes].reshape(message.height, message.step)
    image = rows[:, :row_bytes].reshape(message.height, message.width, channels)
    if color_order == "rgb":
        return image.copy()
    if color_order == "bgr":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if color_order == "rgba":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if color_order == "bgra":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def normalize_image(image, mean, std):
    image = image.astype(np.float32) / 255.0
    return (image - mean) / std


def make_model_inputs(rgb, size, x_mode):
    height, width = rgb.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized_rgb = cv2.resize(
        rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )

    if x_mode == "rgb-gray":
        gray = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)
        modal_x = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        modal_x = np.zeros_like(resized_rgb)

    mean = config.norm_mean.astype(np.float32)
    std = config.norm_std.astype(np.float32)
    normalized_rgb = normalize_image(resized_rgb, mean, std)
    normalized_x = normalize_image(modal_x, mean, std)

    top = (size - resized_height) // 2
    left = (size - resized_width) // 2
    rgb_canvas = np.zeros((size, size, 3), dtype=np.float32)
    x_canvas = np.zeros_like(rgb_canvas)
    rgb_canvas[top : top + resized_height, left : left + resized_width] = normalized_rgb
    x_canvas[top : top + resized_height, left : left + resized_width] = normalized_x

    rgb_tensor = torch.from_numpy(rgb_canvas.transpose(2, 0, 1)).unsqueeze(0)
    x_tensor = torch.from_numpy(x_canvas.transpose(2, 0, 1)).unsqueeze(0)
    return rgb_tensor, x_tensor, (top, left, resized_height, resized_width)


@torch.inference_mode()
def predict(model, rgb, device, size, x_mode):
    preprocess_started = time.perf_counter()
    rgb_tensor, x_tensor, bounds = make_model_inputs(rgb, size, x_mode)
    preprocess_ms = (time.perf_counter() - preprocess_started) * 1000

    transfer_started = time.perf_counter()
    rgb_tensor = rgb_tensor.to(device)
    x_tensor = x_tensor.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    transfer_ms = (time.perf_counter() - transfer_started) * 1000

    inference_started = time.perf_counter()
    logits = model(rgb_tensor, x_tensor)
    prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_ms = (time.perf_counter() - inference_started) * 1000

    top, left, resized_height, resized_width = bounds
    prediction = prediction[top : top + resized_height, left : left + resized_width]
    resize_started = time.perf_counter()
    prediction = cv2.resize(
        prediction, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    resize_ms = (time.perf_counter() - resize_started) * 1000
    timings = {
        "preprocess_ms": preprocess_ms,
        "transfer_ms": transfer_ms,
        "inference_ms": inference_ms,
        "resize_ms": resize_ms,
    }
    return prediction, timings


def make_ros_image(message_type, header, image, encoding):
    message = message_type()
    message.header = header
    message.height, message.width = image.shape[:2]
    message.encoding = encoding
    message.is_bigendian = 0
    message.step = image.shape[1] * (1 if image.ndim == 2 else image.shape[2])
    # rclpy validates bytes element by element; a 'B' array.array takes the fast path.
    message.data = array.array("B", np.ascontiguousarray(image).tobytes())
    return message


def make_overlay(rgb, mask, colors, max_width):
    if 0 < max_width < rgb.shape[1]:
        width = max_width
        height = max(1, round(rgb.shape[0] * width / rgb.shape[1]))
        size = (width, height)
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(rgb, 0.6, colors[mask], 0.4, 0)


def read_process_memory():
    values = {}
    try:
        with open("/proc/self/status", encoding="ascii") as status_file:
            for line in status_file:
                if line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
                    key, value = line.split(":", 1)
                    values[key] = value.strip()
    except OSError:
        LOGGER.debug("Could not read process memory from /proc/self/status", exc_info=True)
    return values


def read_cgroup_value(path):
    try:
        with open(path, encoding="ascii") as value_file:
            return value_file.read().strip()
    except OSError:
        return None


def read_gpu_telemetry(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return "n/a"
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return "unavailable"
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                f"--id={device_index}",
                "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        LOGGER.debug("Could not read GPU telemetry from nvidia-smi", exc_info=True)
        return "unavailable"
    values = [value.strip() for value in result.stdout.strip().split(",")]
    if len(values) != 4:
        return "unavailable"
    return (
        f"util={values[0]}% mem_util={values[1]}% "
        f"temp={values[2]}C power={values[3]}W"
    )


def format_latency(values):
    if not values:
        return "n/a"
    samples = np.asarray(values, dtype=np.float64)
    return (
        f"avg={samples.mean():.1f} p50={np.percentile(samples, 50):.1f} "
        f"p95={np.percentile(samples, 95):.1f} max={samples.max():.1f} ms"
    )


def run_ros_node(args, model, device, colors):
    if args.overlay_max_width < 0:
        raise ValueError("--overlay-max-width must be zero or greater.")
    if not 0 < args.crop_keep_ratio <= 1:
        raise ValueError("--crop-keep-ratio must be in (0, 1].")
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
            qos_profile_sensor_data,
        )
        from sensor_msgs.msg import Image
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 Python packages are unavailable. Run this script inside the "
            "project's ROS 2 Humble Docker image."
        ) from exc

    class SegmentationNode(Node):
        def __init__(self):
            super().__init__("rgbt_segmentation")
            self.get_logger().info(
                f"Keeping top {args.crop_keep_ratio:.1%} of each image before "
                "inference; output dimensions follow the crop."
            )
            self.publisher_mask = self.create_publisher(
                Image, args.mask_topic, qos_profile_sensor_data
            )
            self.publisher_overlay = self.create_publisher(
                Image, args.overlay_topic, qos_profile_sensor_data
            )
            # Keep only the newest frame so slow callbacks never accumulate latency.
            input_qos = QoSProfile(
                depth=1,
                history=HistoryPolicy.KEEP_LAST,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
            self.subscription = self.create_subscription(
                Image, args.topic, self.on_image, input_qos
            )
            self.frame_index = 0
            self.received_frames = 0
            self.mask_frames = 0
            self.overlay_frames = 0
            self.received_total = 0
            self.mask_total = 0
            self.overlay_total = 0
            self.decode_errors = 0
            self.inference_errors = 0
            self.last_frame_monotonic = None
            self.last_frame_details = "no image received yet"
            self.interval_started = time.perf_counter()
            self.last_cpu_time = time.process_time()
            self.latencies = {
                name: deque(maxlen=1000)
                for name in (
                    "callback",
                    "preprocess",
                    "transfer",
                    "inference",
                    "resize",
                    "mask_build",
                    "mask_publish",
                    "overlay_build",
                    "overlay_publish",
                )
            }
            self.class_pixel_counts = np.zeros(config.num_classes, dtype=np.int64)
            self.timer = self.create_timer(
                args.metrics_interval, self.log_metrics
            )
            self.get_logger().info(f"Node started; waiting for images on {args.topic}")
            self.get_logger().info(f"Publishing class IDs on {args.mask_topic}")
            self.get_logger().info(f"Publishing overlay on {args.overlay_topic}")
            self.get_logger().info(
                f"Metrics interval: {args.metrics_interval:.1f}s; device: {device}; "
                f"input size: {args.size}x{args.size}"
            )
            self.get_logger().warning(
                f"Using synthetic second modality {args.x_mode!r}; "
                "this model was trained for RGB plus thermal input."
            )

        def log_metrics(self):
            now = time.perf_counter()
            interval = now - self.interval_started
            if interval <= 0:
                return
            input_fps = self.received_frames / interval
            mask_fps = self.mask_frames / interval
            overlay_fps = self.overlay_frames / interval
            self.interval_started = now
            self.received_frames = 0
            self.mask_frames = 0
            self.overlay_frames = 0

            cpu_time = time.process_time()
            cpu_cores_pct = 100.0 * (cpu_time - self.last_cpu_time) / interval
            self.last_cpu_time = cpu_time
            cpu_logical_pct = cpu_cores_pct / max(os.cpu_count() or 1, 1)
            memory = read_process_memory()
            rss = memory.get("VmRSS", "n/a")
            high_water = memory.get("VmHWM", "n/a")
            virtual = memory.get("VmSize", "n/a")
            cgroup_memory = read_cgroup_value("/sys/fs/cgroup/memory.current")
            cgroup_memory_max = read_cgroup_value("/sys/fs/cgroup/memory.max")
            if cgroup_memory and cgroup_memory.isdigit():
                cgroup_memory = f"{int(cgroup_memory) / (1024 ** 2):.1f} MiB"
            if cgroup_memory_max and cgroup_memory_max.isdigit():
                cgroup_memory_max = f"{int(cgroup_memory_max) / (1024 ** 2):.1f} MiB"
            cgroup_cpu = read_cgroup_value("/sys/fs/cgroup/cpu.stat")
            cpu_usage = "n/a"
            cpu_throttling = "n/a"
            if cgroup_cpu:
                cpu_values = dict(
                    line.split(maxsplit=1)
                    for line in cgroup_cpu.splitlines()
                    if len(line.split(maxsplit=1)) == 2
                )
                usage_line = cpu_values.get("usage_usec")
                if usage_line:
                    cpu_usage = f"{int(usage_line) / 1_000_000:.2f}s total"
                throttled_usec = cpu_values.get("throttled_usec")
                if throttled_usec:
                    cpu_throttling = f"{int(throttled_usec) / 1_000_000:.2f}s total"

            if self.last_frame_monotonic is None:
                frame_age = "never"
            else:
                frame_age = f"{now - self.last_frame_monotonic:.1f}s"

            gpu_memory = "n/a"
            gpu_peak = "n/a"
            gpu_telemetry = read_gpu_telemetry(device)
            if device.type == "cuda" and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
                reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
                peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                gpu_memory = f"allocated={allocated:.0f} MiB reserved={reserved:.0f} MiB"
                gpu_peak = f"{peak:.0f} MiB"
                torch.cuda.reset_peak_memory_stats(device)

            top_classes = "n/a"
            total_pixels = int(self.class_pixel_counts.sum())
            if total_pixels:
                top_indices = np.argsort(self.class_pixel_counts)[-3:][::-1]
                top_classes = ", ".join(
                    f"{config.class_names[index]}:"
                    f"{100 * self.class_pixel_counts[index] / total_pixels:.1f}%"
                    for index in top_indices
                    if self.class_pixel_counts[index] > 0
                )
                self.class_pixel_counts.fill(0)

            self.get_logger().info(
                "METRICS "
                f"input_fps={input_fps:.2f} mask_fps={mask_fps:.2f} "
                f"overlay_fps={overlay_fps:.2f} "
                f"mask_total={self.mask_total} overlay_total={self.overlay_total} "
                f"received_total={self.received_total} "
                f"callback_ms[{format_latency(self.latencies['callback'])}] "
                f"preprocess_ms[{format_latency(self.latencies['preprocess'])}] "
                f"h2d_ms[{format_latency(self.latencies['transfer'])}] "
                f"inference_ms[{format_latency(self.latencies['inference'])}] "
                f"resize_ms[{format_latency(self.latencies['resize'])}] "
                f"mask_build_ms[{format_latency(self.latencies['mask_build'])}] "
                f"mask_publish_ms[{format_latency(self.latencies['mask_publish'])}] "
                f"overlay_build_ms[{format_latency(self.latencies['overlay_build'])}] "
                f"overlay_publish_ms[{format_latency(self.latencies['overlay_publish'])}] "
                f"overlay_every={args.overlay_every} "
                f"frame_age={frame_age} matched_publishers={self.count_publishers(args.topic)} "
                f"errors_decode={self.decode_errors} errors_inference={self.inference_errors} "
                f"last_image=({self.last_frame_details}) "
                f"cpu_process={cpu_cores_pct:.1f}% cores/{cpu_logical_pct:.1f}% logical "
                f"cgroup_cpu={cpu_usage} cpu_throttled={cpu_throttling} "
                f"ram_rss={rss} ram_peak={high_water} ram_virtual={virtual} "
                f"cgroup_mem={cgroup_memory}/{cgroup_memory_max} "
                f"gpu_mem[{gpu_memory}] gpu_peak={gpu_peak} gpu_stats[{gpu_telemetry}] "
                f"top_classes=[{top_classes}]"
            )

        def on_image(self, message):
            callback_started = time.perf_counter()
            self.last_frame_monotonic = callback_started
            self.last_frame_details = (
                f"{message.width}x{message.height} {message.encoding}"
            )
            self.received_frames += 1
            self.received_total += 1
            try:
                rgb = decode_rgb_image(message)
                rgb = crop_rgb_image(rgb, args.crop_keep_ratio)
                self.last_frame_details += f" -> {rgb.shape[1]}x{rgb.shape[0]} crop"
            except ValueError as exc:
                self.decode_errors += 1
                self.get_logger().error(f"Cannot decode input image: {exc}")
                return

            try:
                mask, timings = predict(model, rgb, device, args.size, args.x_mode)
            except Exception as exc:
                self.inference_errors += 1
                self.get_logger().error(
                    f"Inference failed for frame {self.frame_index}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                return
            mask_build_started = time.perf_counter()
            mask_message = make_ros_image(Image, message.header, mask, "mono8")
            mask_build_ms = (time.perf_counter() - mask_build_started) * 1000
            mask_publish_started = time.perf_counter()
            self.publisher_mask.publish(mask_message)
            mask_publish_ms = (time.perf_counter() - mask_publish_started) * 1000
            self.mask_frames += 1
            self.mask_total += 1

            overlay = None
            overlay_build_ms = 0.0
            overlay_publish_ms = 0.0
            publish_overlay = self.frame_index % args.overlay_every == 0
            if publish_overlay:
                overlay_build_started = time.perf_counter()
                if args.save_dir is not None or args.display:
                    overlay = make_overlay(rgb, mask, colors, 0)
                published_overlay = make_overlay(
                    rgb, mask, colors, args.overlay_max_width
                )
                overlay_message = make_ros_image(
                    Image, message.header, published_overlay, "rgb8"
                )
                overlay_build_ms = (
                    time.perf_counter() - overlay_build_started
                ) * 1000
                overlay_publish_started = time.perf_counter()
                self.publisher_overlay.publish(overlay_message)
                overlay_publish_ms = (
                    time.perf_counter() - overlay_publish_started
                ) * 1000
                self.overlay_frames += 1
                self.overlay_total += 1

            if args.save_dir is not None:
                stem = f"frame_{self.frame_index:06d}"
                cv2.imwrite(str(args.save_dir / f"{stem}_mask.png"), mask)
                if overlay is not None:
                    cv2.imwrite(
                        str(args.save_dir / f"{stem}_overlay.png"),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                    )
            if args.display and overlay is not None:
                preview = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    preview,
                    f"FPS: {1.0 / max(time.perf_counter() - callback_started, 1e-6):.1f}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("RGB-T segmentation", preview)
                cv2.waitKey(1)
            self.latencies["preprocess"].append(timings["preprocess_ms"])
            self.latencies["transfer"].append(timings["transfer_ms"])
            self.latencies["inference"].append(timings["inference_ms"])
            self.latencies["resize"].append(timings["resize_ms"])
            self.latencies["mask_build"].append(mask_build_ms)
            self.latencies["mask_publish"].append(mask_publish_ms)
            if publish_overlay:
                self.latencies["overlay_build"].append(overlay_build_ms)
                self.latencies["overlay_publish"].append(overlay_publish_ms)
            self.latencies["callback"].append(
                (time.perf_counter() - callback_started) * 1000
            )
            self.class_pixel_counts += np.bincount(
                mask.ravel(), minlength=config.num_classes
            )
            self.frame_index += 1

    rclpy.init()
    node = SegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.log_metrics()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if args.display:
            cv2.destroyAllWindows()


def main():
    args = parse_args()
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

    checkpoint = args.checkpoint.expanduser().resolve()
    device = resolve_device(args.device)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    LOGGER.info("Loading checkpoint %s on %s", checkpoint, device)
    load_started = time.perf_counter()
    model = make_model(checkpoint, device)
    LOGGER.info("Checkpoint loaded in %.2fs", time.perf_counter() - load_started)
    colors = np.asarray(
        RGBXDataset.get_class_colors()[: config.num_classes], dtype=np.uint8
    )

    LOGGER.info("Checkpoint: %s", checkpoint)
    LOGGER.info("Device: %s", device)
    run_ros_node(args, model, device, colors)


if __name__ == "__main__":
    main()
