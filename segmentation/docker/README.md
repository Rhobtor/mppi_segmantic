# Docker environment

The Docker image provides ROS 2 Humble and Python 3.10 for ROS message
compatibility, with PyTorch 2.5.1 CUDA 12.4 for inference and training.
The ZED ROS 2 wrapper can keep running on the host; the segmentation
container subscribes to its image topic over the host network.

## Original experimental environment

The locally archived experimental Docker image was:

`amsaladopytorch:rgbx-cu124`

## Recorded image identifier:
`sha256:f51058803f388128006acffe5da7b70160f764ea58d858a608b84d1e2fadf5ad`

The important runtime versions were:
```
Python                    3.11.10
PyTorch                   2.5.1+cu124
torchvision               0.20.1+cu124
CUDA                      12.4.1
numpy                     1.26.4
scipy                     1.13.1
easydict                  1.13
opencv-python-headless    4.10.0.84
timm                      0.9.16
tqdm                      4.66.5
tensorboardX              2.6.2.2
tensorboard               2.17.1
matplotlib                3.9.2
pandas                    2.2.2
scikit-learn              1.5.2
Pillow                    10.4.0
PyYAML                    6.0.2
fvcore                    0.1.5.post20221221
```

## Build
Run from the repository root:
```
docker build \
    -f docker/Dockerfile \
    -t rgbt-transformer:humble-cu124 \
    .
```

## Run the ROS 2 ZED subscriber

Use host networking so ROS 2 DDS discovery can see the ZED wrapper running
on the host. Replace the topic if your wrapper publishes a different name:
```
docker run --rm -it \
    --gpus all \
    --ipc=host \
    --network=host \
    -v "$PWD":/workspace/RGBT_Transformer_Github \
    -w /workspace/RGBT_Transformer_Github \
    rgbt-transformer:humble-cu124 \
    python V2_2/eval_zed.py \
    --topic /zed/zed_node/rgb/image_rect_color \
    --device cuda \
    --overlay-every 5
```

The node publishes class-ID masks (`mono8`) on `/segmentation/mask` and
RGB overlays (`rgb8`) on `/segmentation/overlay`. Topic names can be changed
with `--mask-topic` and `--overlay-topic`. The incoming ROS image encoding
is read from the message, and its header is preserved on both outputs.
`--overlay-every 5` keeps masks at every inference while publishing an
overlay once per five masks. The default is `1`.

Startup and performance logs are written to the container console; use
`docker logs -f <container>` when running detached. The node also reports
while waiting for its first image, so a zero input rate is distinguishable
from a frozen process.

`METRICS` lines are emitted every 5 seconds by default (`--metrics-interval`
changes the interval) and include:

- Input, mask, and overlay FPS; cumulative output counts; matched input publishers,
  last image resolution/encoding, and seconds since the last image.
- Rolling callback, preprocessing, host-to-device transfer, inference,
  resize, mask construction/publication, and overlay construction/publication
  latency (average, p50, p95, and maximum).
- Decode/inference error counts, plus the most frequent predicted classes
  during the reporting interval.
- Process CPU use, cgroup CPU time/throttling, process RSS/high-water/virtual
  memory, cgroup memory use/limit, and CUDA allocated/reserved/peak memory
  when running on a GPU. If `nvidia-smi` is available in the container, the
  line also includes GPU utilization, memory utilization, temperature, and
  power draw.

## Build and run with TensorRT

The Docker image also installs TensorRT 10.13 and ONNX. Rebuild after adding
these dependencies:
```
docker build \
    -f docker/Dockerfile \
    -t rgbt-transformer:humble-cu124 .
```

Create a separate TensorRT engine from the existing mit_b0 checkpoint. The
source `.pth` is not modified:
```
docker run --rm --gpus all --ipc=host \
    -v "$PWD":/workspace/RGBT_Transformer_Github \
    -w /workspace/RGBT_Transformer_Github \
    rgbt-transformer:humble-cu124 \
    python V2_2/export_tensorrt.py \
    --checkpoint checkpoints/V2_2/mit_b0/epoch-300.pth \
    --precision fp32
```

This writes a GPU-specific `.engine`, an intermediate `.onnx`, and a `.json`
metadata report beside the checkpoint. Start the TensorRT ROS subscriber with
the generated engine path:
```
docker run --rm -it --gpus all --ipc=host --network=host \
    -v "$PWD":/workspace/RGBT_Transformer_Github \
    -w /workspace/RGBT_Transformer_Github \
    rgbt-transformer:humble-cu124 \
    python V2_2/eval_zed_tensorrt.py \
    --engine checkpoints/V2_2/mit_b0/epoch-300_mit_b0_fp32_480.engine \
    --topic /zed/zed_node/rgb/image_rect_color \
    --device cuda \
    --overlay-every 5
```

FP32 is the reference setting. `--precision fp16` can be tried for more
throughput, but may change predictions; compare masks and evaluate mIoU on
labeled RGB-T data before using it where accuracy matters. TensorRT engines
are tied to the GPU, TensorRT version, and fixed input size used to build
them; rebuild on the target GPU if any of these differ. TensorRT does not
remove the current CPU cost of constructing and serializing ROS output
images.

To inspect the topics from the host, use `ros2 topic list` and
`ros2 topic info /zed/zed_node/rgb/image_rect_color` in the environment
where the ROS 2 Humble wrapper is running.

The model requires RGB and thermal inputs. With an RGB-only topic, the
subscriber defaults to synthesizing its second input from grayscale RGB;
this is a connectivity/inference test, not an equivalent thermal evaluation.
`--x-mode zeros` is available as an alternate diagnostic input.

## Verify the environment

Inside the container:
```
python -c "import rclpy, torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

For GPU inference, Docker must have NVIDIA Container Toolkit enabled.
`--network=host` is used for ROS 2 DDS discovery; if ROS_DOMAIN_ID is set
for the host wrapper, pass the same value with `-e ROS_DOMAIN_ID=<id>`.

## Multi-GPU training
The model implementations use PyTorch Distributed Data Parallel (DDP).
For example, from the repository root inside the container:
```
torchrun \
    --standalone \
    --nproc_per_node=2 \
    CMX/train.py
```

Equivalent commands can be used for:
`V2_0/train.py`
`V2_2/train.py`

The released original checkpoints were generated from the corresponding
model implementations and the documented experimental dataset split.
