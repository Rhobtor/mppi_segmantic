# model_terrain_segmentation

Terrain-MPPI geométrico + semántico para el ARGO J8 (Unity / ROS 2 Humble) usando
segmentación RGB(-T) con el modelo RGBT-Transformer V2.2 `mit_b0`.

```
segmentation/                  Inferencia (subconjunto usado de RGBT-Transformer-SAR, V2_2)
  V2_2/                        eval_zed.py (PyTorch), eval_zed_tensorrt.py, export_tensorrt.py, modelo
  docker/Dockerfile            Imagen rgbt-transformer:humble-cu124 (ROS 2 + CUDA 12.4 + TensorRT)
  checkpoints/V2_2/mit_b0/     Pesos (NO versionados, ver abajo)
  run_unity_segmentation.sh            Imagen completa -> /segmentation/mask
  run_unity_segmentation_cropped.sh    Sin morro (63 % superior) -> /segmentation_cropped/mask
ros2_ws/src/terrain_mppi_segmentation/ Paquete ROS 2: semantic_cost_node + terrain_mppi_segmentation
unity/                         Scripts Unity: cámara RGB y nube organizada /camera/points
```

## Pesos

Los pesos no se suben a git (`.gitignore`). Colocarlos en `segmentation/checkpoints/V2_2/mit_b0/`:

| Archivo | Uso | Nota |
|---|---|---|
| `epoch-300.pth` (140 MB) | PyTorch (`TRT=0`) y export | > límite de GitHub de 100 MB |
| `epoch-300_mit_b0_fp32_480.onnx` (47 MB) | Fuente para generar el engine | |
| `epoch-300_mit_b0_fp32_480.engine` | TensorRT (por defecto) | Específico de GPU/TensorRT: regenerar |

Comprobar con `cd segmentation/checkpoints/V2_2/mit_b0 && sha256sum -c SHA256SUMS`.

## Segmentación

```bash
cd segmentation
docker build -f docker/Dockerfile -t rgbt-transformer:humble-cu124 .

# Regenerar el engine TensorRT en una GPU nueva (desde el .pth)
docker run --rm --gpus all --ipc=host -v "$PWD":/workspace/RGBT_Transformer_Github \
  -w /workspace/RGBT_Transformer_Github rgbt-transformer:humble-cu124 \
  python V2_2/export_tensorrt.py --precision fp32 --size 480

./run_unity_segmentation_cropped.sh            # TensorRT, topic /camera/rgb/image_raw
TRT=0 ./run_unity_segmentation_cropped.sh      # PyTorch
```

La segunda modalidad (térmica) es sintética: gris de la RGB (`--x-mode rgb-gray`).
Clases (`V2_2/config.py`): 0 First-Responder, 1 Civilian, 2 Vegetation, 3 Road, 4 Dirt-Road,
5 Building, 6 Undergrowth, 7 Civilian-Car, 8 Responder-Vehicle, 9 Debris, 10 Command-Post, 11 Wall.

## Navegador

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
MAKEFLAGS=-j2 colcon build --packages-select terrain_mppi_segmentation \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
./build/terrain_mppi_segmentation/test_terrain_semantic_core
source install/setup.bash
ros2 launch terrain_mppi_segmentation terrain_mppi_segmentation.launch.py semantic_enable:=true
```

Requiere `grid_map` (`ros-humble-grid-map`), el mapa de elevación `/elevation_map`, TF `map→base_link`
y `ros_tcp_endpoint`. Detalles de nodos, costes, seguridad y experimentos en
[ros2_ws/src/terrain_mppi_segmentation/README.md](ros2_ws/src/terrain_mppi_segmentation/README.md).

## Licencia

`segmentation/` deriva de RGBT-Transformer-SAR / CMX (MIT, Copyright (c) 2022 Huayao Liu);
ver `segmentation/LICENSE` y `segmentation/NOTICE.md`.
