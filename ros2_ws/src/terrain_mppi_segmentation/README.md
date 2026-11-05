# Terrain-MPPI geométrico + semántico (`terrain_mppi_segmentation`)

Variante independiente derivada de `MPPITeacher3RingsSubpointBOMNode` (paquete `model_terrain_cpp`,
no incluido aquí). Se eliminaron SAC, subobjetivos, anillos near/mid/far y familias persistentes.
Se conservan rollouts, modelo cinemático tracked/skid-steer, objetivo, evaluación geométrica
(pendiente, escalón, rugosidad, desconocido, obstáculos), veto por huella y publicación de `Twist`.

**Optimizador:** igual que el original, *selección del rollout de coste mínimo* con histéresis
(`action_hysteresis_abs`), no el MPPI canónico con pesos exponenciales.

## Nodos

| Nodo | Entrada | Salida |
|---|---|---|
| `semantic_cost_node` | `/segmentation_cropped/mask` (mono8, IDs de clase), `/camera/points` (nube organizada xyz, frame `camera_link`), `/elevation_map`, TF `map←camera_link` | `/semantic/costmap`, `/semantic/obstacles` (OccupancyGrid, mismo frame/origen/resolución que `/elevation_map`), `/semantic/stop_required`, `/semantic/speed_scale`, `/semantic/status`, `/semantic/class_markers` |
| `terrain_mppi_segmentation` | `/elevation_map`, `/goal_local` (PoseArray), `/goal_ps` (PoseStamped), TF `map←base_link`, capas `/semantic/*` | `/cmd_vel`, `/terrain_mppi_segmentation/rollouts`, `/terrain_mppi_segmentation/decision`, `/terrain_mppi_segmentation/goal_reached` |

La proyección se hace fuera del MPPI: cada píxel de la máscara se asocia al punto 3D de la nube
organizada de Unity (rejilla `cloud_image_width/height`, `cloud_pixel_step`, `cloud_roi_v_min`),
se transforma a `map` y se compara con la elevación para separar suelo/obstáculo.
El controlador sólo consulta celdas ya calculadas.

## Coste

`J = J_terrain + J_semantic + J_goal + J_control + J_obstacles`

* `J_semantic = Σ_t w_semantic · cost_clase(x_t, y_t) · dt` (en `rolloutEval()`); celdas sin clase → 0.
* Clases reales (`V2_2/config.py`): Road/Dirt-Road 0.0, Undergrowth 0.25, Vegetation 0.60.
  El modelo **no** tiene clases hierba/tierra natural. Configurable en `class_costs`.
* Obstáculos semánticos `{0,1,5,7,8,9,10,11}` (personas `{0,1}`) se inflan con la huella y
  **vetan** el rollout como un obstáculo geométrico. La semántica nunca relaja la geometría.

## Seguridad

* Segmentación caída → coste de preferencia desactivado, navegación geométrica.
* Obstáculos retenidos y entradas obsoletas → `stop_required` (dentro de `stale_stop_radius_m`).
* Persona vista sin profundidad válida → `speed_scale = caution_speed_scale` durante `caution_hold_sec`.
* El veto se aplica a todos los rollouts, al de histéresis (keep-last) y al comando final.

## Uso

```bash
# 1) segmentación recortada
./segmentation/run_unity_segmentation_cropped.sh
# 2) navegador (A/B con los mismos parámetros)
source ros2_ws/install/setup.bash
ros2 launch terrain_mppi_segmentation terrain_mppi_segmentation.launch.py semantic_enable:=false   # geométrico
ros2 launch terrain_mppi_segmentation terrain_mppi_segmentation.launch.py semantic_enable:=true    # semántico
```

La salida por defecto es `/cmd_vel`, como en el navegador existente. Apagar el otro
controlador antes de arrancar esta variante; ejecutar solo uno de los comandos A/B.
Para pruebas sin mover el robot: `cmd_topic:=/terrain_mppi_segmentation/cmd_vel`.
Los argumentos `goal_topic:=/goal_ps` y `goal_topic_pa:=/goal_local` permiten cambiar
respectivamente las entradas PoseStamped y PoseArray. Enviar el objetivo después de arrancar.

RViz (Fixed Frame `map`): Map `/semantic/costmap`, Map `/semantic/obstacles`,
MarkerArray `/semantic/class_markers`, MarkerArray `/terrain_mppi_segmentation/rollouts`;
texto: `ros2 topic echo /terrain_mppi_segmentation/decision`, `/semantic/status`.

## Experimentos

1. Camino de tierra rodeado de vegetación: con semántica debe preferir Dirt-Road si el desvío es pequeño.
2. Terreno natural sin camino: ambos modos deben llegar; la semántica no debe bloquear.
3. Persona/vehículo bloqueando parte del recorrido: debe aparecer en `/semantic/obstacles` y rodearse.

## Tests

`ros2_ws/build/terrain_mppi_segmentation/test_terrain_semantic_core` (11 tests gtest de proyección, retención de
obstáculos, histéresis/seguridad, lattice Unity→máscara, paleta).
