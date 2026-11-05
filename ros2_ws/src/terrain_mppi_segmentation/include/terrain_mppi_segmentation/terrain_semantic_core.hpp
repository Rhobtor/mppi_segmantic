#pragma once

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <grid_map_core/grid_map_core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace terrain_semantic
{
inline double wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }
inline double stampSeconds(const builtin_interfaces::msg::Time & s)
{
  return s.sec + s.nanosec * 1e-9;
}
inline bool fresh(double now, double stamp, double ttl)
{
  return std::isfinite(stamp) && stamp > 0 && now - stamp >= -0.05 && now - stamp <= ttl;
}

struct Grid
{
  nav_msgs::msg::MapMetaData info;
  std::string frame;
  int index(double x, double y) const
  {
    const double r = info.resolution;
    if (!(r > 0) || !std::isfinite(x) || !std::isfinite(y)) return -1;
    const double fx = (x - info.origin.position.x) / r;
    const double fy = (y - info.origin.position.y) / r;
    if (fx < 0 || fy < 0 || fx >= info.width || fy >= info.height) return -1;
    return static_cast<int>(fy) * static_cast<int>(info.width) + static_cast<int>(fx);
  }
  std::pair<double, double> position(size_t i) const
  {
    return {info.origin.position.x + (i % info.width + 0.5) * info.resolution,
      info.origin.position.y + (i / info.width + 0.5) * info.resolution};
  }
  size_t size() const { return static_cast<size_t>(info.width) * info.height; }
};

inline Grid layout(const grid_map::GridMap & map)
{
  Grid g;
  g.frame = map.getFrameId();
  g.info.resolution = map.getResolution();
  g.info.width = map.getSize()(0);
  g.info.height = map.getSize()(1);
  g.info.origin.position.x = map.getPosition().x() - map.getLength().x() / 2;
  g.info.origin.position.y = map.getPosition().y() - map.getLength().y() / 2;
  g.info.origin.orientation.w = 1;
  if (g.frame.empty() || !g.size() || g.size() > 4000000 || g.info.resolution <= 0) {
    throw std::invalid_argument("Invalid terrain grid geometry");
  }
  return g;
}

inline bool validGrid(const nav_msgs::msg::OccupancyGrid & m)
{
  const auto & q = m.info.origin.orientation;
  return !m.header.frame_id.empty() && m.info.resolution > 0 &&
    std::isfinite(m.info.resolution) && m.info.width > 0 && m.info.height > 0 &&
    std::isfinite(m.info.origin.position.x) && std::isfinite(m.info.origin.position.y) &&
    std::abs(q.x) < 1e-6 && std::abs(q.y) < 1e-6 && std::abs(q.z) < 1e-6 &&
    std::abs(std::abs(q.w) - 1) < 1e-6 &&
    m.data.size() == static_cast<size_t>(m.info.width) * m.info.height &&
    std::all_of(m.data.begin(), m.data.end(), [](int8_t v) {return v >= -1 && v <= 100;});
}

inline std::vector<std::pair<int, int>> diskOffsets(double radius, double resolution)
{
  std::vector<std::pair<int, int>> out;
  const int n = static_cast<int>(std::ceil(radius / resolution + std::sqrt(2.0)));
  // Include cells touching the footprint, not only cells whose centres lie inside.
  const double r = radius + std::sqrt(2.0) * resolution;
  for (int y = -n; y <= n; ++y) {
    for (int x = -n; x <= n; ++x) {
      if (std::hypot(x * resolution, y * resolution) <= r) out.emplace_back(x, y);
    }
  }
  return out;
}

inline std::vector<uint8_t> inflate(
  const Grid & g, const std::vector<uint8_t> & blocked, double radius, bool outside_blocked)
{
  if (blocked.size() != g.size()) throw std::invalid_argument("Invalid occupancy size");
  const auto offsets = diskOffsets(radius, g.info.resolution);
  std::vector<uint8_t> result(g.size(), 0);
  const int w = g.info.width, h = g.info.height;
  const int border = static_cast<int>(std::ceil(radius / g.info.resolution + std::sqrt(2.0)));
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (outside_blocked && (x < border || y < border || x >= w - border || y >= h - border)) {
        result[y * w + x] = 1;
      }
      if (!blocked[y * w + x]) continue;
      for (const auto & d : offsets) {
        const int xx = x + d.first, yy = y + d.second;
        if (xx >= 0 && yy >= 0 && xx < w && yy < h) result[yy * w + xx] = 1;
      }
    }
  }
  return result;
}

struct Motion
{
  bool tracked{true};
  double yaw_gain{0.55}, yaw_gain_min{0.35}, v_ref{1.2}, v_loss{0.25}, w_max{0.7};
  void integrate(double & x, double & y, double & yaw, double v, double w, double dt) const
  {
    if (tracked) {
      const double s = std::clamp(std::abs(v) / v_ref, 0.0, 3.0);
      const double gain = std::clamp(yaw_gain / (1 + 0.6 * s), yaw_gain_min, 1.0);
      const double turn = std::min(1.0, std::abs(w) / w_max);
      v *= 1 - std::clamp(v_loss * turn, 0.0, 0.8);
      w = std::clamp(w * gain, -w_max, w_max);
    }
    x += v * std::cos(yaw) * dt;
    y += v * std::sin(yaw) * dt;
    yaw = wrap(yaw + w * dt);
  }
};

template<typename Safe>
bool advanceSafe(
  const Motion & motion, double resolution, double & x, double & y, double & yaw,
  double v, double w, double duration, Safe safe)
{
  const int n = std::max(1, static_cast<int>(std::ceil(
      std::abs(v) * duration / (0.5 * resolution))));
  for (int j = 0; j < n; ++j) {
    motion.integrate(x, y, yaw, v, w, duration / n);
    if (!safe(x, y, yaw)) return false;
  }
  return true;
}

inline double preferenceCost(
  const Grid & grid, const std::vector<int8_t> & data, double x, double y)
{
  const int i = grid.index(x, y);
  return i < 0 || data[i] < 0 ? 0 : data[i] / 100.0;
}

// Semantics may only add restrictions. Missing preference data never stops the robot; it
// stops only when the producer asks for it or when it last reported obstacles and went silent.
inline bool semanticStopRequired(
  bool enable, bool stop_received, bool stop_flag, bool has_obstacles,
  double now, double obstacles_rx, double ttl)
{
  return enable && ((stop_received && stop_flag) ||
         (has_obstacles && !fresh(now, obstacles_rx, ttl)));
}

// Geometry of an organized PointCloud2 produced by raycasting a pinhole camera on a pixel
// lattice (Unity CameraPointCloudTiledPublisher; ZED registered cloud with step=1, roi=0).
struct CloudLattice
{
  int image_width{1920}, image_height{1080}, pixel_step{3};
  double roi_v_min{0.35};
  int startRow() const
  {
    return std::clamp(static_cast<int>(std::floor(roi_v_min * image_height)), 0, image_height - 1);
  }
  int expectedWidth() const { return (image_width + pixel_step - 1) / pixel_step; }
  int expectedHeight(double roi_v_max = 1.0) const
  {
    const int end = std::clamp(static_cast<int>(std::ceil(roi_v_max * image_height)) - 1,
        0, image_height - 1);
    return (std::max(1, end - startRow() + 1) + pixel_step - 1) / pixel_step;
  }
  bool valid() const
  {
    return image_width > 0 && image_height > 0 && pixel_step > 0 &&
           roi_v_min >= 0 && roi_v_min < 1;
  }
  // Cloud lattice (ix, jy) -> pixel of a mask that keeps the top rows of the same image,
  // possibly scaled uniformly. Returns false for rows removed by the crop (robot hood).
  bool toMask(int ix, int jy, unsigned mask_w, unsigned mask_h, unsigned & u, unsigned & v) const
  {
    const double s = static_cast<double>(mask_w) / image_width;
    const double mu = (ix * pixel_step + 0.5) * s;
    const double mv = (startRow() + jy * pixel_step + 0.5) * s;
    if (mu < 0 || mv < 0 || mu >= mask_w || mv >= mask_h) return false;
    u = static_cast<unsigned>(mu);
    v = static_cast<unsigned>(mv);
    return true;
  }
  // Mask pixel -> cloud lattice cell; false if the pixel lies outside the raycast ROI.
  bool fromMask(unsigned u, unsigned v, unsigned mask_w, int cloud_w, int cloud_h,
    int & ix, int & jy) const
  {
    const double s = static_cast<double>(image_width) / mask_w;
    const double px = (u + 0.5) * s, py = (v + 0.5) * s - startRow();
    if (py < 0) return false;
    ix = static_cast<int>(px / pixel_step);
    jy = static_cast<int>(py / pixel_step);
    return ix >= 0 && jy >= 0 && ix < cloud_w && jy < cloud_h;
  }
};

struct ObstacleCell { double last_seen{0}; uint8_t cls{255}; };
struct CellObservation { int solid{0}, free{0}; uint8_t solid_cls{255}; };

// Solid evidence (re)marks a cell. A retained cell is cleared only by enough ground-level,
// non-solid observations in the same frame, or by the optional maximum age (0 disables).
template<typename Map, typename ObsMap>
void updateObstacleCells(Map & cells, const ObsMap & observations, double now,
  int min_solid, int min_free, double max_age)
{
  for (const auto & kv : observations) {
    const auto & o = kv.second;
    if (o.solid >= min_solid) {
      cells[kv.first] = {now, o.solid_cls};
    } else if (o.solid == 0 && o.free >= min_free) {
      cells.erase(kv.first);
    }
  }
  if (max_age > 0) {
    for (auto it = cells.begin(); it != cells.end(); ) {
      it = now - it->second.last_seen > max_age ? cells.erase(it) : std::next(it);
    }
  }
}

inline std::array<uint8_t, 3> classColor(int id)
{
  // Same bitwise palette as RGBXDataset.get_class_colors used by the overlay.
  uint8_t r = 0, g = 0, b = 0;
  for (int j = 0; j < 7; ++j) {
    r ^= ((id >> 0) & 1) << (7 - j);
    g ^= ((id >> 1) & 1) << (7 - j);
    b ^= ((id >> 2) & 1) << (7 - j);
    id >>= 3;
  }
  return {r, g, b};
}

struct TerrainLimits
{
  double slope{28 * M_PI / 180}, step{0.12}, rough{0.20}, radius{1.54};
  // The original Teacher only penalises unknown cells (conf); blocking them is optional.
  bool unknown_blocked{false};
};

struct Terrain
{
  Grid grid;
  std::vector<float> elevation, risk, confidence, uphill, slope;
  std::vector<uint8_t> blocked;
  double stamp{0};
};

inline Terrain makeTerrain(const grid_map::GridMap & map, const TerrainLimits & limits)
{
  if (!map.exists("elevation")) throw std::invalid_argument("GridMap has no elevation layer");
  Terrain t;
  t.grid = layout(map);
  t.stamp = map.getTimestamp() * 1e-9;
  const size_t n = t.grid.size();
  t.elevation.assign(n, NAN);
  t.risk.assign(n, 1);
  t.confidence.assign(n, 0);
  t.uphill.assign(n, NAN);
  t.slope.assign(n, NAN);
  std::vector<uint8_t> raw(n, limits.unknown_blocked ? 1 : 0), centre_blocked(n, 0);
  for (size_t i = 0; i < n; ++i) {
    const auto xy = t.grid.position(i);
    grid_map::Index index;
    if (!map.getIndex(grid_map::Position(xy.first, xy.second), index)) continue;
    const auto sample = [&](const std::string & layer) -> double {
      return map.exists(layer) ? map.at(layer, index) : NAN;
    };
    const double z = sample("elevation");
    t.elevation[i] = z;
    if (!std::isfinite(z)) continue;
    const double eps = map.getResolution();
    const auto neighbour = [&](double dx, double dy) -> double {
      const grid_map::Position p(xy.first + dx, xy.second + dy);
      grid_map::Index idx;
      return map.getIndex(p, idx) ? map.at("elevation", idx) : NAN;
    };
    const double left = neighbour(-eps, 0), right = neighbour(eps, 0);
    const double down = neighbour(0, -eps), up = neighbour(0, eps);
    const bool stencil = std::isfinite(left) && std::isfinite(right) &&
      std::isfinite(down) && std::isfinite(up);
    double slope = sample("slope"), rough = sample("rough"), step = sample("step");
    if (stencil) {
      if (!std::isfinite(slope)) {
        slope = std::atan(std::hypot((right - left) / (2 * eps), (up - down) / (2 * eps)));
      }
      if (!std::isfinite(rough)) rough = std::abs(z - (left + right + down + up) / 4);
      if (!std::isfinite(step)) {
        step = std::max({std::abs(z - left), std::abs(z - right),
          std::abs(z - down), std::abs(z - up)});
      }
    }
    if (!std::isfinite(slope) || !std::isfinite(rough) || !std::isfinite(step)) continue;
    if (stencil) t.uphill[i] = std::atan2(up - down, right - left);
    t.slope[i] = slope;
    const double sn = std::abs(slope) / limits.slope, rn = std::abs(rough) / limits.rough;
    const double st = std::abs(step) / limits.step;
    const double clearance = sample("clearance"), hazard = sample("hazard_inflated");
    const double clear_def = std::isfinite(clearance) ?
      std::clamp((0.85 - clearance) / 0.85, 0.0, 1.0) : 0;
    t.risk[i] = std::clamp(0.6 * std::min(sn, 1.0) + 0.25 * std::min(st, 1.0) +
      0.1 * std::min(rn, 1.0) + 0.05 * clear_def, 0.0, 1.0);
    t.confidence[i] = 1;
    raw[i] = sn >= 1 || rn >= 1 || st >= 1;
    // hazard_inflated and clearance already account for the footprint.
    centre_blocked[i] = (std::isfinite(hazard) && hazard >= 0.8) ||
      (std::isfinite(clearance) && clearance < limits.radius);
  }
  t.blocked = inflate(t.grid, raw, limits.radius, true);
  for (size_t i = 0; i < n; ++i) t.blocked[i] |= centre_blocked[i];
  return t;
}
}  // namespace terrain_semantic
