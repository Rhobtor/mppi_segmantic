// Projects the segmentation mask onto the egocentric elevation-map grid using the organized
// PointCloud2 of the same camera (Unity /camera/points or ZED cloud_registered). No depth image
// or CameraInfo is needed: the cloud lattice already gives the 3D point of every sampled pixel.
#include "terrain_mppi_segmentation/terrain_semantic_core.hpp"

#include <rclcpp/rclcpp.hpp>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <cstring>
#include <sstream>
#include <unordered_map>

namespace ts = terrain_semantic;

class SemanticCostNode : public rclcpp::Node
{
public:
  SemanticCostNode()
  : Node("semantic_cost_node"), tf_(get_clock()), listener_(tf_)
  {
    frame_ = declare_parameter<std::string>("fixed_frame", "map");
    base_ = declare_parameter<std::string>("base_frame", "base_link");
    lattice_.image_width = declare_parameter<int>("cloud_image_width", 1920);
    lattice_.image_height = declare_parameter<int>("cloud_image_height", 1080);
    lattice_.pixel_step = declare_parameter<int>("cloud_pixel_step", 3);
    lattice_.roi_v_min = declare_parameter<double>("cloud_roi_v_min", 0.35);
    roi_v_max_ = declare_parameter<double>("cloud_roi_v_max", 1.0);
    sync_tol_ = declare_parameter<double>("sync_tolerance_sec", 0.6);
    header_sync_ = declare_parameter<bool>("sync_use_header_stamps", false);
    tf_latest_ = declare_parameter<bool>("tf_use_latest", true);
    min_range_ = declare_parameter<double>("min_range_m", 0.3);
    max_range_ = declare_parameter<double>("max_range_m", 15.0);
    min_obs_h_ = declare_parameter<double>("min_obstacle_height_m", 0.15);
    max_obs_h_ = declare_parameter<double>("max_obstacle_height_m", 2.5);
    ground_tol_ = declare_parameter<double>("ground_tolerance_m", 0.20);
    min_solid_ = declare_parameter<int>("min_obstacle_points", 3);
    min_free_ = declare_parameter<int>("min_free_points", 3);
    min_terrain_ = declare_parameter<int>("min_terrain_points", 2);
    terrain_ttl_ = declare_parameter<double>("terrain_ttl_sec", 4.0);
    obstacle_max_age_ = declare_parameter<double>("obstacle_max_age_sec", 0.0);
    input_timeout_ = declare_parameter<double>("input_timeout_sec", 1.5);
    stale_stop_radius_ = declare_parameter<double>("stale_stop_radius_m", 8.0);
    caution_min_px_ = declare_parameter<int>("caution_min_pixels", 400);
    caution_scale_ = declare_parameter<double>("caution_speed_scale", 0.4);
    caution_hold_ = declare_parameter<double>("caution_hold_sec", 2.0);
    caution_stride_ = declare_parameter<int>("caution_pixel_stride", 4);
    const double rate = declare_parameter<double>("publish_rate_hz", 5.0);
    // IDs from RGBT-Transformer-SAR/V2_2/config.py (C.class_names).
    costs_ = declare_parameter<std::vector<double>>("class_costs",
        {-1.0, -1.0, 0.60, 0.0, 0.0, -1.0, 0.25, -1.0, -1.0, -1.0, -1.0, -1.0});
    const auto solids = declare_parameter<std::vector<int64_t>>("obstacle_class_ids",
        {0, 1, 5, 7, 8, 9, 10, 11});
    const auto persons = declare_parameter<std::vector<int64_t>>("person_class_ids", {0, 1});
    names_ = declare_parameter<std::vector<std::string>>("class_names",
        {"First-Responder", "Civilian", "Vegetation", "Road", "Dirt-Road", "Building",
          "Undergrowth", "Civilian-Car", "Responder-Vehicle", "Debris", "Command-Post", "Wall"});

    if (!lattice_.valid() || !(roi_v_max_ > lattice_.roi_v_min && roi_v_max_ <= 1) ||
      !(sync_tol_ > 0) || !(min_range_ >= 0 && max_range_ > min_range_) ||
      !(max_obs_h_ > min_obs_h_) || !(ground_tol_ > 0) || min_solid_ < 1 || min_free_ < 1 ||
      min_terrain_ < 1 || !(terrain_ttl_ > 0) || obstacle_max_age_ < 0 || !(input_timeout_ > 0) ||
      stale_stop_radius_ < 0 || caution_min_px_ < 1 || !(caution_scale_ >= 0 && caution_scale_ <= 1) ||
      caution_hold_ < 0 || caution_stride_ < 1 || !(rate > 0) || costs_.empty() || costs_.size() > 255) {
      throw std::invalid_argument("Invalid semantic_cost_node parameters");
    }
    for (double c : costs_) {
      if (!std::isfinite(c) || c > 1) throw std::invalid_argument("class_costs must be <= 1 (-1 = none)");
    }
    solid_.assign(costs_.size(), false);
    person_.assign(costs_.size(), false);
    for (auto id : solids) {
      if (id < 0 || id >= static_cast<int64_t>(costs_.size())) throw std::invalid_argument("bad obstacle id");
      solid_[id] = true;
    }
    for (auto id : persons) {
      if (id < 0 || id >= static_cast<int64_t>(costs_.size())) throw std::invalid_argument("bad person id");
      person_[id] = solid_[id] = true;
    }

    const auto latched = rclcpp::QoS(1).reliable().transient_local();
    cost_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/semantic/costmap", latched);
    obstacle_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/semantic/obstacles", latched);
    stop_pub_ = create_publisher<std_msgs::msg::Bool>("/semantic/stop_required", latched);
    scale_pub_ = create_publisher<std_msgs::msg::Float32>("/semantic/speed_scale", latched);
    status_pub_ = create_publisher<std_msgs::msg::String>("/semantic/status", 1);
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/semantic/class_markers", 1);

    map_sub_ = create_subscription<grid_map_msgs::msg::GridMap>(
      declare_parameter<std::string>("grid_map_topic", "/elevation_map"),
      rclcpp::QoS(1).best_effort(),
      [this](grid_map_msgs::msg::GridMap::ConstSharedPtr m) {onMap(*m);});
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      declare_parameter<std::string>("cloud_topic", "/camera/points"),
      rclcpp::SensorDataQoS().keep_last(1),
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr m) {cloud_ = m; cloud_rx_ = now().seconds();});
    mask_sub_ = create_subscription<sensor_msgs::msg::Image>(
      declare_parameter<std::string>("mask_topic", "/segmentation_cropped/mask"),
      rclcpp::SensorDataQoS().keep_last(1),
      [this](sensor_msgs::msg::Image::ConstSharedPtr m) {process(*m, now().seconds());});
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1 / rate)),
      [this]() {publish();});
    RCLCPP_INFO(get_logger(), "semantic_cost_node: organized cloud %dx%d lattice (step %d, roi %.2f)",
      lattice_.expectedWidth(), lattice_.expectedHeight(roi_v_max_), lattice_.pixel_step,
      lattice_.roi_v_min);
  }

private:
  struct TerrainCell {double t{0}; uint8_t cls{0};};

  tf2_ros::Buffer tf_;
  tf2_ros::TransformListener listener_;
  ts::CloudLattice lattice_;
  std::string frame_, base_, status_{"NO_INPUT"};
  std::vector<double> costs_;
  std::vector<bool> solid_, person_;
  std::vector<std::string> names_;
  double roi_v_max_, sync_tol_, min_range_, max_range_, min_obs_h_, max_obs_h_, ground_tol_;
  double terrain_ttl_, obstacle_max_age_, input_timeout_, stale_stop_radius_;
  double caution_scale_, caution_hold_, caution_until_{0}, last_processed_{0}, cloud_rx_{0};
  int min_solid_, min_free_, min_terrain_, caution_min_px_, caution_stride_;
  bool header_sync_, tf_latest_, have_map_{false};
  size_t frames_{0}, last_unlocalized_{0};
  ts::Grid grid_;
  std::vector<float> elevation_;
  std::unordered_map<int64_t, TerrainCell> terrain_;
  std::unordered_map<int64_t, ts::ObstacleCell> obstacles_;
  sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_;
  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr map_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr mask_sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr cost_pub_, obstacle_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr stop_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr scale_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Global cell key: stable while the egocentric map scrolls.
  int64_t key(double x, double y) const
  {
    const int64_t ix = static_cast<int64_t>(std::floor(x / grid_.info.resolution));
    const int64_t iy = static_cast<int64_t>(std::floor(y / grid_.info.resolution));
    return (ix << 32) ^ (iy & 0xffffffffLL);
  }
  std::pair<double, double> keyCentre(int64_t k) const
  {
    const int64_t ix = k >> 32;
    const int64_t iy = static_cast<int32_t>(k & 0xffffffffLL);
    return {(ix + 0.5) * grid_.info.resolution, (iy + 0.5) * grid_.info.resolution};
  }

  void fail(const std::string & why)
  {
    status_ = why;
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "%s", why.c_str());
  }

  void onMap(const grid_map_msgs::msg::GridMap & msg)
  {
    grid_map::GridMap map;
    if (!grid_map::GridMapRosConverter::fromMessage(msg, map) || !map.exists("elevation")) {
      fail("INVALID_ELEVATION_MAP"); return;
    }
    if (map.getFrameId() != frame_) {
      fail("ELEVATION_MAP_FRAME_" + map.getFrameId() + "_NOT_" + frame_); return;
    }
    ts::Grid g;
    try {g = ts::layout(map);} catch (const std::exception & e) {fail(e.what()); return;}
    if (have_map_ && std::abs(g.info.resolution - grid_.info.resolution) > 1e-6) {
      terrain_.clear(); obstacles_.clear();
    }
    grid_ = g;
    elevation_.assign(grid_.size(), NAN);
    for (size_t i = 0; i < grid_.size(); ++i) {
      const auto p = grid_.position(i);
      grid_map::Index idx;
      if (map.getIndex(grid_map::Position(p.first, p.second), idx)) {
        elevation_[i] = map.at("elevation", idx);
      }
    }
    have_map_ = true;
  }

  void process(const sensor_msgs::msg::Image & mask, double rx)
  {
    if (!have_map_) {fail("NO_ELEVATION_MAP"); return;}
    if (!cloud_) {fail("NO_CLOUD"); return;}
    const auto & cloud = *cloud_;
    const double skew = header_sync_ ?
      std::abs(ts::stampSeconds(mask.header.stamp) - ts::stampSeconds(cloud.header.stamp)) :
      std::abs(rx - cloud_rx_);
    if (skew > sync_tol_) {fail("MASK_CLOUD_OUT_OF_SYNC"); return;}
    if (mask.encoding != "mono8" || !mask.width || !mask.height || mask.step < mask.width ||
      mask.data.size() < static_cast<size_t>(mask.step) * mask.height) {
      fail("MASK_NOT_MONO8"); return;
    }
    const int cw = static_cast<int>(cloud.width), ch = static_cast<int>(cloud.height);
    if (cw != lattice_.expectedWidth() || ch != lattice_.expectedHeight(roi_v_max_)) {
      fail("CLOUD_" + std::to_string(cw) + "x" + std::to_string(ch) + "_NOT_EXPECTED_" +
        std::to_string(lattice_.expectedWidth()) + "x" +
        std::to_string(lattice_.expectedHeight(roi_v_max_)) + "_CHECK_cloud_*_PARAMS");
      return;
    }
    int ox = -1, oy = -1, oz = -1;
    for (const auto & f : cloud.fields) {
      if (f.datatype != sensor_msgs::msg::PointField::FLOAT32) continue;
      if (f.name == "x") ox = f.offset;
      if (f.name == "y") oy = f.offset;
      if (f.name == "z") oz = f.offset;
    }
    if (ox < 0 || oy < 0 || oz < 0 || cloud.is_bigendian ||
      cloud.point_step < 12 || cloud.row_step < cloud.width * cloud.point_step ||
      cloud.data.size() < static_cast<size_t>(cloud.row_step) * cloud.height) {
      fail("CLOUD_FIELDS_INVALID"); return;
    }
    tf2::Transform T;
    try {
      const auto msg = tf_.lookupTransform(frame_, cloud.header.frame_id,
          tf_latest_ ? tf2::TimePointZero : tf2_ros::fromMsg(cloud.header.stamp),
          tf2::durationFromSec(tf_latest_ ? 0.0 : 0.05));
      tf2::fromMsg(msg.transform, T);
    } catch (const tf2::TransformException & e) {
      fail(std::string("CLOUD_TF: ") + e.what()); return;
    }
    const auto point = [&](int ix, int jy, tf2::Vector3 & p) {
      const size_t off = static_cast<size_t>(jy) * cloud.row_step +
        static_cast<size_t>(ix) * cloud.point_step;
      float xyz[3];
      std::memcpy(&xyz[0], &cloud.data[off + ox], 4);
      std::memcpy(&xyz[1], &cloud.data[off + oy], 4);
      std::memcpy(&xyz[2], &cloud.data[off + oz], 4);
      if (!std::isfinite(xyz[0]) || !std::isfinite(xyz[1]) || !std::isfinite(xyz[2])) return false;
      p.setValue(xyz[0], xyz[1], xyz[2]);
      const double r = p.length();
      return r >= min_range_ && r <= max_range_;
    };

    const size_t n = costs_.size();
    std::unordered_map<int64_t, ts::CellObservation> obs;
    std::unordered_map<int64_t, std::vector<uint16_t>> votes;
    tf2::Vector3 p;
    for (int jy = 0; jy < ch; ++jy) {
      for (int ix = 0; ix < cw; ++ix) {
        unsigned u, v;
        if (!lattice_.toMask(ix, jy, mask.width, mask.height, u, v)) continue;
        const uint8_t cls = mask.data[static_cast<size_t>(v) * mask.step + u];
        if (cls >= n || !point(ix, jy, p)) continue;
        const tf2::Vector3 q = T * p;
        const int i = grid_.index(q.x(), q.y());
        if (i < 0) continue;
        const double ground = elevation_[i];
        const double h = std::isfinite(ground) ? q.z() - ground : NAN;
        const int64_t k = key(q.x(), q.y());
        if (solid_[cls]) {
          // Unknown ground height: keep solid evidence (conservative), never use it to clear.
          const bool ok = !std::isfinite(h) ||
            (h < max_obs_h_ && h >= (person_[cls] ? -ground_tol_ : min_obs_h_));
          if (ok) {
            auto & o = obs[k];
            ++o.solid;
            if (o.solid_cls == 255 || person_[cls]) o.solid_cls = cls;
          }
          continue;
        }
        if (std::isfinite(h) && std::abs(h) <= ground_tol_) ++obs[k].free;
        if (costs_[cls] >= 0 && (!std::isfinite(h) || h < max_obs_h_)) {
          auto & vv = votes[k];
          if (vv.empty()) vv.assign(n, 0);
          ++vv[cls];
        }
      }
    }

    // Person pixels without a valid 3D point (above the cloud ROI, NaN, out of range) cannot be
    // placed on the map: reduce speed instead of stopping forever.
    size_t unlocalized = 0;
    for (unsigned v = 0; v < mask.height; v += caution_stride_) {
      for (unsigned u = 0; u < mask.width; u += caution_stride_) {
        const uint8_t cls = mask.data[static_cast<size_t>(v) * mask.step + u];
        if (cls >= n || !person_[cls]) continue;
        int ix, jy;
        if (!lattice_.fromMask(u, v, mask.width, cw, ch, ix, jy) || !point(ix, jy, p)) ++unlocalized;
      }
    }
    unlocalized *= static_cast<size_t>(caution_stride_) * caution_stride_;
    last_unlocalized_ = unlocalized;
    if (unlocalized >= static_cast<size_t>(caution_min_px_)) caution_until_ = rx + caution_hold_;

    ts::updateObstacleCells(obstacles_, obs, rx, min_solid_, min_free_, obstacle_max_age_);
    for (const auto & kv : votes) {
      int total = 0, best = -1;
      for (size_t c = 0; c < n; ++c) {
        total += kv.second[c];
        if (best < 0 || kv.second[c] > kv.second[best]) best = static_cast<int>(c);
      }
      if (total >= min_terrain_) terrain_[kv.first] = {rx, static_cast<uint8_t>(best)};
    }
    last_processed_ = rx;
    ++frames_;
    status_ = "OK";
  }

  void publish()
  {
    const double t = now().seconds();
    const bool stale = frames_ == 0 || t - last_processed_ > input_timeout_;
    std_msgs::msg::Float32 scale;
    scale.data = t < caution_until_ ? caution_scale_ : 1.0f;
    std_msgs::msg::Bool stop;
    if (!have_map_) {
      stop_pub_->publish(stop);
      scale_pub_->publish(scale);
      publishStatus(stale, false, scale.data);
      return;
    }
    for (auto it = terrain_.begin(); it != terrain_.end(); ) {
      const auto c = keyCentre(it->first);
      it = (t - it->second.t > terrain_ttl_ || grid_.index(c.first, c.second) < 0) ?
        terrain_.erase(it) : std::next(it);
    }
    for (auto it = obstacles_.begin(); it != obstacles_.end(); ) {
      const auto c = keyCentre(it->first);
      it = grid_.index(c.first, c.second) < 0 ? obstacles_.erase(it) : std::next(it);
    }
    nav_msgs::msg::OccupancyGrid cost, occ;
    cost.header.frame_id = frame_;
    cost.header.stamp = now();
    cost.info = grid_.info;
    cost.data.assign(grid_.size(), -1);
    occ.header = cost.header;
    occ.info = grid_.info;
    occ.data.assign(grid_.size(), 0);
    for (size_t i = 0; i < grid_.size(); ++i) {
      const auto c = grid_.position(i);
      const int64_t k = key(c.first, c.second);
      const auto tt = terrain_.find(k);
      if (tt != terrain_.end()) {
        cost.data[i] = static_cast<int8_t>(std::lround(100 * costs_[tt->second.cls]));
      }
      if (obstacles_.count(k)) occ.data[i] = 100;
    }
    // Retained obstacles with no fresh observations: cannot prove the area is free.
    if (stale && !obstacles_.empty()) {
      try {
        const auto b = tf_.lookupTransform(frame_, base_, tf2::TimePointZero);
        for (const auto & kv : obstacles_) {
          const auto c = keyCentre(kv.first);
          if (std::hypot(c.first - b.transform.translation.x,
            c.second - b.transform.translation.y) <= stale_stop_radius_) {stop.data = true; break;}
        }
      } catch (const tf2::TransformException &) {
        stop.data = true;
      }
    }
    cost_pub_->publish(cost);
    obstacle_pub_->publish(occ);
    stop_pub_->publish(stop);
    scale_pub_->publish(scale);
    publishMarkers(cost.header);
    publishStatus(stale, stop.data, scale.data);
  }

  void publishStatus(bool stale, bool stop, double scale)
  {
    std_msgs::msg::String s;
    std::ostringstream o;
    o << status_ << "; frames=" << frames_ << "; stale=" << stale << "; stop=" << stop
      << "; speed_scale=" << scale << "; terrain_cells=" << terrain_.size()
      << "; obstacle_cells=" << obstacles_.size() << "; unlocalized_person_px=" << last_unlocalized_;
    s.data = o.str();
    status_pub_->publish(s);
  }

  void publishMarkers(const std_msgs::msg::Header & header)
  {
    visualization_msgs::msg::MarkerArray a;
    visualization_msgs::msg::Marker m;
    m.header = header;
    m.type = visualization_msgs::msg::Marker::CUBE_LIST;
    m.pose.orientation.w = 1;
    m.scale.x = m.scale.y = grid_.info.resolution;
    m.lifetime = rclcpp::Duration::from_seconds(1.0);
    const auto z = [&](double x, double y) {
      const int i = grid_.index(x, y);
      return i >= 0 && std::isfinite(elevation_[i]) ? elevation_[i] : 0.0;
    };
    const auto add = [&](double x, double y, double zz, int cls, double alpha) {
      geometry_msgs::msg::Point p; p.x = x; p.y = y; p.z = zz;
      const auto rgb = ts::classColor(cls);
      std_msgs::msg::ColorRGBA c;
      c.r = rgb[0] / 255.0f; c.g = rgb[1] / 255.0f; c.b = rgb[2] / 255.0f; c.a = alpha;
      m.points.push_back(p); m.colors.push_back(c);
    };
    m.ns = "terrain_classes"; m.id = 0; m.scale.z = 0.02;
    for (const auto & kv : terrain_) {
      const auto c = keyCentre(kv.first);
      add(c.first, c.second, z(c.first, c.second) + 0.03, kv.second.cls, 0.6);
    }
    a.markers.push_back(m);
    m.points.clear(); m.colors.clear();
    m.ns = "semantic_obstacles"; m.id = 1; m.scale.z = 1.0;
    for (const auto & kv : obstacles_) {
      const auto c = keyCentre(kv.first);
      add(c.first, c.second, z(c.first, c.second) + 0.5, kv.second.cls, 0.9);
    }
    a.markers.push_back(m);
    marker_pub_->publish(a);
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<SemanticCostNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("semantic_cost_node"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
