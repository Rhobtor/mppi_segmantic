// Independent terrain-only derivative of mppi_teacher_3_rings_subpoint_bom.cpp.
#include "terrain_mppi_segmentation/terrain_semantic_core.hpp"

#include <rclcpp/rclcpp.hpp>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <memory>
#include <random>

namespace ts = terrain_semantic;

class TerrainMppiSegmentation : public rclcpp::Node
{
public:
  TerrainMppiSegmentation()
  : Node("terrain_mppi_segmentation"), tf_(get_clock()), listener_(tf_)
  {
    semantic_ = declare_parameter<bool>("semantic_enable", false);
    base_ = declare_parameter<std::string>("base_frame", "base_link");
    frame_ = declare_parameter<std::string>("fixed_frame", "map");
    dt_ = declare_parameter<double>("roll_dt", 0.1);
    steps_ = declare_parameter<int>("roll_T", 65);
    samples_ = declare_parameter<int>("roll_K", 220);
    v_max_ = declare_parameter<double>("v_max", 0.7);
    reverse_max_ = declare_parameter<double>("reverse_v_max", 0.6);
    allow_reverse_ = declare_parameter<bool>("allow_reverse", true);
    motion_.w_max = declare_parameter<double>("w_max", 0.7);
    const auto model = declare_parameter<std::string>("motion_model", "tracked");
    if (model != "tracked" && model != "unicycle") {
      throw std::invalid_argument("motion_model must be tracked or unicycle");
    }
    motion_.tracked = model == "tracked";
    motion_.yaw_gain = declare_parameter<double>("tracked_yaw_gain", 0.55);
    motion_.yaw_gain_min = declare_parameter<double>("tracked_yaw_gain_min", 0.35);
    motion_.v_ref = declare_parameter<double>("tracked_yaw_v_ref", 1.2);
    motion_.v_loss = declare_parameter<double>("tracked_v_loss_w_gain", 0.25);
    acceleration_ = declare_parameter<double>("a_lin_max", 1.5);
    yaw_acceleration_ = declare_parameter<double>("a_yaw_max", 3.0);
    sigma_v_ = declare_parameter<double>("sigma_v", 0.4);
    sigma_k_ = declare_parameter<double>("sigma_k", 0.7);
    kappa_max_ = declare_parameter<double>("kappa_geom_max", 8.5);
    limits_.radius = declare_parameter<double>("footprint_radius", 1.47) +
      declare_parameter<double>("footprint_margin", 0.07);
    limits_.slope = declare_parameter<double>("max_slope_deg", 28.0) * M_PI / 180;
    limits_.step = declare_parameter<double>("max_step_m", 0.12);
    limits_.rough = declare_parameter<double>("max_rough_m", 0.20);
    limits_.unknown_blocked = declare_parameter<bool>("unknown_blocked", false);
    w_trav_ = declare_parameter<double>("w_trav", 6.0);
    w_unknown_ = declare_parameter<double>("w_unknown", 3.0);
    w_obs_ = declare_parameter<double>("w_obs", 4.0);
    w_goal_ = declare_parameter<double>("w_dterm", 1.0);
    w_progress_ = declare_parameter<double>("w_progress", 2.0);
    w_heading_ = declare_parameter<double>("w_hfinal", 1.5);
    w_control_ = declare_parameter<double>("w_control", 0.005);
    w_change_ = declare_parameter<double>("w_control_change", 0.002);
    w_semantic_ = declare_parameter<double>("w_semantic", 2.0);
    ramp_align_ = declare_parameter<bool>("ramp_align_enable", true);
    ramp_slope_ = declare_parameter<double>("ramp_slope_rad", 0.15);
    ramp_tolerance_ = declare_parameter<double>("ramp_align_tolerance_deg", 18.0) * M_PI / 180;
    ramp_weight_ = declare_parameter<double>("ramp_align_weight", 6.0);
    max_roll_ = declare_parameter<double>("max_cross_slope_deg", 20.0) * M_PI / 180;
    hysteresis_ = declare_parameter<double>("action_hysteresis_abs", 0.10);
    goal_tolerance_ = declare_parameter<double>("goal_reached_dist", 1.0);
    map_ttl_ = declare_parameter<double>("map_timeout_sec", 0.8);
    semantic_ttl_ = declare_parameter<double>("semantic_timeout_sec", 0.8);
    // <= 0 disables the check, as in the original Teacher (TimePointZero lookups).
    tf_ttl_ = declare_parameter<double>("tf_timeout_sec", 0.0);
    control_hz_ = declare_parameter<double>("ctrl_rate_hz", 10.0);
    seed_ = declare_parameter<int>("random_seed", 42);
    rng_.seed(seed_);
    for (double p : {dt_, v_max_, reverse_max_, motion_.w_max, acceleration_, yaw_acceleration_,
      limits_.radius, limits_.slope, limits_.step, limits_.rough, motion_.v_ref,
      motion_.yaw_gain_min, motion_.yaw_gain, motion_.v_loss, control_hz_, map_ttl_,
      semantic_ttl_, tf_ttl_, kappa_max_, sigma_v_, sigma_k_, ramp_slope_, ramp_tolerance_,
      ramp_weight_, max_roll_, goal_tolerance_}) {
      if (!std::isfinite(p)) throw std::invalid_argument("Non-finite terrain controller parameter");
    }
    if (!(dt_ > 0) || steps_ < 3 || samples_ < 8 || samples_ > 5000 ||
      steps_ > 500 || !(v_max_ > 0) || !(reverse_max_ >= 0) || !(motion_.w_max > 0) ||
      !(acceleration_ > 0) || !(yaw_acceleration_ > 0) || !(limits_.radius > 0) ||
      !(limits_.slope > 0 && limits_.slope < M_PI / 2) ||
      !(limits_.step > 0) || !(limits_.rough > 0) ||
      !(motion_.v_ref > 0) || !(motion_.yaw_gain_min > 0 && motion_.yaw_gain_min <= 1) ||
      !(motion_.yaw_gain > 0 && motion_.yaw_gain <= 1) ||
      !(motion_.v_loss >= 0 && motion_.v_loss <= 0.8) ||
      !(control_hz_ > 0) || !(map_ttl_ > 0) || !(semantic_ttl_ > 0) ||
      !(kappa_max_ > 0) || !(sigma_v_ >= 0) || !(sigma_k_ >= 0) ||
      !(ramp_slope_ >= 0) || !(ramp_tolerance_ >= 0) ||
      !std::isfinite(ramp_weight_) || !(ramp_weight_ >= 0) ||
      !(max_roll_ > 0 && max_roll_ < M_PI / 2) || !(goal_tolerance_ > 0)) {
      throw std::invalid_argument("Invalid terrain controller parameters");
    }
    for (double w : {w_trav_, w_unknown_, w_obs_, w_goal_, w_progress_, w_heading_,
      w_control_, w_change_, w_semantic_, hysteresis_}) {
      if (!std::isfinite(w) || w < 0) throw std::invalid_argument("Costs must be nonnegative and finite");
    }
    cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>(
      declare_parameter<std::string>("cmd_topic", "/cmd_vel"), 1);
    debug_pub_ = create_publisher<std_msgs::msg::String>("/terrain_mppi_segmentation/decision", 1);
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/terrain_mppi_segmentation/rollouts", 1);
    goal_pub_ = create_publisher<std_msgs::msg::Bool>(
      "/terrain_mppi_segmentation/goal_reached", 1);
    map_sub_ = create_subscription<grid_map_msgs::msg::GridMap>(
      declare_parameter<std::string>("grid_map_topic", "/elevation_map"),
      rclcpp::QoS(1).best_effort(),
      [this](grid_map_msgs::msg::GridMap::ConstSharedPtr m) {
        try {
          grid_map::GridMap map;
          if (!grid_map::GridMapRosConverter::fromMessage(*m, map) || map.getFrameId() != frame_) {
            throw std::invalid_argument("GridMap must use configured stationary fixed_frame");
          }
          if (have_map_ && map.getTimestamp() * 1e-9 < terrain_.stamp) {
            RCLCPP_WARN(get_logger(), "Ignoring out-of-order terrain map");
            return;
          }
          terrain_ = ts::makeTerrain(map, limits_);
          map_rx_ = now().seconds();
          have_map_ = true;
          semantic_dirty_ = true;
        } catch (const std::invalid_argument & e) {
          have_map_ = false;
          stop(std::string("INVALID_TERRAIN: ") + e.what());
        }
      });
    goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      declare_parameter<std::string>("goal_topic_ps", "/goal_ps"), 1,
      [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr g) {setGoal(*g);});
    goals_sub_ = create_subscription<geometry_msgs::msg::PoseArray>(
      declare_parameter<std::string>("goal_topic_pa", "/goal_local"), 1,
      [this](geometry_msgs::msg::PoseArray::ConstSharedPtr g) {
        if (g->poses.empty()) {have_goal_ = false; stop("EMPTY_GOAL"); return;}
        geometry_msgs::msg::PoseStamped p;
        p.header = g->header;
        p.pose = g->poses.front();
        setGoal(p);
      });
    if (semantic_) {
      const auto qos = rclcpp::QoS(1).reliable().transient_local();
      cost_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        "/semantic/costmap", qos,
        [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr m) {
          if (!ts::validGrid(*m) || m->header.frame_id != frame_) {
            cost_.reset();
            RCLCPP_ERROR(get_logger(), "Invalid semantic preference grid; preferences disabled");
            return;
          }
          cost_ = m;
          cost_rx_ = now().seconds();
          cost_grid_ = {m->info, m->header.frame_id};
        });
      obstacle_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        "/semantic/obstacles", qos,
        [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr m) {
          semantic_grid_fault_ = !ts::validGrid(*m) || m->header.frame_id != frame_;
          if (semantic_grid_fault_) {
            RCLCPP_ERROR(get_logger(), "Invalid semantic obstacle grid; stopping");
            return;
          }
          obstacles_ = m;
          obstacles_rx_ = now().seconds();
          semantic_dirty_ = true;
        });
      safety_sub_ = create_subscription<std_msgs::msg::Bool>(
        "/semantic/stop_required", qos,
        [this](std_msgs::msg::Bool::ConstSharedPtr m) {
          stop_received_ = true;
          stop_flag_ = m->data;
        });
      scale_sub_ = create_subscription<std_msgs::msg::Float32>(
        "/semantic/speed_scale", qos,
        [this](std_msgs::msg::Float32::ConstSharedPtr m) {
          if (std::isfinite(m->data)) speed_scale_ = std::clamp<double>(m->data, 0.0, 1.0);
        });
    }
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1 / control_hz_)), [this]() {tick();});
    RCLCPP_INFO(get_logger(), "Independent minimum-cost rollout controller; semantics=%s; no SAC/rings",
      semantic_ ? "enabled" : "disabled");
  }

private:
  struct Command {double v{0}, w{0};};
  struct Evaluation {
    double cost{std::numeric_limits<double>::infinity()}, semantic_cost{0};
    Command first;
    std::vector<std::pair<double, double>> path;
  };
  tf2_ros::Buffer tf_;
  tf2_ros::TransformListener listener_;
  ts::Terrain terrain_;
  ts::TerrainLimits limits_;
  ts::Motion motion_;
  std::string base_, frame_;
  bool semantic_, allow_reverse_, ramp_align_, cost_active_{false};
  bool have_map_{false}, have_goal_{false}, semantic_dirty_{true};
  bool semantic_grid_fault_{false}, stop_received_{false}, stop_flag_{false};
  bool semantic_has_obstacles_{false};
  geometry_msgs::msg::PoseStamped goal_;
  std::vector<uint8_t> combined_;
  Command last_;
  double last_tick_{0}, map_rx_{0}, cost_rx_{0}, obstacles_rx_{0}, speed_scale_{1.0};
  double dt_, v_max_, reverse_max_, acceleration_, yaw_acceleration_, sigma_v_, sigma_k_, kappa_max_;
  double w_trav_, w_unknown_, w_obs_, w_goal_, w_progress_, w_heading_, w_control_, w_change_;
  double w_semantic_, hysteresis_, goal_tolerance_, map_ttl_, semantic_ttl_, tf_ttl_, control_hz_;
  double ramp_slope_, ramp_tolerance_, ramp_weight_, max_roll_;
  ts::Grid cost_grid_;
  int steps_, samples_, seed_;
  std::mt19937 rng_;
  nav_msgs::msg::OccupancyGrid::ConstSharedPtr cost_, obstacles_;
  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr map_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr goals_sub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr cost_sub_, obstacle_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr safety_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr scale_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr debug_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr goal_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  void setGoal(const geometry_msgs::msg::PoseStamped & g)
  {
    if (g.header.frame_id.empty() || !std::isfinite(g.pose.position.x) ||
      !std::isfinite(g.pose.position.y)) {
      have_goal_ = false;
      stop("INVALID_GOAL");
      return;
    }
    try {
      goal_ = g.header.frame_id == frame_ ? g :
        tf_.transform(g, frame_, tf2::durationFromSec(0));
      have_goal_ = true;
    } catch (const tf2::TransformException & e) {
      have_goal_ = false;
      stop(std::string("GOAL_TF: ") + e.what());
    }
  }

  void publish(Command c, const std::string & reason)
  {
    geometry_msgs::msg::Twist out;
    out.linear.x = c.v;
    out.angular.z = c.w;
    cmd_pub_->publish(out);
    last_ = c;
    std_msgs::msg::String s;
    s.data = reason + "; v=" + std::to_string(c.v) + "; w=" + std::to_string(c.w);
    debug_pub_->publish(s);
  }
  void stop(const std::string & reason)
  {
    publish({}, reason);
    if (marker_pub_) {
      visualization_msgs::msg::MarkerArray a;
      visualization_msgs::msg::Marker m;
      m.action = visualization_msgs::msg::Marker::DELETEALL;
      a.markers.push_back(m);
      marker_pub_->publish(a);
    }
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "%s", reason.c_str());
  }

  void updateSemantic()
  {
    if (!semantic_dirty_) return;
    combined_ = terrain_.blocked;
    semantic_has_obstacles_ = false;
    if (semantic_ && obstacles_) {
      if (std::abs(obstacles_->info.resolution - terrain_.grid.info.resolution) > 1e-5) {
        semantic_grid_fault_ = true;
        RCLCPP_ERROR(get_logger(), "Semantic obstacle resolution does not match terrain");
        return;
      }
      const ts::Grid source{obstacles_->info, obstacles_->header.frame_id};
      std::vector<uint8_t> raw(terrain_.grid.size(), 0);
      for (size_t i = 0; i < obstacles_->data.size(); ++i) {
        if (obstacles_->data[i] < 50) continue;
        semantic_has_obstacles_ = true;
        const auto xy = source.position(i);
        const int j = terrain_.grid.index(xy.first, xy.second);
        if (j >= 0) raw[j] = 1;
      }
      auto inflated = ts::inflate(terrain_.grid, raw, limits_.radius, false);
      for (size_t i = 0; i < combined_.size(); ++i) combined_[i] |= inflated[i];
    }
    semantic_dirty_ = false;
  }

  bool safe(double x, double y) const
  {
    const int i = terrain_.grid.index(x, y);
    return i >= 0 && !combined_[i];
  }

  bool advance(double & x, double & y, double & yaw, Command c, double duration) const
  {
    return ts::advanceSafe(motion_, terrain_.grid.info.resolution, x, y, yaw,
      c.v, c.w, duration, [this](double xx, double yy, double th) {
        if (!safe(xx, yy)) return false;
        const int i = terrain_.grid.index(xx, yy);
        const double uphill = terrain_.uphill[i], slope = terrain_.slope[i];
        if (!std::isfinite(uphill) || !std::isfinite(slope)) return true;
        const double cross_slope = std::atan(
          std::abs(std::tan(slope) * std::sin(th - uphill)));
        return cross_slope < max_roll_;
      });
  }

  double semanticCost(double x, double y) const
  {
    return cost_active_ ? ts::preferenceCost(cost_grid_, cost_->data, x, y) : 0;
  }

  Evaluation rolloutEval(
    double bx, double by, double yaw, double gx, double gy,
    Command target, double cycle_dt, bool noise)
  {
    Evaluation e;
    e.path.reserve(steps_);
    double x = bx, y = by, th = yaw, cost = 0, prev_dist = std::hypot(gx - x, gy - y);
    Command c = last_;
    std::normal_distribution<double> normal(0, 1);
    double k = std::abs(target.v) > 0.05 ? target.w / target.v : 0;
    const double k_prior = k;
    for (int t = 0; t < steps_; ++t) {
      if (noise && std::abs(target.v) > 0.05) {
        const double k_limit = std::min(kappa_max_, motion_.w_max / std::abs(target.v));
        k += 0.25 * (k_prior - k) + 1.5 * sigma_k_ * normal(rng_);
        k = std::clamp(k, -k_limit, k_limit);
        target.w = k * target.v;
      }
      const Command before = c;
      const double transition_dt = t == 0 ? cycle_dt : dt_;
      c.v += std::clamp(target.v - c.v, -acceleration_ * transition_dt, acceleration_ * transition_dt);
      c.w += std::clamp(target.w - c.w, -yaw_acceleration_ * transition_dt, yaw_acceleration_ * transition_dt);
      if (t == 0) e.first = c;
      if (!advance(x, y, th, c, dt_)) return e;
      const int i = terrain_.grid.index(x, y);
      const double sem = w_semantic_ * semanticCost(x, y) * dt_;
      e.semantic_cost += sem;
      const auto & g = terrain_.grid;
      const double boundary = std::min({x - g.info.origin.position.x, y - g.info.origin.position.y,
        g.info.origin.position.x + g.info.width * g.info.resolution - x,
        g.info.origin.position.y + g.info.height * g.info.resolution - y});
      const double clearance_deficit = std::max(0.0, limits_.radius - boundary);
      cost += dt_ * (w_trav_ * terrain_.risk[i] + w_unknown_ * (1 - terrain_.confidence[i]) +
        w_obs_ * clearance_deficit * clearance_deficit +
        w_control_ * (c.v * c.v + c.w * c.w)) + sem;
      cost += w_change_ * (std::pow(c.v - before.v, 2) + std::pow(c.w - before.w, 2));
      if (ramp_align_ && std::abs(c.v) > 0.15 &&
        terrain_.slope[i] > ramp_slope_ && std::isfinite(terrain_.uphill[i])) {
        const double misalignment = std::acos(std::clamp(
            std::abs(std::cos(th - terrain_.uphill[i])), 0.0, 1.0));
        cost += ramp_weight_ * std::pow(std::max(0.0, misalignment - ramp_tolerance_), 2) * dt_;
      }
      const double dist = std::hypot(gx - x, gy - y);
      cost -= w_progress_ * (prev_dist - dist);
      prev_dist = dist;
      e.path.emplace_back(x, y);
    }
    cost += w_goal_ * prev_dist + w_heading_ * (1 - std::cos(
      ts::wrap(std::atan2(gy - y, gx - x) - th)));
    e.cost = cost;
    return e;
  }

  void markers(const std::vector<Evaluation> & all, const Evaluation & best)
  {
    visualization_msgs::msg::MarkerArray a;
    visualization_msgs::msg::Marker candidates;
    candidates.header.frame_id = frame_;
    candidates.header.stamp = now();
    candidates.ns = "rollouts";
    candidates.id = 0;
    candidates.type = visualization_msgs::msg::Marker::LINE_LIST;
    candidates.action = visualization_msgs::msg::Marker::ADD;
    candidates.pose.orientation.w = 1;
    candidates.scale.x = 0.025;
    candidates.color.b = 1;
    candidates.color.a = 0.3;
    candidates.lifetime = rclcpp::Duration::from_seconds(2 / control_hz_);
    for (const auto & e : all) {
      if (!std::isfinite(e.cost)) continue;
      for (size_t j = 1; j < e.path.size(); ++j) {
        for (size_t k : {j - 1, j}) {
          geometry_msgs::msg::Point p;
          p.x = e.path[k].first; p.y = e.path[k].second; p.z = 0.1;
          candidates.points.push_back(p);
        }
      }
    }
    a.markers.push_back(candidates);
    candidates.ns = "selected";
    candidates.id = 1;
    candidates.type = visualization_msgs::msg::Marker::LINE_STRIP;
    candidates.scale.x = 0.08;
    candidates.color.b = 0; candidates.color.g = 1; candidates.color.a = 1;
    candidates.points.clear();
    for (const auto & xy : best.path) {
      geometry_msgs::msg::Point p;
      p.x = xy.first; p.y = xy.second; p.z = 0.15;
      candidates.points.push_back(p);
    }
    a.markers.push_back(candidates);
    marker_pub_->publish(a);
  }

  void tick()
  {
    const double now_s = now().seconds();
    const double cycle_dt = last_tick_ > 0 ?
      std::clamp(now_s - last_tick_, 0.001, 2 / control_hz_) : 1 / control_hz_;
    last_tick_ = now_s;
    if (!have_goal_) {stop("NO_GOAL"); return;}
    // Receipt time: Unity/ROS header clocks may differ.
    if (!have_map_ || !ts::fresh(now_s, map_rx_, map_ttl_)) {
      stop("NO_FRESH_METRIC_TERRAIN"); return;
    }
    updateSemantic();
    if (semantic_ && semantic_grid_fault_) {stop("INVALID_SEMANTIC_OBSTACLE_GRID"); return;}
    // Missing/stale preferences only disable J_semantic: geometric navigation continues.
    cost_active_ = semantic_ && cost_ && ts::fresh(now_s, cost_rx_, semantic_ttl_);
    if (ts::semanticStopRequired(semantic_, stop_received_, stop_flag_,
        semantic_has_obstacles_, now_s, obstacles_rx_, semantic_ttl_)) {
      stop("SEMANTIC_UNRESOLVED_OR_STALE_HAZARD"); return;
    }
    const double scale = semantic_ ? speed_scale_ : 1.0;
    const double v_max = v_max_ * scale, reverse_max = reverse_max_ * scale;
    double bx, by, yaw;
    try {
      const auto pose = tf_.lookupTransform(frame_, base_, tf2::TimePointZero);
      const double pose_stamp = ts::stampSeconds(pose.header.stamp);
      if (tf_ttl_ > 0 && pose_stamp != 0 && !ts::fresh(now_s, pose_stamp, tf_ttl_)) {
        stop("STALE_ROBOT_TF"); return;
      }
      bx = pose.transform.translation.x; by = pose.transform.translation.y;
      tf2::Quaternion q;
      tf2::fromMsg(pose.transform.rotation, q);
      double roll, pitch;
      tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    } catch (const tf2::TransformException & e) {
      stop(std::string("ROBOT_TF: ") + e.what()); return;
    }
    const double gx = goal_.pose.position.x, gy = goal_.pose.position.y;
    std_msgs::msg::Bool reached;
    reached.data = std::hypot(gx - bx, gy - by) <= goal_tolerance_;
    goal_pub_->publish(reached);
    if (reached.data) {stop("GOAL_REACHED"); return;}
    if (!safe(bx, by)) {stop("CURRENT_FOOTPRINT_UNSAFE"); return;}
    std::vector<Evaluation> all;
    all.reserve(samples_ + 1);
    const Command keep{std::clamp(last_.v, -reverse_max, v_max), last_.w};
    auto best = rolloutEval(bx, by, yaw, gx, gy, keep, cycle_dt, false);
    const double heading = ts::wrap(std::atan2(gy - by, gx - bx) - yaw);
    std::normal_distribution<double> normal(0, 1);
    for (int k = 0; k < samples_; ++k) {
      Command target;
      if (k < 3) {
        target = {0, (k - 1) * motion_.w_max};
      } else {
        target.v = std::clamp(v_max + sigma_v_ * normal(rng_),
          allow_reverse_ ? -reverse_max : 0.0, v_max);
        if (allow_reverse_ && k % 8 == 0) target.v = -reverse_max;
        target.w = std::clamp(heading / std::max(dt_ * steps_, 0.1) +
          motion_.w_max * (2.0 * (k - 3) / std::max(1, samples_ - 4) - 1),
          -motion_.w_max, motion_.w_max);
      }
      auto e = rolloutEval(bx, by, yaw, gx, gy, target, cycle_dt, k >= 3);
      if (std::isfinite(e.cost) &&
        (!std::isfinite(best.cost) || e.cost < best.cost - hysteresis_)) best = e;
      all.push_back(std::move(e));
    }
    if (!std::isfinite(best.cost)) {stop("NO_SAFE_ROLLOUT"); return;}
    // Validate the actual first command after all ramping/hysteresis, with no later overrides.
    double x = bx, y = by, th = yaw;
    const double shield_time = std::max(0.8, std::abs(best.first.v) / acceleration_ + cycle_dt);
    for (double t = 0; t < shield_time; t += dt_) {
      if (!advance(x, y, th, best.first, dt_)) {stop("FINAL_COMMAND_COLLISION"); return;}
    }
    markers(all, best);
    publish(best.first, "MINIMUM_COST_ROLLOUT; J=" + std::to_string(best.cost) +
      "; J_semantic=" + std::to_string(best.semantic_cost));
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<TerrainMppiSegmentation>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("terrain_mppi_segmentation"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
