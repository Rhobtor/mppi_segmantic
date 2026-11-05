#include "terrain_mppi_segmentation/terrain_semantic_core.hpp"
#include <gtest/gtest.h>
#include <unordered_map>

namespace ts = terrain_semantic;

TEST(TerrainSemantic, FootprintInteriorAndOutsideVeto)
{
  ts::Grid g;
  g.info.resolution = .1;
  g.info.width = g.info.height = 50;
  std::vector<uint8_t> blocked(g.size(), 0);
  blocked[25 * 50 + 25] = 1;
  auto inflated = ts::inflate(g, blocked, 1, true);
  EXPECT_TRUE(inflated[25 * 50 + 30]);
  EXPECT_TRUE(inflated[0]);
  EXPECT_FALSE(inflated[12 * 50 + 25]);
  EXPECT_EQ(g.index(-.001, 0), -1);
}

TEST(TerrainSemantic, FlatMapAndMetricHazards)
{
  grid_map::GridMap map({"elevation", "slope", "step", "rough"});
  map.setGeometry(grid_map::Length(8, 8), .1);
  map.setFrameId("map"); map.setTimestamp(1000000000);
  for (const auto & layer : map.getLayers()) map[layer].setZero();
  ts::TerrainLimits limits;
  limits.radius = .2;
  auto terrain = ts::makeTerrain(map, limits);
  int centre = terrain.grid.index(0, 0);
  ASSERT_GE(centre, 0);
  EXPECT_FALSE(terrain.blocked[centre]);
  map.atPosition("step", grid_map::Position(.3, 0)) = .2;
  terrain = ts::makeTerrain(map, limits);
  EXPECT_TRUE(terrain.blocked[terrain.grid.index(.3, 0)]);
  EXPECT_FALSE(terrain.blocked[terrain.grid.index(-2, -2)]);
  map["step"].setZero();
  map.atPosition("slope", grid_map::Position(.3, 0)) = .6;
  terrain = ts::makeTerrain(map, limits);
  EXPECT_TRUE(terrain.blocked[terrain.grid.index(.3, 0)]);
}

TEST(TerrainSemantic, SkidSteerAndFreshness)
{
  ts::Motion m;
  double x = 0, y = 0, yaw = 0;
  m.integrate(x, y, yaw, 0, .7, .1);
  EXPECT_DOUBLE_EQ(x, 0);
  EXPECT_DOUBLE_EQ(y, 0);
  EXPECT_NEAR(yaw, .7 * .55 * .1, 1e-10);
  EXPECT_TRUE(ts::fresh(10, 9.5, .8));
  EXPECT_FALSE(ts::fresh(10, 9, .8));
  EXPECT_FALSE(ts::fresh(10, 11, .8));
  EXPECT_FALSE(ts::fresh(10, 0, .8));
}

TEST(TerrainSemantic, PreferencesNeverGrantCollisionPermission)
{
  ts::Grid g;
  g.info.resolution = .1;
  g.info.width = g.info.height = 40;
  std::vector<int8_t> costs(g.size(), -1);
  costs[g.index(2, 2)] = 60;
  costs[g.index(2, 1)] = 0;
  EXPECT_DOUBLE_EQ(ts::preferenceCost(g, costs, 2, 2), .6);
  EXPECT_DOUBLE_EQ(ts::preferenceCost(g, costs, 2, 1), 0);
  EXPECT_DOUBLE_EQ(ts::preferenceCost(g, costs, 0, 0), 0);
  EXPECT_DOUBLE_EQ(ts::preferenceCost(g, costs, -1, 0), 0);
  std::vector<uint8_t> blocked(g.size(), 0);
  blocked[g.index(2, 1)] = 1;
  auto inflated = ts::inflate(g, blocked, .2, true);
  EXPECT_TRUE(inflated[g.index(2, 1)]);
}

TEST(TerrainSemantic, SweptCollisionCatchesIntermediateCellsAndReverse)
{
  ts::Motion motion;
  motion.tracked = false;
  const auto safe = [](double x, double, double) {return std::abs(x - .5) > .049;};
  double x = 0, y = 0, yaw = 0;
  EXPECT_FALSE(ts::advanceSafe(motion, .1, x, y, yaw, 2, 0, .5, safe));
  x = 1; y = 0; yaw = 0;
  EXPECT_FALSE(ts::advanceSafe(motion, .1, x, y, yaw, -2, 0, .5, safe));
  x = 0; y = 0; yaw = 0;
  const auto spin_safe = [](double, double, double angle) {return std::abs(angle) < .5;};
  EXPECT_FALSE(ts::advanceSafe(motion, .1, x, y, yaw, 0, 1, 1, spin_safe));
}

TEST(TerrainSemantic, RollingGridUsesWorldOriginAndCellCentres)
{
  grid_map::GridMap map({"elevation"});
  map.setGeometry(grid_map::Length(21, 21), .1, grid_map::Position(12, -3));
  map.setFrameId("map"); map["elevation"].setZero();
  auto g = ts::layout(map);
  EXPECT_NEAR(g.info.origin.position.x, 1.5, 1e-5);
  EXPECT_NEAR(g.info.origin.position.y, -13.5, 1e-5);
  const size_t i = 105 * g.info.width + 105;
  const auto p = g.position(i);
  EXPECT_EQ(g.index(p.first, p.second), static_cast<int>(i));
  map.move(grid_map::Position(13, -2));
  g = ts::layout(map);
  EXPECT_NEAR(g.info.origin.position.x, 2.5, 1e-5);
}

TEST(TerrainSemantic, UnknownCellsArePenalisedButOnlyBlockedOnRequest)
{
  grid_map::GridMap map({"elevation"});
  map.setGeometry(grid_map::Length(4, 4), .1);
  map.setFrameId("map");
  map["elevation"].setZero();
  ts::TerrainLimits limits;
  limits.radius = .1;
  auto t = ts::makeTerrain(map, limits);
  EXPECT_FALSE(t.blocked[t.grid.index(0, 0)]);
  map.atPosition("elevation", grid_map::Position(0, 0)) = NAN;
  t = ts::makeTerrain(map, limits);
  const int i = t.grid.index(0, 0);
  EXPECT_FALSE(t.blocked[i]);  // same as original Teacher: cost only
  EXPECT_FLOAT_EQ(t.risk[i], 1);
  EXPECT_FLOAT_EQ(t.confidence[i], 0);
  limits.unknown_blocked = true;
  t = ts::makeTerrain(map, limits);
  EXPECT_TRUE(t.blocked[t.grid.index(0, 0)]);
}

TEST(TerrainSemantic, StopOnlyForReportedOrSilentHazards)
{
  // Disabled: never.
  EXPECT_FALSE(ts::semanticStopRequired(false, true, true, true, 10, 0, .8));
  // Segmentation never started: geometric navigation continues.
  EXPECT_FALSE(ts::semanticStopRequired(true, false, false, false, 10, 0, .8));
  // Producer requests stop.
  EXPECT_TRUE(ts::semanticStopRequired(true, true, true, false, 10, 9.9, .8));
  // Obstacles reported and then the producer went silent.
  EXPECT_TRUE(ts::semanticStopRequired(true, true, false, true, 10, 9, .8));
  // Silent but last grid had no obstacles: preferences off, keep navigating.
  EXPECT_FALSE(ts::semanticStopRequired(true, true, false, false, 10, 9, .8));
  EXPECT_FALSE(ts::semanticStopRequired(true, true, false, true, 10, 9.9, .8));
}

TEST(TerrainSemantic, UnityLatticeMapsToCroppedMask)
{
  ts::CloudLattice l;  // Unity defaults: 1920x1080, step 3, roi 0.35
  EXPECT_EQ(l.startRow(), 378);
  EXPECT_EQ(l.expectedWidth(), 640);
  EXPECT_EQ(l.expectedHeight(), 234);
  unsigned u, v;
  // Cropped mask keeps the top 63%: 1920x680.
  ASSERT_TRUE(l.toMask(0, 0, 1920, 680, u, v));
  EXPECT_EQ(u, 0u); EXPECT_EQ(v, 378u);
  ASSERT_TRUE(l.toMask(639, 100, 1920, 680, u, v));
  EXPECT_EQ(u, 1917u); EXPECT_EQ(v, 678u);
  EXPECT_FALSE(l.toMask(0, 101, 1920, 680, u, v));  // hood rows removed by the crop
  // Same image downscaled to 640 wide.
  ASSERT_TRUE(l.toMask(300, 50, 640, 227, u, v));
  EXPECT_EQ(u, 300u); EXPECT_EQ(v, 176u);
  int ix, jy;
  EXPECT_FALSE(l.fromMask(10, 100, 1920, 640, 234, ix, jy));  // above the raycast ROI
  ASSERT_TRUE(l.fromMask(1917, 678, 1920, 640, 234, ix, jy));
  EXPECT_EQ(ix, 639); EXPECT_EQ(jy, 100);
  // ZED registered cloud: one point per pixel, no ROI.
  ts::CloudLattice zed{1280, 720, 1, 0.0};
  ASSERT_TRUE(zed.toMask(640, 300, 1280, 453, u, v));
  EXPECT_EQ(u, 640u); EXPECT_EQ(v, 300u);
}

TEST(TerrainSemantic, ObstaclesRetainedUntilObservedFree)
{
  std::unordered_map<int64_t, ts::ObstacleCell> cells;
  std::unordered_map<int64_t, ts::CellObservation> obs;
  obs[7] = {5, 0, 1};
  ts::updateObstacleCells(cells, obs, 1.0, 3, 3, 0);
  ASSERT_TRUE(cells.count(7));
  obs.clear();
  ts::updateObstacleCells(cells, obs, 50.0, 3, 3, 0);  // unobserved: retained
  EXPECT_TRUE(cells.count(7));
  obs[7] = {1, 10, 1};  // mixed evidence: neither marks nor clears
  ts::updateObstacleCells(cells, obs, 51.0, 3, 3, 0);
  EXPECT_TRUE(cells.count(7));
  obs[7] = {0, 2, 255};  // too little free evidence
  ts::updateObstacleCells(cells, obs, 52.0, 3, 3, 0);
  EXPECT_TRUE(cells.count(7));
  obs[7] = {0, 3, 255};
  ts::updateObstacleCells(cells, obs, 53.0, 3, 3, 0);
  EXPECT_FALSE(cells.count(7));
  obs.clear();
  obs[8] = {3, 0, 11};
  ts::updateObstacleCells(cells, obs, 60.0, 3, 3, 5.0);
  obs.clear();
  ts::updateObstacleCells(cells, obs, 66.0, 3, 3, 5.0);  // optional max age
  EXPECT_FALSE(cells.count(8));
}

TEST(TerrainSemantic, PaletteMatchesOverlay)
{
  EXPECT_EQ(ts::classColor(0), (std::array<uint8_t, 3>{0, 0, 0}));
  EXPECT_EQ(ts::classColor(1), (std::array<uint8_t, 3>{128, 0, 0}));
  EXPECT_EQ(ts::classColor(2), (std::array<uint8_t, 3>{0, 128, 0}));
  EXPECT_EQ(ts::classColor(4), (std::array<uint8_t, 3>{0, 0, 128}));
  EXPECT_EQ(ts::classColor(8), (std::array<uint8_t, 3>{64, 0, 0}));
  EXPECT_EQ(ts::classColor(11), (std::array<uint8_t, 3>{192, 128, 0}));
}
