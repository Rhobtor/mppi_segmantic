cmake_minimum_required(VERSION 3.8)
project(terrain_mppi_segmentation)

find_package(ament_cmake REQUIRED)
set(DEPS rclcpp std_msgs geometry_msgs nav_msgs sensor_msgs visualization_msgs
  grid_map_msgs grid_map_ros grid_map_core tf2 tf2_ros tf2_geometry_msgs)
foreach(dep ${DEPS})
  find_package(${dep} REQUIRED)
endforeach()

foreach(target semantic_cost_node terrain_mppi_segmentation)
  add_executable(${target} src/${target}.cpp)
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  target_include_directories(${target} PRIVATE include)
  ament_target_dependencies(${target} ${DEPS})
  install(TARGETS ${target} DESTINATION lib/${PROJECT_NAME})
endforeach()

install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_terrain_semantic_core test/test_terrain_semantic_core.cpp)
  target_compile_features(test_terrain_semantic_core PRIVATE cxx_std_17)
  target_include_directories(test_terrain_semantic_core PRIVATE include)
  ament_target_dependencies(test_terrain_semantic_core nav_msgs sensor_msgs grid_map_core)
endif()

ament_package()
