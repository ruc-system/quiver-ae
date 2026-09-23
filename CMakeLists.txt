cmake_minimum_required(VERSION 3.18)

project(Quiver LANGUAGES CXX)

enable_language(CUDA)
enable_language(C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Strawman ablation binaries: strawman design paths used to defend §Key Idea.
# Default ON so experiment builds include the full strawman set. Flip OFF for
# lean production-style builds.
option(QUIVER_BUILD_STRAWMEN "Build strawman ablation binaries" ON)
option(QUIVER_ENABLE_SPDK "Build SPDK-backed IO support" ON)
option(QUIVER_ENABLE_LIGHT_BREAKDOWN
       "Enable lightweight AE breakdown instrumentation" OFF)

if (QUIVER_ENABLE_SPDK)
  add_compile_definitions(QUIVER_ENABLE_SPDK=1 USE_SPDK)
else()
  add_compile_definitions(QUIVER_ENABLE_SPDK=0)
endif()

find_package(OpenMP)

add_subdirectory(deps/spdlog)
add_subdirectory(deps/argparse)
add_subdirectory(deps/DiskANN)
if (TARGET build_memory_index)
  set_target_properties(build_memory_index PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

add_subdirectory(src)
add_subdirectory(bin)
