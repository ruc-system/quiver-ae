# ================================================================
# Shared SPDK wrapper (external dependency, all systems share)
# ================================================================
if (QUIVER_ENABLE_SPDK)
  add_subdirectory(shared/io/spdk)
endif()

function(enable_light_breakdown_if_requested target_name)
  if (QUIVER_ENABLE_LIGHT_BREAKDOWN)
    target_compile_definitions(${target_name} PUBLIC QUIVER_LIGHT_BREAKDOWN=1)
  endif()
endfunction()

# ================================================================
# FlashANNS — fully self-contained (frozen baseline)
# ================================================================
set(flashanns_io_sources
  flashanns/shared/io/mem_loader.cpp)
if (QUIVER_ENABLE_SPDK)
  list(APPEND flashanns_io_sources flashanns/shared/io/spdk/spdk_io.cpp)
endif()

add_library(flashanns_lib
  flashanns/search.cu
  flashanns/shared/common/shared_common_stub.cu
  flashanns/shared/index/pq_search.cu
  flashanns/shared/index/nav_graph.cu
  ${flashanns_io_sources}
)

target_include_directories(flashanns_lib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/flashanns)

target_link_libraries(flashanns_lib
  PUBLIC spdlog)
if (QUIVER_ENABLE_SPDK)
  target_link_libraries(flashanns_lib PUBLIC shared_spdk_wrapper)
endif()

target_compile_options(flashanns_lib
  PRIVATE "-lineinfo"
  PUBLIC "-O2" "-g"
)
enable_light_breakdown_if_requested(flashanns_lib)

add_library(flashanns_probe_lib
  flashanns/search.cu
  flashanns/shared/common/shared_common_stub.cu
  flashanns/shared/index/pq_search.cu
  flashanns/shared/index/nav_graph.cu
  ${flashanns_io_sources}
)

target_include_directories(flashanns_probe_lib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/flashanns)

target_link_libraries(flashanns_probe_lib
  PUBLIC spdlog)
if (QUIVER_ENABLE_SPDK)
  target_link_libraries(flashanns_probe_lib PUBLIC shared_spdk_wrapper)
endif()

target_compile_options(flashanns_probe_lib
  PRIVATE "-lineinfo"
  PUBLIC "-O2" "-g"
)

target_compile_definitions(flashanns_probe_lib
  PUBLIC FLASHANNS_LATENCY_PROBE)

add_library(flashanns_pw2_lib
  flashanns/search.cu
  flashanns/shared/common/shared_common_stub.cu
  flashanns/shared/index/pq_search.cu
  flashanns/shared/index/nav_graph.cu
  ${flashanns_io_sources}
)
target_include_directories(flashanns_pw2_lib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/flashanns)
target_link_libraries(flashanns_pw2_lib
  PUBLIC spdlog)
if (QUIVER_ENABLE_SPDK)
  target_link_libraries(flashanns_pw2_lib PUBLIC shared_spdk_wrapper)
endif()
target_compile_options(flashanns_pw2_lib
  PRIVATE "-lineinfo"
  PUBLIC "-O2" "-g"
)
target_compile_definitions(flashanns_pw2_lib
  PUBLIC FLASHANNS_FIXED_PIPE_WIDTH=2)

add_library(flashanns_probe_pw2_lib
  flashanns/search.cu
  flashanns/shared/common/shared_common_stub.cu
  flashanns/shared/index/pq_search.cu
  flashanns/shared/index/nav_graph.cu
  ${flashanns_io_sources}
)
target_include_directories(flashanns_probe_pw2_lib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/flashanns)
target_link_libraries(flashanns_probe_pw2_lib
  PUBLIC spdlog)
if (QUIVER_ENABLE_SPDK)
  target_link_libraries(flashanns_probe_pw2_lib PUBLIC shared_spdk_wrapper)
endif()
target_compile_options(flashanns_probe_pw2_lib
  PRIVATE "-lineinfo"
  PUBLIC "-O2" "-g"
)
target_compile_definitions(flashanns_probe_pw2_lib
  PUBLIC FLASHANNS_LATENCY_PROBE FLASHANNS_FIXED_PIPE_WIDTH=2)

function(add_fixed_flashanns_lib target_name fixed_pipe_width)
  add_library(${target_name}
    flashanns/search.cu
    flashanns/shared/common/shared_common_stub.cu
    flashanns/shared/index/pq_search.cu
    flashanns/shared/index/nav_graph.cu
    ${flashanns_io_sources}
  )
  target_include_directories(${target_name}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/flashanns)
  target_link_libraries(${target_name}
    PUBLIC spdlog)
  if (QUIVER_ENABLE_SPDK)
    target_link_libraries(${target_name} PUBLIC shared_spdk_wrapper)
  endif()
  target_compile_options(${target_name}
    PRIVATE "-lineinfo"
    PUBLIC "-O2" "-g"
  )
  target_compile_definitions(${target_name}
    PUBLIC FLASHANNS_FIXED_PIPE_WIDTH=${fixed_pipe_width})
endfunction()

add_fixed_flashanns_lib(flashanns_pw1_lib 1)
add_fixed_flashanns_lib(flashanns_pw3_lib 3)
add_fixed_flashanns_lib(flashanns_pw4_lib 4)

function(add_fixed_flashanns_probe_lib target_name fixed_pipe_width)
  add_library(${target_name}
    flashanns/search.cu
    flashanns/shared/common/shared_common_stub.cu
    flashanns/shared/index/pq_search.cu
    flashanns/shared/index/nav_graph.cu
    ${flashanns_io_sources}
  )
  target_include_directories(${target_name}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/flashanns)
  target_link_libraries(${target_name}
    PUBLIC spdlog)
  if (QUIVER_ENABLE_SPDK)
    target_link_libraries(${target_name} PUBLIC shared_spdk_wrapper)
  endif()
  target_compile_options(${target_name}
    PRIVATE "-lineinfo"
    PUBLIC "-O2" "-g"
  )
  target_compile_definitions(${target_name}
    PUBLIC FLASHANNS_LATENCY_PROBE FLASHANNS_FIXED_PIPE_WIDTH=${fixed_pipe_width})
endfunction()

add_fixed_flashanns_probe_lib(flashanns_probe_pw1_lib 1)
add_fixed_flashanns_probe_lib(flashanns_probe_pw3_lib 3)
add_fixed_flashanns_probe_lib(flashanns_probe_pw4_lib 4)

# ================================================================
# GustANN — fully self-contained (frozen baseline)
# ================================================================
set(gustann_io_sources
  gustann/shared/io/mem_loader.cpp)
if (QUIVER_ENABLE_SPDK)
  list(APPEND gustann_io_sources gustann/shared/io/spdk/spdk_io.cpp)
endif()

add_library(gustann_lib
  gustann/executor.cu
  gustann/task_runner.cu
  gustann/shared/common/shared_common_stub.cu
  gustann/shared/index/pq_search.cu
  gustann/shared/index/nav_graph.cu
  ${gustann_io_sources}
)

target_include_directories(gustann_lib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/gustann)

target_link_libraries(gustann_lib
  PUBLIC spdlog)
if (QUIVER_ENABLE_SPDK)
  target_link_libraries(gustann_lib PUBLIC shared_spdk_wrapper)
endif()

target_compile_options(gustann_lib
  PRIVATE "-lineinfo"
  PUBLIC "-O2" "-g"
)
enable_light_breakdown_if_requested(gustann_lib)

add_library(gustann_pw1_lib
  gustann/executor.cu
  gustann/task_runner.cu
  gustann/shared/common/shared_common_stub.cu
  gustann/shared/index/pq_search.cu
  gustann/shared/index/nav_graph.cu
  ${gustann_io_sources}
)

target_include_directories(gustann_pw1_lib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/gustann)

target_link_libraries(gustann_pw1_lib
  PUBLIC spdlog)
if (QUIVER_ENABLE_SPDK)
  target_link_libraries(gustann_pw1_lib PUBLIC shared_spdk_wrapper)
endif()

target_compile_options(gustann_pw1_lib
  PRIVATE "-lineinfo"
  PUBLIC "-O2" "-g"
)

target_compile_definitions(gustann_pw1_lib
  PUBLIC GUSTANN_FIXED_PIPE_WIDTH=1)

add_library(gustann_probe_pw1_lib
  gustann/executor.cu
  gustann/task_runner.cu
  gustann/shared/common/shared_common_stub.cu
  gustann/shared/index/pq_search.cu
  gustann/shared/index/nav_graph.cu
  ${gustann_io_sources}
)

target_include_directories(gustann_probe_pw1_lib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/gustann)

target_link_libraries(gustann_probe_pw1_lib
  PUBLIC spdlog)
if (QUIVER_ENABLE_SPDK)
  target_link_libraries(gustann_probe_pw1_lib PUBLIC shared_spdk_wrapper)
endif()

target_compile_options(gustann_probe_pw1_lib
  PRIVATE "-lineinfo"
  PUBLIC "-O2" "-g"
)

target_compile_definitions(gustann_probe_pw1_lib
  PUBLIC GUSTANN_LATENCY_PROBE GUSTANN_FIXED_PIPE_WIDTH=1)

# ================================================================
# Quiver — uses shared/ directly (actively developed)
# ================================================================
add_library(shared_common_lib
  shared/common/shared_common_stub.cu)

target_include_directories(shared_common_lib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(shared_common_lib
  PUBLIC spdlog)

add_library(shared_index_lib
  shared/index/pq_search.cu
  shared/index/nav_graph.cu)

target_include_directories(shared_index_lib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(shared_index_lib
  PUBLIC shared_common_lib)

target_compile_options(shared_index_lib
  PRIVATE "-lineinfo"
  PUBLIC "-O2" "-g"
)

set(shared_io_sources
  shared/io/mem_loader.cpp)
if (QUIVER_ENABLE_SPDK)
  list(APPEND shared_io_sources shared/io/spdk/spdk_io.cpp)
endif()

add_library(shared_io_lib
  ${shared_io_sources})

target_include_directories(shared_io_lib
  PUBLIC "${PROJECT_SOURCE_DIR}/src")

target_link_libraries(shared_io_lib
  PUBLIC shared_common_lib)
if (QUIVER_ENABLE_SPDK)
  target_link_libraries(shared_io_lib PUBLIC shared_spdk_wrapper)
endif()

target_compile_options(shared_io_lib
  PUBLIC "-O2" "-g")

add_executable(spdk_lba_offset_test
  shared/io/spdk_lba_offset_test.cpp)

target_include_directories(spdk_lba_offset_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(spdk_lba_offset_test
  PRIVATE shared_io_lib)

target_compile_options(spdk_lba_offset_test
  PRIVATE "-O2" "-g")

add_executable(shared_mem_loader_test
  shared/io/mem_loader_test.cpp
  shared/io/mem_loader.cpp)

target_include_directories(shared_mem_loader_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(shared_mem_loader_test
  PRIVATE spdlog)

target_compile_options(shared_mem_loader_test
  PRIVATE "-O2" "-g")

add_executable(gustann_mem_loader_test
  gustann/shared/io/mem_loader_test.cpp
  gustann/shared/io/mem_loader.cpp)
target_include_directories(gustann_mem_loader_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gustann)
target_link_libraries(gustann_mem_loader_test
  PRIVATE spdlog)
target_compile_options(gustann_mem_loader_test
  PRIVATE "-O2" "-g")

add_executable(flashanns_mem_loader_test
  flashanns/shared/io/mem_loader_test.cpp
  flashanns/shared/io/mem_loader.cpp)
target_include_directories(flashanns_mem_loader_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/flashanns)
target_link_libraries(flashanns_mem_loader_test
  PRIVATE spdlog)
target_compile_options(flashanns_mem_loader_test
  PRIVATE "-O2" "-g")

add_executable(flashanns_probe_report_test
  flashanns/probe_report_test.cpp)
set_source_files_properties(flashanns/probe_report_test.cpp PROPERTIES
  LANGUAGE CUDA)
target_include_directories(flashanns_probe_report_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/flashanns)
target_compile_options(flashanns_probe_report_test
  PRIVATE "-O2" "-g")

add_executable(quiver_probe_report_test
  quiver/probe_report_test.cpp)
set_source_files_properties(quiver/probe_report_test.cpp PROPERTIES
  LANGUAGE CUDA)
target_include_directories(quiver_probe_report_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/quiver)
target_compile_options(quiver_probe_report_test
  PRIVATE "-O2" "-g")

set(quiver_lib_sources quiver/search.cu)

# Strawman host launchers (compile-time gated).
# Only built when -DQUIVER_BUILD_STRAWMEN=ON; production builds stay lean.
if (QUIVER_BUILD_STRAWMEN)
  list(APPEND quiver_lib_sources quiver/search_strawmen.cu)
endif()

add_library(quiver_lib ${quiver_lib_sources})

target_include_directories(quiver_lib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(quiver_lib
  PUBLIC shared_common_lib shared_index_lib shared_io_lib)

target_compile_options(quiver_lib
  PRIVATE "-lineinfo"
  PUBLIC "-O2" "-g"
)
enable_light_breakdown_if_requested(quiver_lib)

add_library(quiver_probe_lib ${quiver_lib_sources})

target_include_directories(quiver_probe_lib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(quiver_probe_lib
  PUBLIC shared_common_lib shared_index_lib shared_io_lib)

target_compile_options(quiver_probe_lib
  PRIVATE "-lineinfo"
  PUBLIC "-O2" "-g"
)

target_compile_definitions(quiver_probe_lib
  PUBLIC QUIVER_LATENCY_PROBE)

if (QUIVER_BUILD_STRAWMEN)
  target_compile_definitions(quiver_lib PUBLIC QUIVER_BUILD_STRAWMEN)
  target_compile_definitions(quiver_probe_lib PUBLIC QUIVER_BUILD_STRAWMEN)
endif()

function(add_fixed_quiver_lib target_name fixed_pipe_width fixed_queries_per_block)
  add_library(${target_name} ${quiver_lib_sources})
  target_include_directories(${target_name}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${target_name}
    PUBLIC shared_common_lib shared_index_lib shared_io_lib)
  target_compile_options(${target_name}
    PRIVATE "-lineinfo"
    PUBLIC "-O2" "-g"
  )
  target_compile_definitions(${target_name}
    PUBLIC
      QUIVER_FIXED_PIPE_WIDTH=${fixed_pipe_width}
      QUIVER_FIXED_QUERIES_PER_BLOCK=${fixed_queries_per_block})
  if (QUIVER_BUILD_STRAWMEN)
    target_compile_definitions(${target_name} PUBLIC QUIVER_BUILD_STRAWMEN)
  endif()
endfunction()

add_fixed_quiver_lib(quiver_lib_pw1_q1 1 1)
add_fixed_quiver_lib(quiver_lib_pw1_q2 1 2)
add_fixed_quiver_lib(quiver_lib_pw1_q4 1 4)
add_fixed_quiver_lib(quiver_lib_pw1_q8 1 8)
add_fixed_quiver_lib(quiver_lib_pw2_q1 2 1)
add_fixed_quiver_lib(quiver_lib_pw2_q2 2 2)
add_fixed_quiver_lib(quiver_lib_pw2_q3 2 3)
add_fixed_quiver_lib(quiver_lib_pw2_q4 2 4)
add_fixed_quiver_lib(quiver_lib_pw4_q1 4 1)

function(add_fixed_quiver_probe_lib target_name fixed_pipe_width fixed_queries_per_block)
  add_library(${target_name} ${quiver_lib_sources})
  target_include_directories(${target_name}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${target_name}
    PUBLIC shared_common_lib shared_index_lib shared_io_lib)
  target_compile_options(${target_name}
    PRIVATE "-lineinfo"
    PUBLIC "-O2" "-g"
  )
  target_compile_definitions(${target_name}
    PUBLIC
      QUIVER_LATENCY_PROBE
      QUIVER_FIXED_PIPE_WIDTH=${fixed_pipe_width}
      QUIVER_FIXED_QUERIES_PER_BLOCK=${fixed_queries_per_block})
  if (QUIVER_BUILD_STRAWMEN)
    target_compile_definitions(${target_name} PUBLIC QUIVER_BUILD_STRAWMEN)
  endif()
endfunction()

add_fixed_quiver_probe_lib(quiver_probe_lib_pw1_q1 1 1)
add_fixed_quiver_probe_lib(quiver_probe_lib_pw1_q2 1 2)
add_fixed_quiver_probe_lib(quiver_probe_lib_pw1_q4 1 4)
add_fixed_quiver_probe_lib(quiver_probe_lib_pw1_q8 1 8)
add_fixed_quiver_probe_lib(quiver_probe_lib_pw2_q1 2 1)
add_fixed_quiver_probe_lib(quiver_probe_lib_pw2_q2 2 2)
add_fixed_quiver_probe_lib(quiver_probe_lib_pw2_q3 2 3)
add_fixed_quiver_probe_lib(quiver_probe_lib_pw2_q4 2 4)
add_fixed_quiver_probe_lib(quiver_probe_lib_pw4_q1 4 1)
