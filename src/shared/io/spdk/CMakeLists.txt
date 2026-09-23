find_package(PkgConfig REQUIRED)
include(ProcessorCount)

get_filename_component(SPDK_PATH "${PROJECT_SOURCE_DIR}/deps/spdk" ABSOLUTE)

function(quiver_spdk_assets_complete spdk_path result_var)
  if (EXISTS "${spdk_path}/build/lib/pkgconfig/spdk_nvme.pc"
      AND EXISTS "${spdk_path}/build/include/spdk/nvme.h"
      AND EXISTS "${spdk_path}/dpdk/build/lib/librte_eal.a"
      AND EXISTS "${spdk_path}/isa-l/.libs/libisal.a"
      AND EXISTS "${spdk_path}/isa-l-crypto/.libs/libisal_crypto.a")
    set(${result_var} TRUE PARENT_SCOPE)
  else()
    set(${result_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(quiver_spdk_pkgconfig_paths_valid pkgconfig_output result_var)
  set(valid TRUE)
  string(REGEX MATCHALL "-L[^ ;]+" lib_dirs "${pkgconfig_output}")
  foreach(dir_token IN LISTS lib_dirs)
    string(REGEX REPLACE "^-L" "" dir_path "${dir_token}")
    if (NOT EXISTS "${dir_path}")
      message(STATUS "SPDK pkg-config references missing path: ${dir_path}")
      set(valid FALSE)
    endif()
  endforeach()
  set(${result_var} ${valid} PARENT_SCOPE)
endfunction()

function(quiver_query_spdk_pkgconfig spdk_path libs_var syslibs_var)
  set(saved_pkg_config_path "$ENV{PKG_CONFIG_PATH}")
  if (EXISTS "${spdk_path}/build/lib/pkgconfig")
    set(ENV{PKG_CONFIG_PATH} "${spdk_path}/build/lib/pkgconfig")
  else()
    set(ENV{PKG_CONFIG_PATH} "")
  endif()

  execute_process(
    COMMAND pkg-config --libs spdk_nvme spdk_env_dpdk
    OUTPUT_VARIABLE libs
    RESULT_VARIABLE libs_result
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE libs_error)
  execute_process(
    COMMAND pkg-config --libs --static spdk_syslibs
    OUTPUT_VARIABLE syslibs
    RESULT_VARIABLE syslibs_result
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE syslibs_error)

  set(ENV{PKG_CONFIG_PATH} "${saved_pkg_config_path}")

  if (NOT libs_result EQUAL 0 OR NOT syslibs_result EQUAL 0)
    message(STATUS "pkg-config query failed: ${libs_error}${syslibs_error}")
    set(${libs_var} "" PARENT_SCOPE)
    set(${syslibs_var} "" PARENT_SCOPE)
    return()
  endif()

  set(${libs_var} "${libs}" PARENT_SCOPE)
  set(${syslibs_var} "${syslibs}" PARENT_SCOPE)
endfunction()

function(quiver_build_spdk_from_source spdk_path)
  if (NOT EXISTS "${spdk_path}/configure")
    message(FATAL_ERROR "SPDK source not found at ${spdk_path}")
  endif()

  ProcessorCount(build_jobs)
  if (NOT build_jobs)
    set(build_jobs 4)
  endif()

  message(STATUS "Building SPDK from source in ${spdk_path} (-j${build_jobs})")

  execute_process(
    COMMAND "${spdk_path}/configure"
      --disable-tests
      --disable-unit-tests
      --disable-examples
      --disable-apps
      --without-crypto
      --without-fsdev
      --without-idxd
      --without-iscsi-initiator
      --without-nvme-cuse
      --without-ocf
      --without-rbd
      --without-shared
      --without-uring
      --without-usdt
      --without-vhost
      --without-vtune
      --without-virtio
      --without-xnvme
    WORKING_DIRECTORY "${spdk_path}"
    RESULT_VARIABLE configure_result
    OUTPUT_VARIABLE configure_output
    ERROR_VARIABLE configure_error)
  if (NOT configure_result EQUAL 0)
    message(FATAL_ERROR
      "SPDK configure failed (exit ${configure_result}):\n"
      "${configure_output}\n${configure_error}")
  endif()

  execute_process(
    COMMAND make -j${build_jobs}
    WORKING_DIRECTORY "${spdk_path}"
    RESULT_VARIABLE make_result
    OUTPUT_VARIABLE make_output
    ERROR_VARIABLE make_error)
  if (NOT make_result EQUAL 0)
    message(FATAL_ERROR
      "SPDK build failed (exit ${make_result}):\n"
      "${make_output}\n${make_error}")
  endif()
endfunction()

function(quiver_ensure_spdk_built spdk_path)
  quiver_spdk_assets_complete("${spdk_path}" assets_complete)
  set(need_build FALSE)

  if (NOT assets_complete)
    set(need_build TRUE)
    quiver_query_spdk_pkgconfig("${spdk_path}" queried_libs queried_syslibs)
    if (queried_libs STREQUAL "" OR queried_syslibs STREQUAL "")
      message(STATUS "Local SPDK build missing and no usable pkg-config found; building from source")
    else()
      quiver_spdk_pkgconfig_paths_valid("${queried_libs} ${queried_syslibs}" paths_valid)
      if (paths_valid)
        message(STATUS "Local SPDK build missing; building from source instead of external install")
      else()
        message(STATUS "Local SPDK build missing and external pkg-config paths are invalid; building from source")
      endif()
    endif()
  else()
    quiver_query_spdk_pkgconfig("${spdk_path}" queried_libs queried_syslibs)
    quiver_spdk_pkgconfig_paths_valid("${queried_libs} ${queried_syslibs}" paths_valid)
    if (NOT paths_valid)
      set(need_build TRUE)
      message(STATUS "Local SPDK pkg-config references invalid paths; rebuilding from source")
    endif()
  endif()

  if (need_build)
    quiver_build_spdk_from_source("${spdk_path}")
    quiver_spdk_assets_complete("${spdk_path}" assets_complete)
    if (NOT assets_complete)
      message(FATAL_ERROR "SPDK build finished but expected artifacts are still missing")
    endif()
  endif()
endfunction()

quiver_ensure_spdk_built("${SPDK_PATH}")

set(ENV{PKG_CONFIG_PATH} "${SPDK_PATH}/build/lib/pkgconfig")

execute_process(
  COMMAND pkg-config --libs spdk_nvme spdk_env_dpdk
  OUTPUT_VARIABLE SPDK_LIBRARY
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE SPDK_LIBS_RESULT
  ERROR_VARIABLE SPDK_LIBS_ERROR)
execute_process(
  COMMAND pkg-config --libs --static spdk_syslibs
  OUTPUT_VARIABLE SPDK_SYSLIBS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE SPDK_SYSLIBS_RESULT
  ERROR_VARIABLE SPDK_SYSLIBS_ERROR)

if (NOT SPDK_LIBS_RESULT EQUAL 0 OR NOT SPDK_SYSLIBS_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Failed to query local SPDK pkg-config:\n"
    "${SPDK_LIBS_ERROR}\n${SPDK_SYSLIBS_ERROR}")
endif()

set(SPDK_PKGCONFIG_OUTPUT "${SPDK_LIBRARY} ${SPDK_SYSLIBS}")
quiver_spdk_pkgconfig_paths_valid("${SPDK_PKGCONFIG_OUTPUT}" SPDK_PATHS_VALID)
if (NOT SPDK_PATHS_VALID)
  message(FATAL_ERROR
    "Local SPDK pkg-config still references missing library paths:\n"
    "${SPDK_PKGCONFIG_OUTPUT}")
endif()

set(SPDK_ASSET_ROOT "${SPDK_PATH}")

separate_arguments(SPDK_LIBRARY UNIX_COMMAND "${SPDK_LIBRARY}")
separate_arguments(SPDK_SYSLIBS UNIX_COMMAND "${SPDK_SYSLIBS}")

message(STATUS "SPDK root: ${SPDK_ASSET_ROOT}")
message(STATUS "SPDK libraries: ${SPDK_LIBRARY}")
message(STATUS "SPDK system libraries: ${SPDK_SYSLIBS}")

add_library(shared_spdk_wrapper spdk_wrapper.cc)
target_include_directories(shared_spdk_wrapper
  PUBLIC "${SPDK_ASSET_ROOT}/include"
         "${SPDK_ASSET_ROOT}/build/include")
target_compile_options(shared_spdk_wrapper PUBLIC "-O2")
target_link_libraries(shared_spdk_wrapper PRIVATE spdlog)
target_link_options(shared_spdk_wrapper INTERFACE
  -pthread
  "SHELL:-Wl,--whole-archive"
  "SHELL:-Wl,-Bstatic"
  ${SPDK_LIBRARY}
  "SHELL:-Wl,-Bdynamic"
  "SHELL:-Wl,--no-whole-archive"
  ${SPDK_SYSLIBS})

add_library(shared_spdk_io spdk_io.cpp)
target_link_libraries(shared_spdk_io PUBLIC shared_spdk_wrapper)
target_compile_options(shared_spdk_io PUBLIC "-O2")
