add_library(shared_io_lib
  mem_loader.cpp)

target_include_directories(shared_io_lib
  PUBLIC "${PROJECT_SOURCE_DIR}/src")

target_link_libraries(shared_io_lib
  PUBLIC shared_common_lib)

target_compile_options(shared_io_lib
  PUBLIC "-O2" "-g")

add_executable(spdk_lba_offset_test
  spdk_lba_offset_test.cpp)

target_include_directories(spdk_lba_offset_test
  PRIVATE "${PROJECT_SOURCE_DIR}/src")

target_link_libraries(spdk_lba_offset_test
  PRIVATE shared_io_lib)

target_compile_options(spdk_lba_offset_test
  PRIVATE "-O2" "-g")

add_executable(spdk_write_buffer_test
  spdk_write_buffer_test.cpp)

target_include_directories(spdk_write_buffer_test
  PRIVATE "${PROJECT_SOURCE_DIR}/src")

target_link_libraries(spdk_write_buffer_test
  PRIVATE shared_io_lib)

target_compile_options(spdk_write_buffer_test
  PRIVATE "-O2" "-g")

if (QUIVER_ENABLE_SPDK)
  add_subdirectory(spdk)
  target_link_libraries(shared_io_lib
    PUBLIC shared_spdk_io)
  target_compile_definitions(shared_io_lib
    PUBLIC "USE_SPDK")
endif()
