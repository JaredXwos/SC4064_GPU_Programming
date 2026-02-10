cmake_minimum_required(VERSION 4.1)
project(SC4046TestBed CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 80)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(utils STATIC ${SRC_DIR}/utils.cu)
set_target_properties(utils PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(SC4046Tut1Q1 ${SRC_DIR}/vector_addition.cu)
add_executable(SC4046Tut1Q2 ${SRC_DIR}/matrix_addition.cu)
add_executable(SC4046Tut1Q3 ${SRC_DIR}/matrix_multiplication.cu)

target_link_libraries(SC4046Tut1Q1 PRIVATE utils)
target_link_libraries(SC4046Tut1Q2 PRIVATE utils)
target_link_libraries(SC4046Tut1Q3 PRIVATE utils)

set(CUDA_VERBOSE_FLAGS -lineinfo -Xptxas=-v,-warn-spills)

target_compile_options(utils PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_VERBOSE_FLAGS}>
)

target_compile_options(SC4046Tut1Q1 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_VERBOSE_FLAGS}>
)

target_compile_options(SC4046Tut1Q2 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_VERBOSE_FLAGS}>
)

target_compile_options(SC4046Tut1Q3 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_VERBOSE_FLAGS}>
)