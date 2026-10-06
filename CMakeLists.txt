cmake_minimum_required(VERSION 4.2)
project(SieveOfEratosthenes CUDA)

set(CMAKE_CUDA_STANDARD 23)

file(GLOB_RECURSE SOURCE_FILES src/*.cu)

file(GLOB_RECURSE HEADER_FILES src/*.cuh)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${HEADER_FILES})

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

set_target_properties(SieveOfEratosthenes PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
