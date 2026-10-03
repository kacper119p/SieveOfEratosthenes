cmake_minimum_required(VERSION 4.2)
project(SieveOfEratosthenes CUDA)

set(CMAKE_CUDA_STANDARD 23)

add_executable(SieveOfEratosthenes main.cu)

set_target_properties(SieveOfEratosthenes PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
