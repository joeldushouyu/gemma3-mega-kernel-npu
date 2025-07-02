cmake_minimum_required(VERSION 3.15)
project(MegaKernelNPU)

# Optionally, set C++ standard
set(CMAKE_CXX_STANDARD 23)

add_subdirectory(safetensors-cpp)
add_subdirectory(tokenizers-cpp)
add_subdirectory(ggml)
add_subdirectory(json)
add_subdirectory(cpp_code)