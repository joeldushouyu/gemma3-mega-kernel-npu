cmake_minimum_required(VERSION 3.15)
project(MegaKernelNPU)

# Optionally, set C++ standard
set(CMAKE_CXX_STANDARD 23)

add_subdirectory(SubModules/safetensors-cpp)
add_subdirectory(SubModules/tokenizers-cpp)
add_subdirectory(SubModules/ggml)
add_subdirectory(SubModules/json)
add_subdirectory(cpp_code)