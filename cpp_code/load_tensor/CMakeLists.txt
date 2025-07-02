
find_package(nlohmann_json 3.11.0 REQUIRED)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)




set(TEST_TARGET load_tensor_main)
add_executable(${TEST_TARGET} load_tensor_main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common gemma3_common  nlohmann_json::nlohmann_json safetensors_cpp )

