

find_package(nlohmann_json 3.11.0 REQUIRED)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(Gemma3Library gemma3_common)
add_library(${Gemma3Library} STATIC gemma3_common.cpp)
target_include_directories(${Gemma3Library} PUBLIC .)
# print GGMl/include
message( "CMAKE_SOURCE_DIR" ${CMAKE_SOURCE_DIR} )
message( "CURRENT_SOURCE_DIR" ${CMAKE_CURRENT_SOURCE_DIR} )
target_include_directories(${Gemma3Library} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../SubModules/ggml/src)
target_link_libraries(${Gemma3Library} PRIVATE ggml common  nlohmann_json::nlohmann_json safetensors_cpp ) 


set(Gemma3_GGML_TO_SAFETENSOR gemma3_ggml_to_tensor)
add_executable(${Gemma3_GGML_TO_SAFETENSOR} "test/gemma3_ggml_to_tensor.cpp")
target_link_libraries(${Gemma3_GGML_TO_SAFETENSOR} PRIVATE ggml common ${Gemma3Library}  nlohmann_json::nlohmann_json safetensors_cpp )



set(TEST_TARGET random_ggml_to_tensor)
add_executable(${TEST_TARGET} "test/random_ggml_to_tensor.cpp")
target_link_libraries(${TEST_TARGET} PRIVATE ggml common ${Gemma3Library}  nlohmann_json::nlohmann_json safetensors_cpp )

