#!/bin/bash
source ../venv/bin/activate
mkdir -p build
cd build
cmake  -DGGML_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . --config Debug -j 8
if [ $? -ne 0 ]; then
    echo "CMake configuration or build failed."
    exit 1
fi