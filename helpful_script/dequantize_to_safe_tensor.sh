#!/bin/bash

set -e  # Exit on any error

# Get current and root directories
script_dir=$(dirname "$(readlink -f "$0")")
root_dir="$script_dir/.."
UNQUOTED_CACHE_DIR="$root_dir/python_code/gemma3-evaluation/hf_cache"

# Function to prepare and convert a model
prepare_and_convert() {
    local model_name="$1"
    local source_dir="$UNQUOTED_CACHE_DIR/models--google--gemma-3-4b-it"
    local snapshot_dir="$source_dir/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"

    local target_dir_snapshot="$source_dir/snapshots/$model_name" 
    local gguf_model_path="$2"
    local mmproj_path="$3"

    echo "Processing $model_name..."

    # Clean old directory if it exists
    rm -rf "$target_dir_snapshot"

    # Copy base model
    cp -r "$snapshot_dir" "$target_dir_snapshot"

    # Remove specific files from snapshot
    rm -f "$target_dir_snapshot"/model-00001-of-00002.safetensors
    rm -f "$target_dir_snapshot"/model-00002-of-00002.safetensors
    rm -f "$target_dir_snapshot"/model.safetensors.index.json

    # Run conversion
    "$root_dir/build/cpp_code/ggml_to_tensor/gemma3_ggml_to_tensor" \
        -m "$gguf_model_path" \
        "$mmproj_path" \
        "$target_dir_snapshot" \
        "YES"
}

# # === Run conversions ===

# # 1. Google Q4_0 version
# prepare_and_convert \
#     "dequant-q40" \
#     "$root_dir/gemma3-quantize/google_4bit_gguf/gemma-3-4b-it-q4_0.gguf" \
#     "$root_dir/gemma3-quantize/google_4bit_gguf/mmproj-model-f16-4B.gguf"

# 2. Unsloth Q4_K_M version
prepare_and_convert \
    "dequant-q4k-unsloth" \
    "$root_dir/gemma3-quantize/unsloth_4bit_gguf/gemma-3-4b-it-Q4_K_M.gguf" \
    "$root_dir/gemma3-quantize/unsloth_4bit_gguf/mmproj-BF16.gguf"


# # 2. Unsloth Q4_K_M version
# prepare_and_convert \
#     "dequant-q4k-unsloth" \
#     "$root_dir/gemma3-quantize/unsloth_4bit_gguf/gemma-3-4b-it-BF16.gguf" \
#     "$root_dir/gemma3-quantize/unsloth_4bit_gguf/mmproj-BF16.gguf"



# prepare_and_convert \
#     "dequant-q4k-unsloth" \
#     "$root_dir/gemma3-quantize/test.gguf" \
#     "$root_dir/gemma3-quantize/unsloth_4bit_gguf/mmproj-BF16.gguf"
