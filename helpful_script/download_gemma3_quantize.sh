#!/bin/bash

# download from google's official 

cd ..
mkdir -p gemma3-quantize
cd gemma3-quantize

mkdir -p google_4bit_gguf
huggingface-cli download  google/gemma-3-4b-it-qat-q4_0-gguf --local-dir ./google_4bit_gguf  --repo-type model

mkdir -p unsloth_4bit_gguf
huggingface-cli download  unsloth/gemma-3-4b-it-GGUF --local-dir ./unsloth_4bit_gguf  --repo-type model