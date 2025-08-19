#!/bin/bash



# download q4_1 from unsloth

cd ..
mkdir -p gpt-oss
cd gpt-oss

mkdir -p unsloth_q4_1_gguf
huggingface-cli download  unsloth/gpt-oss-20b-GGUF --local-dir ./unsloth_q4_1_gguf  --repo-type model