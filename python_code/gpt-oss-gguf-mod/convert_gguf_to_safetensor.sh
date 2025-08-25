python3 gguf_to_safetensor.py ../../gpt-oss/unsloth_q4_1_gguf/gpt-oss-20b-Q4_1.gguf > debuginf.txt
# python3 reconstruct_tokenizer.py
rm -rf ../gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/extract_safetensors

cp -r ../gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/d666cf3b67006cf8227666739edf25164aaffdeb ../gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/extract_safetensors

# remove all .safetensors
rm -rf ../gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/extract_safetensors/*.safetensors
rm -rf ../gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/extract_safetensors/model.safetensors.index.json
# rm -rf  ../gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/extract_safetensors/special_tokens_map.json
# rm -rf  ../gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/extract_safetensors/tokenizer_config.json
# rm -rf  ../gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/extract_safetensors/tokenizer.json

# rm -rf  ../gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/extract_safetensors/config.json
# rm -rf  ../gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/extract_safetensors/generation_config.json 


cp ./model-00001-of-00001.safetensors ../gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/extract_safetensors/
cp ./model.safetensors.index.json ../gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/extract_safetensors/
# cp ./special_tokens_map.json ../gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/extract_safetensors/
# cp ./tokenizer_config.json ../gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/extract_safetensors/
# cp ./tokenizer.json  ../gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/extract_safetensors/
# cp ./config.json ../gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/extract_safetensors/
# cp ./generation_config.json ../gpt-oss-evaluation/hf_cache/models--openai--gpt-oss-20b/snapshots/extract_safetensors/