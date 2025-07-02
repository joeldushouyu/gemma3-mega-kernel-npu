#!/bin/bash
VENV_DIR="../venv"

python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

pip install tensorflow-cpu
# install requirements from ggml
pip install -r ../SubModules/ggml/requirements.txt

# install transformers
cd ../SubModules/transformers
pip install -e .

# install other dependencies
pip install huggingface_hub
pip install llama-cpp-python