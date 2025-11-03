#!/bin/bash
set -e

VENV_DIR="../venv"
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# Always start with a fresh toolchain
pip install --upgrade pip setuptools wheel

# # ───────────────────────────────────────────────
# # 1️⃣  Install TensorFlow CPU first (handles keras/ml_dtypes/tensorboard)
# # ───────────────────────────────────────────────
# pip install tensorflow-cpu==2.20.0 --upgrade --no-cache-dir

# # ───────────────────────────────────────────────
# # 2️⃣  Install PyTorch 2.6 + torchvision that matches
# # ───────────────────────────────────────────────
# pip install torch==2.6.0+cpu torchvision==0.21.0+cpu --index-url https://download.pytorch.org/whl/cpu

# # ───────────────────────────────────────────────
# # 3️⃣  Install project dependencies
# # ───────────────────────────────────────────────
# pip install -r ../SubModules/ggml/requirements.txt --no-deps

# transformers (editable)
cd ../SubModules/transformers
pip install -e .

cd ../..

# sentence-transformers (editable)
cd SubModules/sentence-transformers
pip install -e .

cd ../..

# ───────────────────────────────────────────────
# 4️⃣  Misc useful deps
# ───────────────────────────────────────────────
pip install huggingface_hub llama-cpp-python accelerate matplotlib
pip install torchvision
# # ───────────────────────────────────────────────
# # 5️⃣  Verify installation
# # ───────────────────────────────────────────────
# python - <<'PYCODE'
# import torch, tensorflow as tf
# print("Torch:", torch.__version__)
# print("TensorFlow:", tf.__version__)
# print("TF devices:", tf.config.list_physical_devices('CPU'))
# PYCODE
