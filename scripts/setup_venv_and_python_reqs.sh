# ===- setup_venv_and_python_reqs.sh ----------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

#!/bin/bash -x

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
ROOT_DIR="$SCRIPT_DIR/.."

python3 -m venv .venv --prompt triton
source .venv/bin/activate

pip install --upgrade pip
pip install -r $ROOT_DIR/triton/python/requirements.txt
pip install pytest numpy
pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels \
    --pre \
    torch==2.8.0+cpu \
    torchvision==0.23.0+cpu \
    torch-mlir \
    torchao==0.10.0+cpu \
    torchtune==0.6.0+cpu \
    transformers==4.52.4
