# ===- build.sh -------------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

#!/bin/bash -x

set -euo pipefail

#HOST_COMPILER: the C/C++ compiler that builds Triton, Triton-Shared, etc.
#TARGET_COMPILER: the LLVM build that lowers MLIR and generates ARM64 code.
if [ "$#" -eq 2 ]; then 
  HOST_COMPILER=$1
  TARGET_COMPILER=$2
elif [ "$#" -eq 1 ] || [ "$#" -gt 2 ]; then
	echo "Usage: $0 <HOST_COMPILER_PATH> <TARGET_COMPILER_PATH>"
  exit 1
fi

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
ROOT_DIR="$SCRIPT_DIR/.."

# Host LLVM to compile Triton, Triton-Shared and our backend
PATH=$HOST_COMPILER/bin:$PATH
TRITON_PLUGIN_DIRS="$ROOT_DIR/triton-shared;$ROOT_DIR/qcom_arm64_backend"

# Build and install Triton against the LLVM above
cd $ROOT_DIR/triton
# DEBUG=1 \
TRITON_PLUGIN_DIRS="$ROOT_DIR/triton-shared;$ROOT_DIR/qcom_arm64_backend" \
TRITON_BUILD_WITH_CLANG_LLD=1 \
LLVM_INCLUDE_DIRS=$TARGET_COMPILER/include \
LLVM_LIBRARY_DIR=$TARGET_COMPILER/lib \
LLVM_SYSPATH=$TARGET_COMPILER \
pip install -e . --verbose 2>&1 | tee install.log
