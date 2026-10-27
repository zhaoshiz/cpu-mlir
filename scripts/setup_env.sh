# ===- setup_env.sh ---------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

# Report error if more than one argument is provided
if [ "$#" -gt "1" ]; then
	echo "Usage: $0 [<Path to ARM64 TARGET COMPILER (LLVM)>]"
  exit 1
elif [ "$#" -eq "1" ]; then
  export TARGET_COMPILER=$1
fi

# Target compiler: The LLVM that does MLIR transformations and generates ARM64 code
export LLVM_BINARY_DIR=$TARGET_COMPILER/bin

# Setup Triton-Shared
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
ROOT_DIR="$SCRIPT_DIR/.."
# The build directory is dependent on what version of python the user has
matches=($ROOT_DIR/triton/build/cmake.linux-x86_64-cpython-*)
if [ ${#matches[@]} -gt 1 ]; then
    echo "Error: multiple Triton builds found, cannot determine which to use:"
    printf '  %s\n' "${matches[@]}"
    exit 1
elif [ ! -d "${matches[0]}" ]; then
    echo "Error: no Triton build directory found in $ROOT_DIR/triton/build"
    exit 1
fi
export TRITON_BUILD_DIR=$(realpath "${matches[0]}")
export TRITON_SHARED_OPT_PATH=$TRITON_BUILD_DIR/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt
# Triton compiler options
export TRITON_ALWAYS_COMPILE=1

# add backend bin dir to PATH
export PATH=$TRITON_BUILD_DIR/third_party/qcom_arm64_backend/bin:$PATH
