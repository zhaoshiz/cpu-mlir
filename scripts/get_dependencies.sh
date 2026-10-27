# ===- get_dependencies.sh --------------------------------------------------===
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

# Init Triton-Shared
git submodule update --init --recursive

# Checkout the Triton commit pinned by Triton-Shared
cd $ROOT_DIR
cd triton && git checkout $(cat ../triton-shared/triton-hash.txt)

# Apply a downstream patch to Triton
git apply < $ROOT_DIR/patches/Triton-Add-ARM64-Backend.patch
git apply < $ROOT_DIR/patches/triton_host_build.patch

cd $ROOT_DIR/triton-shared
git apply < $ROOT_DIR/patches/triton_shared_host_build.patch

cd $ROOT_DIR/..
mkdir -p llvm_host
cd llvm_host
if [ -d "LLVM-22.1.8-Linux-X64" ]; then
    echo "Already extracted llvm."
else
    wget https://github.com/llvm/llvm-project/releases/download/llvmorg-22.1.8/LLVM-22.1.8-Linux-X64.tar.xz
    tar -xJvf LLVM-22.1.8-Linux-X64.tar.xz
fi
