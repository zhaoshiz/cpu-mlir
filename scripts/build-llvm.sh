# ===- build-llvm.sh --------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

if [ "$#" -lt "4" ] || [ "$#" -gt "5" ]; then
  echo "Usage: $0 <HOST_COMPILER> <LLVM_SOURCE_DIR> <LLVM_BUILD_DIR> <LLVM_INSTALL_DIR> [<GCC_TOOLCHAIN>]"
  exit 1
fi

HOST_COMPILER=$1
SRC_DIR=$2
BUILD_DIR=$3
INSTALL_DIR=$4

export PATH=$HOST_COMPILER/bin:$PATH

echo "Building llvm for triton"
mkdir -p $BUILD_DIR && pushd $BUILD_DIR
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_TARGETS_TO_BUILD="AArch64;X86;NVPTX;AMDGPU" \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm;clang;lld" \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
  -DMLIR_RUN_ARM_SVE_TESTS=OFF \
  -DMLIR_RUN_ARM_SME_TESTS=OFF \
  $SRC_DIR/llvm 2>&1 | tee cmake.log
ninja 2>&1 | tee build.log
ninja install 2>&1 | tee install.log

# If the device is AArch64 (aarch64 on Linux and arm64 on Mac) we can skip
# building c++ libs and compiler-rt since they are already availible
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    echo "Building on AArch64, skipping compiler-rt and c++ libs"
    exit 0
fi

export PATH="${INSTALL_DIR}/bin:${PATH}"
RESOURCE_DIR="$("${INSTALL_DIR}/bin/clang" -print-resource-dir)"
AARCH64_LINUX_TRIPLE="aarch64-linux-gnu"
AARCH64_SYSROOT="/usr/aarch64-linux-gnu"

echo "Building compiler-rt for AArch64"
COMPILER_RT_FLAGS="--sysroot=${AARCH64_SYSROOT} --target=aarch64-linux-gnu -mcpu=cortex-a53"
mkdir -p compiler-rt-aarch64-linux && pushd compiler-rt-aarch64-linux
cmake -G Ninja \
    -DCMAKE_INSTALL_PREFIX="${RESOURCE_DIR}" \
    -DCMAKE_TRY_COMPILE_TARGET_TYPE="STATIC_LIBRARY" \
    -DCMAKE_C_COMPILER="${INSTALL_DIR}/bin/clang" \
    -DCMAKE_CXX_COMPILER="${INSTALL_DIR}/bin/clang++" \
    -DCMAKE_PREFIX_PATH="${INSTALL_DIR}" \
    -DCMAKE_C_FLAGS="${COMPILER_RT_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${COMPILER_RT_FLAGS}" \
    -DCMAKE_ASM_FLAGS="${COMPILER_RT_FLAGS}" \
    -DCMAKE_SYSTEM_NAME="Generic" \
    -DCOMPILER_RT_DEFAULT_TARGET_TRIPLE="${AARCH64_LINUX_TRIPLE}" \
    -DCOMPILER_RT_OS_DIR="${AARCH64_LINUX_TRIPLE}" \
    -DCOMPILER_RT_TEST_COMPILER_CFLAGS="${COMPILER_RT_FLAGS}" \
    -DCOMPILER_RT_TEST_COMPILER="${INSTALL_DIR}/bin/clang" \
    -DCMAKE_BUILD_TYPE="Release" \
    -DLLVM_ENABLE_ASSERTIONS:BOOL="ON" \
    $SRC_DIR/compiler-rt
ninja 2>&1 | tee compiler-rt-build.log
ninja install 2>&1 | tee compiler-rt-install.log
popd # compiler-rt-aarch64-linux

if [ "$#" -ne "5" ]; then
    echo "No GCC toolchain provided, cannot build c++ libs for AArch64"
    exit 1
fi

echo "Building c++ libs for AArch64"
GCC_TOOLCHAIN=$5
CXX_LIBS_FLAGS="-target ${AARCH64_LINUX_TRIPLE} --sysroot=${AARCH64_SYSROOT} --gcc-toolchain=${GCC_TOOLCHAIN} -mcpu=cortex-a53 -ffunction-sections -fdata-sections -D_GNU_SOURCE -nostdlib"
RELEASE_FLAGS="-Os -DNDEBUG"

mkdir -p c++-libs-aarch64-linux && pushd c++-libs-aarch64-linux
cmake -G Ninja -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}/${AARCH64_LINUX_TRIPLE}" -DCMAKE_BUILD_TYPE="Release" -DCMAKE_C_COMPILER="clang" -DCMAKE_CXX_COMPILER="clang++" \
    -DHAVE_LIBCXXABI="True" -DCMAKE_SYSTEM_NAME="Linux" \
    -DCMAKE_C_FLAGS_RELEASE="${RELEASE_FLAGS}" \
    -DCMAKE_CXX_FLAGS_RELEASE="${RELEASE_FLAGS}" \
    -DCMAKE_C_FLAGS="${CXX_LIBS_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${CXX_LIBS_FLAGS}" \
    -DCMAKE_ASM_FLAGS="${CXX_LIBS_FLAGS}" \
    -DLIBCXX_ENABLE_SHARED="False" \
    -DLIBCXX_ENABLE_ABI_LINKER_SCRIPT="False" \
    -DLIBCXXABI_USE_LLVM_UNWINDER="True" \
    -DLIBCXXABI_ENABLE_SHARED="False" \
    -DLIBCXXABI_ENABLE_WERROR="True" \
    -DLIBUNWIND_ENABLE_SHARED="False" \
    -DLIBUNWIND_TARGET_TRIPLE="${AARCH64_LINUX_TRIPLE}" \
    -S "${SRC_DIR}/runtimes" "-DLLVM_ENABLE_RUNTIMES=libcxx;libcxxabi;libunwind"
ninja 2>&1 | tee c++-libs-build.log
ninja install 2>&1 | tee c++-libs-install.log
popd # c++-libs-aarch64-linux
popd # $BUILD_DIR
