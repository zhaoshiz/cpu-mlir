//===- Passes.h -----------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ARM64_LOWER_TRITON_LINALG_TO_LLVM_PASSES_H
#define TRITON_ARM64_LOWER_TRITON_LINALG_TO_LLVM_PASSES_H

#include "LowerTritonLinalgToLLVM.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "LowerTritonLinalgToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir
#endif // TRITON_ARM64_LOWER_TRITON_LINALG_TO_LLVM_PASSES_H
