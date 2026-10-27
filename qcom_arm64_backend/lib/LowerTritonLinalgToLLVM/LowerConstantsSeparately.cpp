//===- LowerConstantsSeparately.cpp ---------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "LowerTritonLinalgToLLVM/LowerConstantsSeparately.h"

#include "mlir/IR/BuiltinOps.h" // For ModuleOp
#include "mlir/Pass/Pass.h"     // For OperationPass

using namespace mlir;
using namespace mlir::triton;

// Create an instance of the LowerConstantsSeparatelyPass
std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createLowerConstantsSeparatelyPass(
    std::shared_ptr<std::vector<ModuleOp>> sink) {
  return std::make_unique<LowerConstantsSeparatelyPass>(std::move(sink));
}
