//===- BubbleUpExtractSlicePass.cpp - sink extract_slice into linalg.generic =//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
// Applies populateBubbleUpExtractSliceOpPatterns so that a tensor.extract_slice
// of a linalg.generic result is replaced by extract_slices on the inputs and a
// smaller generic.  This allows bufferization to write directly into the output
// buffer instead of allocating a full-block-size temporary.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "OptimizeTritonShared/OptimizeTritonShared.h"

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_DEF_TRITONARM64BUBBLEUPEXTRACTSLICE
#include "OptimizeTritonShared/Passes.h.inc"

namespace {

struct TritonARM64BubbleUpExtractSlicePass
    : public ::impl::TritonARM64BubbleUpExtractSliceBase<
          TritonARM64BubbleUpExtractSlicePass> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    linalg::populateBubbleUpExtractSliceOpPatterns(patterns);
    tensor::populateFoldTensorEmptyPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::triton::createTritonARM64BubbleUpExtractSlicePass() {
  return std::make_unique<TritonARM64BubbleUpExtractSlicePass>();
}
