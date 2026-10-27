//===- ARM64MatmulToMmt4DPass.cpp -----------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
// This file implements the conversion of linalg.matmul to a sequence of
// linalg.pack, linalg.mmt4d, and linalg.unpack operations.
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "OptimizeTritonShared/OptimizeTritonShared.h"

#define DEBUG_TYPE "triton-arm64-matmul-to-mmt4d"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace triton;
using namespace mlir::linalg;

#define GEN_PASS_DEF_TRITONARM64MATMULTOMMT4D
#include "OptimizeTritonShared/Passes.h.inc"

namespace {

struct LowerPackPattern : public OpRewritePattern<PackOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PackOp packOp,
                                PatternRewriter &rewriter) const override {
    return linalg::lowerPack(rewriter, packOp,
                             /*lowerPadLikeWithInsertSlice*/false);
  }
};

struct LowerUnPackPattern : public OpRewritePattern<UnPackOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnPackOp unPackOp,
                                PatternRewriter &rewriter) const override {
    return linalg::lowerUnPack(rewriter, unPackOp,
                             /*lowerUnpadLikeWithInsertSlice*/false);
  }
};

struct TritonARM64MatmulToMmt4DPass
    : public ::impl::TritonARM64MatmulToMmt4DBase<TritonARM64MatmulToMmt4DPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto ctx = moduleOp.getContext();
    RewritePatternSet patterns(ctx);

    ControlBlockPackMatmulFn controlFn =
        [&](linalg::LinalgOp op) -> BlockPackMatmulOptions {
      BlockPackMatmulOptions options;
      // TODO: figure out optimal block sizes
      options.blockFactors = SmallVector<int64_t>{32, 32, 1};
      return options;
    };

    patterns.add<LowerPackPattern>(ctx);
    patterns.add<LowerUnPackPattern>(ctx);

    linalg::populateBlockPackMatmulPatterns(patterns, controlFn);
    linalg::populateLinalgGenericOpsSpecializationPatterns(patterns);
    linalg::populateFoldAddIntoDestPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(),
                                     std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonARM64MatmulToMmt4DPass() {
  return std::make_unique<TritonARM64MatmulToMmt4DPass>();
}
