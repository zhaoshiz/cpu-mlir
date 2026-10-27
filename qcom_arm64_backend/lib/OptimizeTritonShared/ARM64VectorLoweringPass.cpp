//===- ARM64VectorLoweringPass.cpp - ARM64 Vector Lowering Pass -----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
// This file implements lowering ops in 'vector' dialect of inner loop so that
// vector to llvm pass can convert to vectorized llvm-ir.
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#include "llvm/Support/Debug.h"

#include "OptimizeTritonShared/OptimizeTritonShared.h"

#define DEBUG_TYPE "triton-arm64-vector-lowering"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace triton;

#define GEN_PASS_DEF_TRITONARM64VECTORLOWERING
#include "OptimizeTritonShared/Passes.h.inc"

namespace {

static LogicalResult doVectorLowering(ModuleOp moduleOp, arm64VectorExtension vecExt) {
  MLIRContext *ctx = moduleOp.getContext();
  IRRewriter rewriter(ctx);

  SmallVector<scf::ForOp> loops;
  moduleOp.walk<WalkOrder::PostOrder>(
      [&](scf::ForOp loop) { loops.push_back(loop); });

  {
    RewritePatternSet patterns(ctx);
    vector::populateVectorMaskLoweringPatternsForSideEffectingOps(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    moduleOp.walk([&](func::FuncOp funcOp) {
      (void)applyPatternsGreedily(funcOp, frozenPatterns);
    });
  }

  for (scf::ForOp loop : loops) {
    moveLoopInvariantCode(loop);
    hoistLoopInvariantSubsets(rewriter, loop);
  }

  {
    RewritePatternSet patterns(ctx);
    vector::populateVectorReductionToContractPatterns(patterns);
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    transform::ApplyCanonicalizationPatternsOp canonicalizationPatterns;
    canonicalizationPatterns.populatePatterns(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    moduleOp.walk([&](func::FuncOp funcOp) {
      (void)applyPatternsGreedily(funcOp, frozenPatterns);
    });
  }

  {
    RewritePatternSet patterns(ctx);
    vector::populateVectorReductionToContractPatterns(patterns);
    vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
    vector::populateVectorContractLoweringPatterns(
        patterns, vector::VectorContractLowering::OuterProduct,
        /*benefit=*/1, /*disableOuterProductLowering=*/vecExt == SME);
    vector::populateDropUnitDimWithShapeCastPatterns(patterns);
    transform::ApplyCanonicalizationPatternsOp canonicalizationPatterns;
    canonicalizationPatterns.populatePatterns(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    moduleOp.walk([&](func::FuncOp funcOp) {
      (void)applyPatternsGreedily(funcOp, frozenPatterns);
      DominanceInfo domInfo(funcOp);
      eliminateCommonSubExpressions(rewriter, domInfo, funcOp);
    });
  }

  {
    using namespace bufferization;
    OneShotBufferizationOptions opts;
    opts.allowReturnAllocsFromLoops = true;
    opts.bufferizeFunctionBoundaries = true;
    opts.setFunctionBoundaryTypeConversion(LayoutMapOption::IdentityLayoutMap);
    BufferizationState bufState;
    if (failed(runOneShotModuleBufferize(moduleOp, opts, bufState)))
      return failure();
  }

  {
    RewritePatternSet patterns(ctx);
    vector::populateVectorMaskOpLoweringPatterns(patterns);
    auto vectorMultiReductionLowering =
        vector::VectorMultiReductionLowering::InnerReduction;
    vector::populateVectorMultiReductionReorderPatterns(
        patterns, vectorMultiReductionLowering);
    vector::populateVectorMultiReductionFlatteningPatterns(
        patterns, vectorMultiReductionLowering);
    vector::populateVectorMultiReductionUnrollingPatterns(
        patterns, vectorMultiReductionLowering);
    vector::populateVectorTransferDropUnitDimsPatterns(patterns);
    vector::populateDropUnitDimWithShapeCastPatterns(patterns);
    transform::ApplyCanonicalizationPatternsOp canonicalizationPatterns;
    canonicalizationPatterns.populatePatterns(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    moduleOp.walk([&](func::FuncOp funcOp) {
      (void)applyPatternsGreedily(funcOp, frozenPatterns);
      DominanceInfo domInfo(funcOp);
      eliminateCommonSubExpressions(rewriter, domInfo, funcOp);
    });
  }

  return success();
}

struct TritonARM64VectorLoweringPass
    : public ::impl::TritonARM64VectorLoweringBase<TritonARM64VectorLoweringPass> {
public:
  explicit TritonARM64VectorLoweringPass(const TritonARM64VectorLoweringOptions &options)
      : TritonARM64VectorLoweringBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, func::FuncDialect,
                    scf::SCFDialect, tensor::TensorDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (failed(doVectorLowering(moduleOp, vectorExt)))
      return signalPassFailure();
    DBG("Module after vector lowering:\n" << moduleOp << "\n");
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonARM64VectorLoweringPass(const TritonARM64VectorLoweringOptions &options) {
  return std::make_unique<TritonARM64VectorLoweringPass>(options);
}
