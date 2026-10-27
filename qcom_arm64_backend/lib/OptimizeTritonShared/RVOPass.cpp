//===- RVOPass.cpp - output copy elision (return value optimization) ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
// Moves the reinterpret_cast of the output func-arg to an early position so
// that one-shot-bufferize (run inside ARM64VectorLoweringPass) can write the
// result directly into the output buffer without a copy.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "OptimizeTritonShared/OptimizeTritonShared.h"

#define DEBUG_TYPE "triton-arm64-rvo"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_DEF_TRITONARM64RVO
#include "OptimizeTritonShared/Passes.h.inc"

namespace {

using MaterializeOp = bufferization::MaterializeInDestinationOp;
using ReinterpretOp = memref::ReinterpretCastOp;
using SubViewOp = memref::SubViewOp;
using CandidateTy =
    std::tuple<MaterializeOp, ReinterpretOp, Value, std::optional<SubViewOp>>;
using CandidatesTy = SmallVector<CandidateTy>;

// Replace linalg.generic outs operands whose block argument is unused in the
// region body with tensor.empty. Removes ins/outs aliasing that would force
// one-shot bufferization to allocate a copy.
static void replaceAllocsWithEmpty(func::FuncOp funcOp, IRRewriter &rewriter) {
  funcOp.walk([&](linalg::GenericOp genericOp) {
    for (unsigned i = 0; i < genericOp.getNumDpsInits(); ++i) {
      OpOperand *operand = genericOp.getDpsInitOperand(i);
      if (!genericOp.getMatchingBlockArgument(operand).use_empty())
        continue;
      Value val = operand->get();
      // Replacing an scf.for iter arg with tensor.empty() breaks the buffer
      // equivalence between the yield operand and the iter bbArg required by
      // one-shot-bufferize. The in-place write is safe as-is (identity maps).
      if (auto bbArg = dyn_cast<BlockArgument>(val)) {
        Operation *owner = bbArg.getOwner()->getParentOp();
        if (isa<scf::ForOp>(owner))
          continue;
        // A tensor func arg is already caller-allocated. Replacing it with
        // tensor.empty() severs the only use of that buffer, so
        // empty-tensor-elimination has nothing to fold into and bufferize
        // emits a memref.alloc per call
        if (isa<func::FuncOp>(owner) &&
            !llvm::is_contained(genericOp.getDpsInputs(), val))
          continue;
      }
      auto tType = dyn_cast<TensorType>(val.getType());
      if (!tType)
        continue;
      auto loc = genericOp.getLoc();
      rewriter.setInsertionPointAfterValue(val);
      auto empty = tensor::EmptyOp::create(
          rewriter, loc, tensor::getMixedSizes(rewriter, loc, val),
          tType.getElementType());
      genericOp.setDpsInitOperand(i, empty.getResult());
    }
  });
}

static std::pair<bool, Value>
isaViewOfFuncInputBuffer(Value value, func::FuncOp funcOp) {
  BufferViewFlowAnalysis analysis(funcOp);
  for (BlockArgument arg : funcOp.getArguments()) {
    if (!isa<BaseMemRefType>(arg.getType()))
      continue;
    for (Value view : analysis.resolve(arg))
      if (view == value)
        return {true, arg};
  }
  return {false, nullptr};
}

static void populateCandidate(MaterializeOp op, func::FuncOp funcOp,
                               CandidatesTy &candidates) {
  // Try direct pattern: materialize → reinterpret_cast → func_arg.
  auto castOp = op.getDest().getDefiningOp<ReinterpretOp>();
  std::optional<SubViewOp> subviewOp;

  if (!castOp) {
    // Try via subview: materialize → subview → reinterpret_cast → func_arg.
    auto sv = op.getDest().getDefiningOp<SubViewOp>();
    if (!sv)
      return;
    castOp = sv.getSource().getDefiningOp<ReinterpretOp>();
    if (!castOp)
      return;
    subviewOp = sv;
  }

  auto [isViewOf, funcArg] =
      isaViewOfFuncInputBuffer(castOp.getSource(), funcOp);
  if (!isViewOf || !funcArg.hasOneUse())
    return;
  candidates.emplace_back(op, castOp, funcArg, subviewOp);
}

static bool
doValuesDominateInsertionPoint(const DominanceInfo &domInfo,
                                Operation *insertionPoint,
                                const SmallVector<Value> &neededValues) {
  for (Value val : neededValues) {
    if (auto bbArg = dyn_cast<BlockArgument>(val)) {
      Block *owner = bbArg.getOwner();
      if (!owner->findAncestorOpInBlock(*insertionPoint))
        return false;
    } else {
      auto opResult = cast<OpResult>(val);
      if (!domInfo.properlyDominates(opResult.getOwner(), insertionPoint))
        return false;
    }
  }
  return true;
}

static bool insertionPointDominatesUses(const DominanceInfo &domInfo,
                                        Operation *insertionPoint,
                                        Operation *op) {
  return llvm::all_of(op->getUsers(), [&](Operation *user) {
    return domInfo.dominates(insertionPoint, user);
  });
}

static Operation *findEarlyInsertionPoint(CandidateTy &candidate) {
  DominanceInfo domInfo;
  SmallVector<Operation *> insertionPointCandidates;

  auto bbArg = cast<BlockArgument>(std::get<2>(candidate));
  insertionPointCandidates.push_back(
      &bbArg.getOwner()->getOperations().front());

  SmallVector<Value> neededValues;
  auto castOp = std::get<1>(candidate);
  auto subviewOp = std::get<3>(candidate);

  llvm::append_range(neededValues, castOp.getOffsets());
  llvm::append_range(neededValues, castOp.getSizes());
  llvm::append_range(neededValues, castOp.getStrides());

  if (subviewOp) {
    llvm::append_range(neededValues, subviewOp->getOffsets());
    llvm::append_range(neededValues, subviewOp->getSizes());
    llvm::append_range(neededValues, subviewOp->getStrides());
  }

  for (Value val : neededValues) {
    if (auto bbArg = dyn_cast<BlockArgument>(val)) {
      insertionPointCandidates.push_back(
          &bbArg.getOwner()->getOperations().front());
    } else {
      insertionPointCandidates.push_back(val.getDefiningOp()->getNextNode());
    }
  }

  for (Operation *insertionPoint : insertionPointCandidates) {
    if (!doValuesDominateInsertionPoint(domInfo, insertionPoint, neededValues))
      continue;
    if (!insertionPointDominatesUses(domInfo, insertionPoint, castOp))
      continue;
    return insertionPoint;
  }
  return nullptr;
}

static void replaceReinterpretOp(CandidatesTy &candidates,
                                  IRRewriter &rewriter) {
  for (auto &candidate : candidates) {
    auto materializeOp = std::get<0>(candidate);
    auto castOp = std::get<1>(candidate);
    auto subviewOp = std::get<3>(candidate);
    DBG("current candidate: " << materializeOp);
    DBG(" -> dst castOp : " << castOp);

    auto insertionPoint = findEarlyInsertionPoint(candidate);
    if (!insertionPoint) {
      DBG(" -> copy elision not possible for : " << materializeOp);
      continue;
    }
    rewriter.setInsertionPoint(insertionPoint);
    DBG(" -> insertion point : " << *insertionPoint);

    auto newCast = ReinterpretOp::create(
        rewriter, castOp.getLoc(), castOp.getType(), castOp.getSource(),
        castOp.getOffsets(), castOp.getSizes(), castOp.getStrides(),
        castOp.getStaticOffsets(), castOp.getStaticSizes(),
        castOp.getStaticStrides());
    rewriter.replaceAllUsesWith(castOp.getResult(), newCast.getResult());

    if (subviewOp) {
      // Hoist the subview immediately after the new cast. All subview operands
      // are defined at this point (findEarlyInsertionPoint ensured it), and the
      // source already points to newCast via replaceAllUsesWith above.
      rewriter.setInsertionPointAfter(newCast);
      Operation *newSubview = rewriter.clone(*subviewOp->getOperation());
      rewriter.replaceAllUsesWith(subviewOp->getResult(),
                                  newSubview->getResult(0));
    }

    materializeOp.setRestrict(true);
    DBG(" -> copy elision succeeded! replacement:" << newCast);
  }
}

struct TritonARM64RVOPass
    : public ::impl::TritonARM64RVOBase<TritonARM64RVOPass> {

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(&getContext());
    CandidatesTy candidates;

    replaceAllocsWithEmpty(funcOp, rewriter);
    funcOp.walk([&](MaterializeOp op) {
      populateCandidate(op, funcOp, candidates);
      return WalkResult::advance();
    });
    replaceReinterpretOp(candidates, rewriter);
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::triton::createTritonARM64RVOPass() {
  return std::make_unique<TritonARM64RVOPass>();
}
