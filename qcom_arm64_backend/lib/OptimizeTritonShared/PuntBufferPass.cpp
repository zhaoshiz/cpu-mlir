//===- PuntBufferPass.cpp - eliminate read-only input copies before to_tensor =//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
// Removes memref.copy + alloc pairs where the source is a view of a function
// input argument and the only consumer of the alloc is bufferization.to_tensor.
// Instead, to_tensor is pointed directly at the source (read-only alias).
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "OptimizeTritonShared/OptimizeTritonShared.h"

#define DEBUG_TYPE "triton-arm64-punt-buffer"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_DEF_TRITONARM64PUNTBUFFER
#include "OptimizeTritonShared/Passes.h.inc"

namespace {

struct TritonARM64PuntBufferPass
    : public ::impl::TritonARM64PuntBufferBase<TritonARM64PuntBufferPass> {

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(&getContext());

    BufferViewFlowAnalysis analysis(funcOp);

    // Returns the backing func argument that `v` is a view of, or a null
    // BlockArgument if `v` is not derived from any func argument.
    auto getBackingInputArg = [&](Value v) -> BlockArgument {
      for (BlockArgument arg : funcOp.getArguments()) {
        if (!isa<BaseMemRefType>(arg.getType()))
          continue;
        for (Value view : analysis.resolve(arg))
          if (view == v)
            return arg;
      }
      return {};
    };

    // Collect input args that are written to via materialize_in_destination.
    // If an alloc copies from such an arg and that arg is later overwritten,
    // the alloc is a necessary scratch buffer (e.g. for a swap) and must not
    // be eliminated.
    SmallVector<BlockArgument> writtenInputArgs;
    funcOp.walk([&](bufferization::MaterializeInDestinationOp mat) {
      if (BlockArgument arg = getBackingInputArg(mat.getDest()))
        if (!llvm::is_contained(writtenInputArgs, arg))
          writtenInputArgs.push_back(arg);
    });

    // For the direct pattern (copy target is the alloc), srcBacking and allocOp
    // are null — transform via replaceAllUsesWith(target, source).
    // For the through-subview pattern (copy target is a subview of the alloc),
    // srcBacking is the memref to plug into to_tensor, and allocOp is the alloc
    // to clean up along with the subview.
    struct Candidate {
      memref::CopyOp copyOp;
      bufferization::ToTensorOp toTensor;
      memref::AllocOp allocOp;
      Value srcBacking;
    };
    SmallVector<Candidate> candidates;

    funcOp.walk([&](memref::CopyOp copyOp) {
      DBG("checking: " << copyOp);

      BlockArgument backingArg = getBackingInputArg(copyOp.getSource());
      if (!backingArg) {
        DBG("  source is not a view of a func input arg");
        return WalkResult::advance();
      }
      // If the backing arg is also a write destination (e.g. a swap kernel),
      // the alloc is a necessary scratch buffer — don't eliminate it.
      if (llvm::is_contained(writtenInputArgs, backingArg)) {
        DBG("  source backing arg is also a write target; keeping alloc");
        return WalkResult::advance();
      }

      // --- Pattern 1: copy target is directly an alloc ---
      if (auto directAlloc = copyOp.getTarget().getDefiningOp<memref::AllocOp>()) {
        // Copy must be at the same nesting level as the alloc; if it's inside a
        // nested loop the alloc is only partially written per copy and punting
        // to the copy source would produce a shape mismatch in to_tensor.
        if (copyOp->getBlock() != directAlloc->getBlock())
          return WalkResult::advance();
        auto uses = copyOp.getTarget().getUses();
        if (std::distance(uses.begin(), uses.end()) != 2)
          return WalkResult::advance();

        Operation *first = uses.begin()->getOwner();
        Operation *second = std::next(uses.begin())->getOwner();
        Operation *other = (first == copyOp) ? second : first;

        auto toTensor = dyn_cast<bufferization::ToTensorOp>(other);
        if (!toTensor)
          return WalkResult::advance();

        DBG("  punting copy to to_tensor (direct)");
        candidates.push_back({copyOp, toTensor, nullptr, nullptr});
        return WalkResult::advance();
      }

      // --- Pattern 2: copy target is a subview of an alloc ---
      auto dstSubview = copyOp.getTarget().getDefiningOp<memref::SubViewOp>();
      if (!dstSubview)
        return WalkResult::advance();

      auto allocOp = dstSubview.getSource().getDefiningOp<memref::AllocOp>();
      if (!allocOp)
        return WalkResult::advance();

      // Copy must be at the same nesting level as the alloc; a copy inside a
      // loop only partially fills the alloc, so the source shape doesn't match
      // the full alloc shape that to_tensor expects.
      if (copyOp->getBlock() != allocOp->getBlock())
        return WalkResult::advance();

      // Alloc must have exactly 2 uses: dstSubview and a to_tensor.
      auto allocUses = allocOp.getResult().getUses();
      if (std::distance(allocUses.begin(), allocUses.end()) != 2)
        return WalkResult::advance();

      Operation *first = allocUses.begin()->getOwner();
      Operation *second = std::next(allocUses.begin())->getOwner();
      Operation *other = (first == dstSubview) ? second : first;

      auto toTensor = dyn_cast<bufferization::ToTensorOp>(other);
      if (!toTensor)
        return WalkResult::advance();

      // The backing source for to_tensor is the source of the copy's source
      // subview (e.g. the reinterpret_cast of the input arg), so that to_tensor
      // reads the same shape as the original alloc.
      Value srcBacking;
      if (auto srcSubview = copyOp.getSource().getDefiningOp<memref::SubViewOp>())
        srcBacking = srcSubview.getSource();
      else
        srcBacking = copyOp.getSource();

      DBG("  punting copy to to_tensor (through subview)");
      candidates.push_back({copyOp, toTensor, allocOp, srcBacking});
      return WalkResult::advance();
    });

    for (auto &c : candidates) {
      if (c.allocOp) {
        // Through-subview pattern: redirect to_tensor to the input backing, then
        // clean up the copy, the dst subview, and the alloc.
        auto dstSubview = c.copyOp.getTarget().getDefiningOp<memref::SubViewOp>();
        rewriter.modifyOpInPlace(c.toTensor, [&]() {
          c.toTensor->setOperand(0, c.srcBacking);
        });
        rewriter.eraseOp(c.copyOp);
        if (dstSubview && dstSubview.getResult().use_empty())
          rewriter.eraseOp(dstSubview);
        if (c.allocOp.getResult().use_empty())
          rewriter.eraseOp(c.allocOp);
      } else {
        rewriter.replaceAllUsesWith(c.copyOp.getTarget(), c.copyOp.getSource());
        rewriter.eraseOp(c.copyOp);
      }
      c.toTensor->removeAttr("writable");
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::triton::createTritonARM64PuntBufferPass() {
  return std::make_unique<TritonARM64PuntBufferPass>();
}
