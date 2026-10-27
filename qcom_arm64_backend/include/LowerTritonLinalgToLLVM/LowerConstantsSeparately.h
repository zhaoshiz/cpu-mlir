//===- LowerConstantsSeparately.h -----------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ARM64_LOWER_CONSTANTS_SEPARATELY_H
#define TRITON_ARM64_LOWER_CONSTANTS_SEPARATELY_H

#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h" // For divideCeil

#include "LowerTritonLinalgToLLVM/Passes.h" // For the base class LowerConstantsSeparatelyBase

#define DEBUG_TYPE "triton-arm64-lower-constants-separately"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_DEF_TRITONARM64LOWERCONSTANTSSEPARATELY
#include "LowerTritonLinalgToLLVM/Passes.h.inc"

namespace {

// Kept only for easier debugging to print the data of a blob (from the
// dialect_resources) that's in hexadecimal format with two characters per byte
void printValueOfConstant(ArrayRef<char> originalData) {
  for (char c : originalData) {
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << (int)(unsigned char)c;
  }
  std::cout << std::endl;
}

// This class implements the main logic of the constant splitting, which will
// move constant tensors to (potentially many) new separate modules, which will
// all be compiled and linked individually.
struct LowerConstantsSeparatelyPass
    : public ::impl::TritonARM64LowerConstantsSeparatelyBase<
          LowerConstantsSeparatelyPass> {

  // Optional external sink: if set, produced modules are written here after
  // runOnOperation() completes, allowing the caller to retrieve them without
  // needing RTTI to cast back to this type.
  std::shared_ptr<std::vector<ModuleOp>> outputSink;

  explicit LowerConstantsSeparatelyPass(
      std::shared_ptr<std::vector<ModuleOp>> sink = nullptr)
      : outputSink(std::move(sink)) {}

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    mlir::registerAllExtensions(registry);
  }

  // Maximum size above which a module is considered "full" and a new one will
  // be created (can happen several times, leading to several const-only mods)
  const int maximumBytesPerSharedObject = 5e8; // 500,000,000 bytes = 0.5 GB

  // Utility function to compute the memory footprint (in bytes) of a constant
  // tensor op, which will be used to know when to create a new mlir module.
  // If passed an operation that's not a valid constant tensor (containing
  // int or float values) it returns 0.
  int estimateConstantTensorMemoryFootprint(mlir::LLVM::GlobalOp op) {
    // Ensure the global is marked constant
    if (!op.getConstant()) {
      return 0;
    }

    mlir::Type type = op.getType();
    mlir::Type previousType;
    int64_t totalElements = 1;

    // Unwrap nested LLVM array types to get shape and element type
    while (auto arrayType = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(type)) {
      totalElements *= arrayType.getNumElements();
      previousType = type;
      type = arrayType.getElementType();
      // Break if type doesn't change (safety against infinite loop)
      if (type == previousType) {
        return 0;
      }
    }

    // Now `type` should be the element type (int or float)
    if (!type.isIntOrFloat()) {
      std::cerr << "Element type is not int or float." << std::endl;
      return 0;
    }

    int64_t elementBitWidth = type.getIntOrFloatBitWidth();
    int64_t elementByteSize = llvm::divideCeil(elementBitWidth, 8);
    int64_t totalSize = totalElements * elementByteSize;

    return totalSize;
  }

  // Main function of the pass, which goes through the module, looking for
  // constants to be moved to (potentially) many new mlir modules. Does so
  // while making sure that each produced module stays under the limit
  // defined by `maximumBytesPerSharedObject`. Takes care not only of
  // moving the constants, but also of moving the associated entries defined
  // in the dialect_resources.
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    mlir::MLIRContext &newContext = *(mod.getContext());

    // Create a new MLIR module in which we'll be putting the current op
    DBG("[LowerConstantsSeparately] Creating a new module for constants only"
        << "\n");
    mlir::ModuleOp newMod =
        mlir::ModuleOp::create(mlir::UnknownLoc::get(&newContext));
    // The size of the new module will be the sum of the weights (in bytes) of
    // the constant tensors it contains. It is 0 for now since it's empty.
    int newModSize = 0;

    // Collection to store the keys of the dense resources needed, which we will
    // find as we discover the constants definitions in the original module
    std::vector<std::string> resourceKeys;

    // Walking the original MLIR module, trying to find global constants to move
    // to the new MLIR module we just made
    mod.walk([&](mlir::Operation *op) {
      // If it's a GlobalOp...
      if (auto globalOp = llvm::dyn_cast<mlir::LLVM::GlobalOp>(op)) {
        // And if it's a constant...
        if (globalOp.getConstant()) {
          // 1 - Copy the constant definition to the new module being created
          // ----------------------------------------------------------------
          // Clone the global constant
          mlir::LLVM::GlobalOp clonedOp = globalOp.clone();
          // Set its linkage attribute to External so it will be accessible
          // from other .o files
          auto linkageAttrClonedOp =
              LLVM::LinkageAttr::get(&newContext, LLVM::Linkage::External);
          clonedOp.setLinkageAttr(linkageAttrClonedOp);

          // Compute the size (in bytes) needed by the current constant
          int currentOpSize = estimateConstantTensorMemoryFootprint(globalOp);

          // If the current op can't fit in the current module for constants
          if (newModSize + currentOpSize > maximumBytesPerSharedObject) {
            // Push the current mod into the collection of produced modules
            producedModules.push_back(newMod);
            // Create a new MLIR module in which we'll be putting the current op
            DBG("[LowerConstantsSeparately] Creating another new "
                "module for constants only"
                << "\n");
            newMod = mlir::ModuleOp::create(mlir::UnknownLoc::get(&newContext));
            // Its size is 0 for now
            newModSize = 0;
          } else {
            // The current module has enough space left, so we'll be using it
            newModSize += currentOpSize;
          }

          // Insert the cloned operation (constant definition) into the new
          // module
          newMod.push_back(clonedOp);

          // Extract the key from the dense_resource attribute and add it to our
          // collection of keys that are defined (and which will therefore need
          // the relevant meta-data copied)
          if (auto denseResourceAttr =
                  mlir::dyn_cast<mlir::DenseResourceElementsAttr>(
                      globalOp->getAttr("value"))) {
            std::string key = denseResourceAttr.getRawHandle().getKey().str();
            resourceKeys.push_back(key);
          }

          // 2 - Replace the constant by an extern in the main module
          // ----------------------------------------------------------------
          OpBuilder builder(globalOp.getContext());
          builder.setInsertionPoint(globalOp);

          // Create a new extern GlobalOp
          auto externGlobalOp =
              LLVM::GlobalOp::create(builder, globalOp.getLoc(),
                                     /*type=*/globalOp.getType(),
                                     /*isConstant=*/false,
                                     /*linkage=*/LLVM::Linkage::External,
                                     /*name=*/globalOp.getSymName(),
                                     /*value=*/Attribute{});

          // Replace the original GlobalOp that was the constant definition with
          // the new "extern" GlobalOp, which references a constant from the
          // outside
          globalOp.getOperation()->replaceAllUsesWith(
              externGlobalOp.getOperation());
          globalOp.erase();
        }
      }
    });

    // Ensure the new module has the needed resources (containing the definition
    // of the constants it uses)
    auto *resourceInterface =
        mod->getContext()
            ->getLoadedDialect<mlir::BuiltinDialect>()
            ->getRegisteredInterface<
                mlir::ResourceBlobManagerDialectInterface>();
    auto *newResourceInterface =
        newMod->getContext()
            ->getLoadedDialect<mlir::BuiltinDialect>()
            ->getRegisteredInterface<
                mlir::ResourceBlobManagerDialectInterface>();

    if (resourceInterface && newResourceInterface) {
      mlir::DialectResourceBlobManager &resourceMap =
          resourceInterface->getBlobManager();
      mlir::DialectResourceBlobManager &newResourceMap =
          newResourceInterface->getBlobManager();

      // For each resource being defined in the new module, move the associated
      // blob from the original "resource blob manager" to the new one
      for (const auto &key : resourceKeys) {
        DialectResourceBlobManager::BlobEntry *blobEntry =
            resourceMap.lookup(key);
        if (blobEntry) {
          // Moving the binary blob
          newResourceMap.insert(key, std::move(*(blobEntry->getBlob())));
        }
      }
      // Copy module-level attributes
      newMod->setAttrs(mod->getAttrs());

    } else {
      std::cerr << "Error: ResourceBlobManagerDialectInterface not found!"
                << std::endl;
    }

    // Push the final module for constants (the non-full one)
    producedModules.push_back(newMod);

    // If an external sink was provided, write the produced modules into it so
    // the caller can retrieve them without needing RTTI.
    if (outputSink)
      *outputSink = producedModules;
  }

  // Method to access the newly produced modules
  std::vector<ModuleOp> getProducedModules() const { return producedModules; }

private:
  // Modules produced by the pass, which contain only constants (moved
  // from the original module)
  std::vector<ModuleOp> producedModules;

}; // end of class

} // end anonymous namespace

#endif // TRITON_ARM64_LOWER_CONSTANTS_SEPARATELY_H
