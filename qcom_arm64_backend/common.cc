//===- common.cc ----------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "common.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/Debug.h"

// Convert vector extension from string to enum type.
arm64VectorExtension getVectorExtFromString (std::string vec_ext) {
  if (vec_ext == "neon")
    return NEON;
  else if (vec_ext == "sve")
    return SVE;
  // Default to SME
  return SME;
}
const char *ARM64VectorExtensionNames[] = {"NEON", "SVE", "SME"};

namespace mlir {
void enablePrintIR(mlir::PassManager &pm, bool printDebugInfo) {
  auto *context = pm.getContext();

  const char *envPassVal = std::getenv("MLIR_DUMP_TRITON_SHARED_SPECIFIC_PASS");
  std::string specificPassDump = envPassVal ? std::string(envPassVal) : "";
  const char *envEnableVal = std::getenv("MLIR_ENABLE_DUMP_TRITON_SHARED");
  bool enableDump = (envEnableVal && std::string(envEnableVal) == "1");
  if (enableDump) {
    mlir::OpPrintingFlags printingFlags;
    const char *envAllDataVal = std::getenv("MLIR_DUMP_ALL_IR_DATA");
    bool enableAllDataDump = (envAllDataVal && std::string(envAllDataVal) == "1");
    if(!enableAllDataDump) { // Default is to not dump constant data
      printingFlags.elideLargeElementsAttrs(1);   // elide after 1 element
      printingFlags.elideLargeResourceString(1);  // elide resource blobs after 1 byte
    }

    if(printDebugInfo) {
      // enable debug info so loc(...) is printed for every op (including
      // loc(unknown)), making it easy to spot where locations are lost.
      printingFlags.enableDebugInfo();
    }

    context->disableMultithreading(); // MT must be disabled to allow MLIR dump
    pm.enableIRPrinting(
        /*shouldPrintBeforePass=*/[](mlir::Pass *pass, mlir::Operation *op) { return false; },
        /*shouldPrintAfterPass=*/[specificPassDump](mlir::Pass *pass, mlir::Operation *op) {
          if(specificPassDump == "" || pass->getName() == specificPassDump)
            return true;
          else
            return false;
        },
        /*printModuleScope=*/true,
        /*printAfterOnlyOnChange=*/false,
        /*printAfterOnlyOnFailure=*/false,
        llvm::errs(), // Output stream (can be a file stream)
        printingFlags
    );
  }
}
};
