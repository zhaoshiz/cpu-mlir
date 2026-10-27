//===- common.h -----------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ARM64_COMMON_H
#define TRITON_ARM64_COMMON_H

#include <cstdlib>
#include <string>

// TODO: Add SME2, SME2p1 (inc. SVE2p1) when target hardware support them.
enum arm64VectorExtension { NEON = 0, SVE = 1, SME = 2 };
arm64VectorExtension getVectorExtFromString (std::string);
extern const char *ARM64VectorExtensionNames[];

namespace mlir {
class PassManager;
void enablePrintIR(PassManager &pm, bool printDebugInfo);
};
#endif // TRITON_ARM64_COMMON_H
