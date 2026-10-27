//===- TmTensorAttrs.cpp - TmTensor dialect attributes implementation -----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "Dialect/TmTensor/IR/TmTensorDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tm_tensor;

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "Dialect/TmTensor/IR/TmTensorAttrs.cpp.inc"

void TmTensorDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/TmTensor/IR/TmTensorAttrs.cpp.inc"
      >();
}
