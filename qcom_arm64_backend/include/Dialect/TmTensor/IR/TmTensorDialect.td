//===- TmTensorDialect.td -------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef TRITONARM64_DIALECT_TMTENSOR_IR_TMTENSOR_DIALECT_TD
#define TRITONARM64_DIALECT_TMTENSOR_IR_TMTENSOR_DIALECT_TD

include "mlir/IR/OpBase.td"

def TmTensor_Dialect : Dialect {
  let name = "tm_tensor";

  let cppNamespace = "::mlir::tm_tensor";

  let summary = "TmTensor operations";

  let description = [{
    TmTensor Dialect for tensor operations including attention mechanisms.
  }];

  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];

  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();
    void registerOperations();
  }];
}

#endif // TRITONARM64_DIALECT_TMTENSOR_IR_TMTENSOR_DIALECT_TD
