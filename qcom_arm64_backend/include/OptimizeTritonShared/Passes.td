//===- Passes.td ----------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ARM64_OPTIMIZE_TRITON_SHARED_PASSES
#define TRITON_ARM64_OPTIMIZE_TRITON_SHARED_PASSES

include "mlir/Pass/PassBase.td"

def VectorExt : Option<"vectorExt", "vector-ext", "arm64VectorExtension", "SME",
    "enable SME lowering",
    [{::llvm::cl::values(
        clEnumValN(arm64VectorExtension::NEON, "neon", "NEON"),
        clEnumValN(arm64VectorExtension::SVE, "sve", "SVE"),
        clEnumValN(arm64VectorExtension::SME, "sme", "SME")
    )}]>;

def TritonARM64LinalgTiling :
    Pass<"triton-arm64-linalg-tiling", "mlir::ModuleOp"> {
  let summary = "Pass to tile tensor linalg ops. Presently, "
                " just for vectorization.";

  let options = [VectorExt];

  let constructor = "triton::createTritonARM64LinalgTilingPass()";
}

def TritonARM64LinalgVectorization :
    Pass<"triton-arm64-linalg-vectorization", "mlir::ModuleOp"> {
  let summary = "Pass to vectorize tensor linalg ops.";

  let options = [VectorExt];

  let constructor = "triton::createTritonARM64LinalgVectorizationPass()";
}

def TritonARM64VectorLowering :
    Pass<"triton-arm64-vector-lowering", "mlir::ModuleOp"> {
  let summary = "Pass to simplify vector multi-reduction and contraction.";

  let options = [VectorExt];

  let constructor = "triton::createTritonARM64VectorLoweringPass()";
}

def TritonARM64RVO :
    Pass<"triton-arm64-rvo", "func::FuncOp"> {
  let summary = "Hoist output reinterpret_cast early to enable output copy elision in bufferization";
  let constructor = "triton::createTritonARM64RVOPass()";
}

def TritonARM64PuntBuffer :
    Pass<"triton-arm64-punt-buffer", "func::FuncOp"> {
  let summary = "Erase copies of func-input buffers to local allocs before to_tensor";
  let constructor = "triton::createTritonARM64PuntBufferPass()";
}

def TritonARM64BubbleUpExtractSlice :
    Pass<"triton-arm64-bubble-up-extract-slice", "func::FuncOp"> {
  let summary = "Sink tensor.extract_slice through linalg.generic to reduce computation size";
  let constructor = "triton::createTritonARM64BubbleUpExtractSlicePass()";
}

def TritonARM64MatmulToMmt4D :
    Pass<"triton-arm64-matmul-to-mmt4d", "mlir::ModuleOp"> {
  let summary = "Pass to lower linalg.matmul to linalg.mmt4d.";

  let constructor = "triton::createTritonARM64MatmulToMmt4DPass()";
}

#endif
