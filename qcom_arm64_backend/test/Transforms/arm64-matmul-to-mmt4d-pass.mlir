// RUN: arm64-mlir-opt %s -triton-arm64-matmul-to-mmt4d | FileCheck %s

// CHECK-LABEL: func.func @test_matmul
// CHECK:       linalg.mmt4d
func.func @test_matmul(%A: tensor<?x?xf32>,
                       %B: tensor<?x?xf32>,
                       %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul
    ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
