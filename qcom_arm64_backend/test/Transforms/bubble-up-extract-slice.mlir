// RUN: arm64-mlir-opt %s -triton-arm64-bubble-up-extract-slice -split-input-file | FileCheck %s

// Basic case: extract_slice of a linalg.generic result is sunk into the
// generic so that bufferization can write directly into the smaller output.
// CHECK-LABEL: @bubble_up_basic
func.func @bubble_up_basic(%A: tensor<128x64xf32>, %B: tensor<128x64xf32>) -> tensor<64x64xf32> {
  %init = tensor.empty() : tensor<128x64xf32>
  %res = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0, d1)>],
                          iterator_types = ["parallel", "parallel"]}
                  ins(%A, %B : tensor<128x64xf32>, tensor<128x64xf32>)
                  outs(%init : tensor<128x64xf32>) {
  ^bb0(%in: f32, %in2: f32, %out: f32):
    %add = arith.addf %in, %in2 : f32
    linalg.yield %add : f32
  } -> tensor<128x64xf32>

  %slice = tensor.extract_slice %res[0, 0][64, 64][1, 1]
             : tensor<128x64xf32> to tensor<64x64xf32>
  return %slice : tensor<64x64xf32>
}

// After the pass the extract_slice should be gone and the generic should
// produce the smaller tensor<64x64xf32> directly.
// CHECK: linalg.generic
// CHECK-SAME: tensor<64x64xf32>
// CHECK-NOT: tensor.extract_slice

// -----

// No extract_slice consumer — generic must be left unchanged.
// CHECK-LABEL: @no_extract_slice
func.func @no_extract_slice(%A: tensor<128x64xf32>, %B: tensor<128x64xf32>) -> tensor<128x64xf32> {
  %init = tensor.empty() : tensor<128x64xf32>
  %res = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0, d1)>],
                          iterator_types = ["parallel", "parallel"]}
                  ins(%A, %B : tensor<128x64xf32>, tensor<128x64xf32>)
                  outs(%init : tensor<128x64xf32>) {
  ^bb0(%in: f32, %in2: f32, %out: f32):
    %add = arith.addf %in, %in2 : f32
    linalg.yield %add : f32
  } -> tensor<128x64xf32>
  return %res : tensor<128x64xf32>
}

// CHECK: linalg.generic
// CHECK-SAME: tensor<128x64xf32>
