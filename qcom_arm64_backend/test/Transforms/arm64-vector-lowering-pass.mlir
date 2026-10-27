// RUN: arm64-mlir-opt %s -triton-arm64-vector-lowering | FileCheck %s
// RUN: arm64-mlir-opt %s -triton-arm64-vector-lowering="vector-ext=neon" | FileCheck %s --check-prefix=CHECK-NEON
// RUN: arm64-mlir-opt %s -triton-arm64-vector-lowering="vector-ext=sve" | FileCheck %s --check-prefix=CHECK-SVE

// For SME (default), vector.contract -> vector.outerproduct and outerproduct
// lowering is disabled so vector.outerproduct remains for the fmopa path.
// For NEON/SVE, outerproduct lowering is enabled so vector.outerproduct is
// further lowered to vector.fma.
// CHECK-LABEL: func.func @test_contraction_to_outerproduct
// CHECK-NOT:      vector.contract
// CHECK-COUNT-8:  vector.outerproduct
// CHECK-NEON-LABEL: func.func @test_contraction_to_outerproduct
// CHECK-NEON-NOT:   vector.contract
// CHECK-NEON-NOT:   vector.outerproduct
// CHECK-NEON-COUNT-32: vector.fma
// CHECK-SVE-LABEL: func.func @test_contraction_to_outerproduct
// CHECK-SVE-NOT:   vector.contract
// CHECK-SVE-NOT:   vector.outerproduct
// CHECK-SVE-COUNT-32: vector.fma
func.func @test_contraction_to_outerproduct(%arg0: vector<4x8xf32>, %arg1: vector<8x4xf32>, %arg2: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                         affine_map<(d0, d1, d2) -> (d2, d1)>,
                                         affine_map<(d0, d1, d2) -> (d0, d1)>],
                        iterator_types = ["parallel", "parallel", "reduction"]}
    %arg0, %arg1, %arg2 : vector<4x8xf32>, vector<8x4xf32> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// CHECK-LABEL: func.func @test_multi_reduction
// CHECK-NOT:     vector.multi_reduction
// CHECK-COUNT-4: vector.reduction <add>
func.func @test_multi_reduction(
    %v: vector<4x4xf32>,
    %acc: vector<4xf32>) -> vector<4xf32> {
  %result = vector.multi_reduction <add>, %v, %acc
      [1] : vector<4x4xf32> to vector<4xf32>
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @test_multi_reduction_outer_axis
// CHECK-NOT:     vector.multi_reduction
// CHECK:         vector.transpose
// CHECK-COUNT-4: vector.reduction <add>
func.func @test_multi_reduction_outer_axis(
    %v: vector<4x4xf32>,
    %acc: vector<4xf32>) -> vector<4xf32> {
  %result = vector.multi_reduction <add>, %v, %acc
      [0] : vector<4x4xf32> to vector<4xf32>
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @test_transfer_permutation
// CHECK-NOT:   permutation_map = affine_map<(d0, d1) -> (d1, d0)>
// CHECK:       [[VREAD:%.+]] = vector.transfer_read {{.+}} : memref<4x4xf32>, vector<4x4xf32>
// CHECK:       vector.transpose [[VREAD]]
func.func @test_transfer_permutation(%mem: memref<4x4xf32>) -> vector<4x4xf32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %v = vector.transfer_read %mem[%c0, %c0], %pad
      {permutation_map = affine_map<(d0, d1) -> (d1, d0)>}
      : memref<4x4xf32>, vector<4x4xf32>
  return %v : vector<4x4xf32>
}

// CHECK-LABEL: func.func @test_transfer_write_permutation
// CHECK-NOT:   permutation_map = affine_map<(d0, d1) -> (d1, d0)>
// CHECK:       [[XPOSE:%.+]] = vector.transpose
// CHECK:       vector.transfer_write [[XPOSE]], {{.+}} : vector<4x4xf32>, memref<4x4xf32>
func.func @test_transfer_write_permutation(%mem: memref<4x4xf32>, %v: vector<4x4xf32>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %v, %mem[%c0, %c0]
      {permutation_map = affine_map<(d0, d1) -> (d1, d0)>}
      : vector<4x4xf32>, memref<4x4xf32>
  return
}

// CHECK-LABEL: func.func @test_to_tensor_no_flags
// CHECK-NOT: bufferization.to_tensor
// CHECK-NOT: tensor<8xf32>
// CHECK:      memref.alloc
func.func @test_to_tensor_no_flags() -> tensor<8xf32> {
  %mem = memref.alloc() : memref<8xf32>
  %t = bufferization.to_tensor %mem restrict : memref<8xf32> to tensor<8xf32>
  return %t : tensor<8xf32>
}

// CHECK-LABEL: func.func @test_to_tensor_with_flags
// CHECK-NOT: bufferization.to_tensor
// CHECK-NOT: tensor<8xf32>
// CHECK:      memref.alloc
func.func @test_to_tensor_with_flags() -> tensor<8xf32> {
  %mem = memref.alloc() : memref<8xf32>
  %t = bufferization.to_tensor %mem restrict writable : memref<8xf32> to tensor<8xf32>
  return %t : tensor<8xf32>
}

// CHECK-LABEL: func.func @test_masked_transfer_read_lowering
// CHECK-NOT:   vector.mask
// CHECK:       [[C0:%.+]] = arith.constant 0 : index
// CHECK:       [[CST:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK:       [[READ:%.+]] = vector.transfer_read %arg0[[[C0]]], [[CST]], %arg1 {in_bounds = [true]} : memref<8xf32>, vector<8xf32>
// CHECK:       return [[READ]] : vector<8xf32>
func.func @test_masked_transfer_read_lowering(%arg0: memref<8xf32>, %arg1: vector<8xi1>) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %v = vector.mask %arg1 {
    vector.transfer_read %arg0[%c0], %cst : memref<8xf32>, vector<8xf32>
  } : vector<8xi1> -> vector<8xf32>
  return %v : vector<8xf32>
}

// CHECK-LABEL: func.func @test_masked_transfer_write_lowering
// CHECK-NOT:   vector.mask
// CHECK:       [[C0:%.+]] = arith.constant 0 : index
// CHECK:       vector.transfer_write %arg1, %arg0[[[C0]]], %arg2 {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
func.func @test_masked_transfer_write_lowering(%arg0: memref<8xf32>, %arg1: vector<8xf32>, %arg2: vector<8xi1>) {
  %c0 = arith.constant 0 : index
  vector.mask %arg2 {
    vector.transfer_write %arg1, %arg0[%c0] : vector<8xf32>, memref<8xf32>
  } : vector<8xi1>
  return
}

// CHECK-LABEL: func.func @test_no_vector_ops_unchanged
// CHECK:       arith.addf {{.+}}, {{.+}} : f32
// CHECK:       arith.addf {{.+}}, {{.+}} : vector<4xf32>
// CHECK-NOT:   vector.
func.func @test_no_vector_ops_unchanged(%a: f32, %b: f32,
                                        %v0: vector<4xf32>, %v1: vector<4xf32>) -> (f32, vector<4xf32>) {
  %0 = arith.addf %a, %b : f32
  %1 = arith.addf %v0, %v1 : vector<4xf32>
  return %0, %1 : f32, vector<4xf32>
}

// CHECK-LABEL: func.func @test_rank_reducing_subview_read
// CHECK:       [[SUBVIEW:%.+]] = memref.subview {{.+}} : memref<1x4xf32> to memref<4xf32, strided<[1]>>
// CHECK:       vector.transfer_read [[SUBVIEW]]{{.+}} : memref<4xf32, strided<[1]>>, vector<4xf32>
func.func @test_rank_reducing_subview_read(%mem: memref<1x4xf32>) -> vector<1x4xf32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %v = vector.transfer_read %mem[%c0, %c0], %pad
      {in_bounds = [true, true]}
      : memref<1x4xf32>, vector<1x4xf32>
  return %v : vector<1x4xf32>
}

// CHECK-LABEL: func.func @test_rank_reducing_subview_write
// CHECK:       [[SUBVIEW:%.+]] = memref.subview {{.+}} : memref<1x4xf32> to memref<4xf32, strided<[1]>>
// CHECK:       vector.transfer_write {{.+}}, [[SUBVIEW]]{{.+}} : vector<4xf32>, memref<4xf32, strided<[1]>>
func.func @test_rank_reducing_subview_write(%mem: memref<1x4xf32>, %v: vector<1x4xf32>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %v, %mem[%c0, %c0]
      {in_bounds = [true, true]}
      : vector<1x4xf32>, memref<1x4xf32>
  return
}
