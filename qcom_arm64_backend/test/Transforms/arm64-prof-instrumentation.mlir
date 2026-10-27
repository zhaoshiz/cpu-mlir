// RUN: env TRITON_ARM64_PROFILE=1 arm64-mlir-opt %s -triton-arm64-prof-instrumentation | FileCheck %s

module {
  func.func @minimal_loop() -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %zero = arith.constant 0.0 : f32

    %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%val = %zero) -> (f32) {
      %next = arith.addf %val, %val : f32
      scf.yield %next : f32
    }

    return %result : f32
  }
}

// CHECK: llvm.func @arm64_prof_func

// function instrumentation
// CHECK-LABEL: func.func @minimal_loop
// CHECK: %[[MARKER:.*]] = llvm.mlir.constant({{[0-9]+}} : i32) : i32
// CHECK: llvm.call @arm64_prof_func(%[[MARKER]]) : (i32) -> ()

// loop instrumentation
// CHECK: %[[LOOP_MARKER:.*]] = llvm.mlir.constant({{[0-9]+}} : i32) : i32
// CHECK: llvm.call @arm64_prof_func(%[[LOOP_MARKER]]) : (i32) -> ()
// CHECK: scf.for
// CHECK: llvm.call @arm64_prof_func(%[[LOOP_MARKER]]) : (i32) -> ()

// CHECK: llvm.call @arm64_prof_func(%[[MARKER]]) : (i32) -> ()
