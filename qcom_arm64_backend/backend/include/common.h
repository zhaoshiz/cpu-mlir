//===- common.h -----------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef COMMON_H
#define COMMON_H

#include <cstdint>

template<typename T, int N>
struct MemRefDescriptor {
    T *allocated;
    T *aligned;
    int64_t offset;
    int64_t sizes[N];
    int64_t strides[N];
};

typedef struct MemRefDescriptor<float, 4> Ty_MemRefDescFloat4D;
typedef struct MemRefDescriptor<float, 3> Ty_MemRefDescFloat3D;
typedef struct MemRefDescriptor<float, 2> Ty_MemRefDescFloat2D;
typedef struct MemRefDescriptor<float, 1> Ty_MemRefDescFloat1D;
typedef struct MemRefDescriptor<_Float16, 1> Ty_MemRefDescHalf1D;
typedef struct MemRefDescriptor<_Float16, 2> Ty_MemRefDescHalf2D;

#endif  //COMMON_H
