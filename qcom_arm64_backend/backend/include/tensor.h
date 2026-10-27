//===- tensor.h ------------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef TENSOR_H
#define TENSOR_H

#include <cerrno>
#include <cstring>
#include <iterator>
#include <algorithm>
#include <cstdint>
#include <string>
#include "malloc.h"

#include "common.h"

enum MemType { HEAP };
enum DataType {
    Half = 1,
    Float = 2,
    Double = 3,
    Int8 = 4,
    Int16 = 5,  // short
    Int32 = 6,  // long
    Int64 = 7,  // long long
    UInt8 = 8,
    UInt16 = 9,
    UInt32 = 10,
};

template<int N>
struct TensorShape {
    int64_t sizes[N];
    int64_t strides[N];
    int64_t alignment;
};

template<typename T, int N>
class Tensor {
    MemRefDescriptor<T, N> *memRefDesc = nullptr;
    bool owns_memref = false;
    TensorShape<N> _shape;
    MemType _mem;
    bool need_alloc;
    int64_t _size;

    static TensorShape<N> createShape(MemRefDescriptor<T, N> &desc, int alignment) {
        TensorShape<N> temp;
        temp.alignment = alignment;
        if constexpr (N > 0) { // Check required to properly handle scalar case
            std::copy(std::begin(desc.sizes), std::end(desc.sizes), std::begin(temp.sizes));
            std::copy(std::begin(desc.strides), std::end(desc.strides), std::begin(temp.strides));
        }
        return temp;
    }

public:
    T *buf;
    Tensor(const Tensor &) = delete;
    Tensor(TensorShape<N> shape, T *prealloc_buf, MemType mem)
        : _shape(shape), buf(prealloc_buf), _mem(mem), need_alloc(false) {
    }
    Tensor(TensorShape<N> shape, MemType mem)
        : _shape(shape), _mem(mem), need_alloc(true) {
        _size = 1;
        for (int i = 0; i < N; i++)
            _size = std::max(_size, _shape.sizes[i] * _shape.strides[i]);

        switch (_mem) {
        case MemType::HEAP:
            buf = (T *)memalign(_shape.alignment, _size * sizeof(T));
            break;
        }
    }
    Tensor(MemRefDescriptor<T, N> &desc, MemType mem, int alignment)
        : memRefDesc(&desc), _shape(createShape(desc, alignment)), _mem(mem), need_alloc(false), buf(desc.aligned) {
        _size = 1;
        for (int i = 0; i < N; i++)
            _size = std::max(_size, _shape.sizes[i] * _shape.strides[i]);
    }

    MemRefDescriptor<T, N> *toMemRefDesc() {
        if (!memRefDesc) {
            memRefDesc = (MemRefDescriptor<T, N> *)memalign(8, sizeof(MemRefDescriptor<T, N>));
            memRefDesc->allocated = buf;
            memRefDesc->aligned = buf;
            memRefDesc->offset = 0;
            if constexpr (N > 0) { // Check required to properly handle scalar case
                std::copy(std::begin(_shape.sizes), std::end(_shape.sizes), std::begin(memRefDesc->sizes));
                std::copy(std::begin(_shape.strides), std::end(_shape.strides), std::begin(memRefDesc->strides));
            }
            owns_memref = true;
        }
        return memRefDesc;
    }

    int64_t *get_shape() {
        return _shape.sizes;
    }

    int64_t get_rank() {
        return N;
    }

    int32_t get_dtype() {
        if (std::is_same<T, _Float16>::value || std::is_same<T, __fp16>::value) {
            return DataType::Half;
        } else if (std::is_same<T, float>::value) {
            return DataType::Float;
        } else if (std::is_same<T, double>::value) {
            return DataType::Double;
        } else if (std::is_same<T, int8_t>::value) {
            return DataType::Int8;
        } else if (std::is_same<T, int16_t>::value) {
            return DataType::Int16;
        } else if (std::is_same<T, int32_t>::value) {
            return DataType::Int32;
        } else if (std::is_same<T, int64_t>::value) {
            return DataType::Int64;
        } else if (std::is_same<T, uint8_t>::value) {
            return DataType::UInt8;
        } else if (std::is_same<T, uint16_t>::value) {
            return DataType::UInt16;
        } else if (std::is_same<T, uint32_t>::value) {
            return DataType::UInt32;
        } else if (std::is_same<T, int>::value) {
            return DataType::Int32;
        } else {
            throw std::runtime_error("Unsupported tensor's data type");
        }
    }

    // This is required for the simulator because of a strange interaction
    // where calling free() in the class destructor requires unnecessary
    // pthread symbols we cannot resolve in the simulator
    void free_mem() {
        if (!need_alloc) {
            std::runtime_error("Cannot free memory which tensor object "
                               "does not own");
        }
        switch (_mem) {
        case MemType::HEAP:
            free(buf);
            break;
        }
        buf = nullptr;
    }

    ~Tensor() {
        if (need_alloc && buf)
            free_mem();
        if (owns_memref)
            free(memRefDesc);
    }

    // Load tensor from file
    void load_from_file(std::string path) {
        FILE *fp = fopen(path.c_str(), "rb");
        if (fp == NULL) {
            fprintf(stderr, "[tensor.h] Error opening file %s: %s\n", path.c_str(), std::strerror(errno));
            exit(EXIT_FAILURE);
        }

        int64_t elems = _size;
        const size_t elems_read = fread(buf, sizeof(T), elems, fp);
        if (elems_read != elems) {
            // error handling
            if (feof(fp))
                fprintf(stderr, "[tensor.h] Error reading file %s: unexpected end of file\n", path.c_str());
            else if (ferror(fp))
                fprintf(stderr, "[tensor.h] Could not read %ld elems from file %s", elems, path.c_str());
            exit(EXIT_FAILURE);
        }
        fclose(fp);
    }

    // Dump tensor to a file
    // Output file format:
    //   4 byte for tensor datatype +
    //   8 bytes for tensor rank +
    //   8 * tensor_rank bytes for tensor shape +
    //   tensor_size * size(tensor_data_type) bytes for tensor data
    void dump_to_file(std::string path) {
        FILE *fp = fopen(path.c_str(), "wb");
        if (fp == NULL) {
            fprintf(stderr, "[tensor.h] Error opening file %s: %s\n", path.c_str(), std::strerror(errno));
            exit(EXIT_FAILURE);
        }

        int32_t dtype = get_dtype();
        int64_t elems = _size;
        int64_t *tensor_shape = _shape.sizes;
        int64_t tensor_rank = N;

        fwrite(&dtype, sizeof(int32_t), 1, fp);
        fwrite(&tensor_rank, sizeof(int64_t), 1, fp);
        fwrite(tensor_shape, sizeof(int64_t), tensor_rank, fp);
        const size_t elems_written = fwrite(buf, sizeof(T), elems, fp);
        if (elems_written != elems) {
            // error handling
            if (ferror(fp))
                fprintf(stderr, "[tensor.h] Could not write %ld elems to file %s", elems, path.c_str());
            exit(EXIT_FAILURE);
        }
        fclose(fp);
    }
};

#endif  //TENSOR_H
