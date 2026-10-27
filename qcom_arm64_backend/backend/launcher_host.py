# ===- launcher_host.py -----------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

def generate_torch_mlir_host_launcher(input_profs, return_types, func_name, is_streaming=False):
    profiled_inputs, _ = input_profs

    # Build the packed output struct type
    if len(return_types) == 1:
        r = return_types[0]
        packed_output_type = f"StridedMemRefType<{r.dtype}, {r.rank}>"
        packed_output_struct_def = ""
    else:
        fields = " ".join(
            f"StridedMemRefType<{r.dtype}, {r.rank}> f{r.idx};"
            for r in return_types
        )
        packed_output_type = "PackedOutputs"
        packed_output_struct_def = f"struct PackedOutputs {{ {fields} }};"

    decl_args = [f"{packed_output_type}*"]
    for p in profiled_inputs:
        if p.input_type == 'tensor':
            decl_args.append(f"StridedMemRefType<{p.dtype}, {p.rank}>*")

    launch_void_args = []
    memref_constructions = []

    for r in return_types:
        vname = f"out{r.idx}"
        sizes_str, strides_str = r.shape
        launch_void_args.append(f"void* {vname}_ptr")
        memref_constructions.append(
            f"StridedMemRefType<{r.dtype}, {r.rank}> {vname}{{({r.dtype}*){vname}_ptr, ({r.dtype}*){vname}_ptr, 0, {sizes_str}, {strides_str}}};"
        )

    for p in profiled_inputs:
        if p.input_type != 'tensor':
            continue
        vname = f"in{p.idx}"
        sizes_str, strides_str = p.shape
        launch_void_args.append(f"void* {vname}_ptr")
        memref_constructions.append(
            f"StridedMemRefType<{p.dtype}, {p.rank}> {vname}{{({p.dtype}*){vname}_ptr, ({p.dtype}*){vname}_ptr, 0, {sizes_str}, {strides_str}}};"
        )

    # Build the packed output struct and the ciface call args.
    if len(return_types) == 1:
        r = return_types[0]
        packed_output_init = f"StridedMemRefType<{r.dtype}, {r.rank}> packed_output = out0;"
    else:
        field_inits = ", ".join(f"out{r.idx}" for r in return_types)
        packed_output_init = f"PackedOutputs packed_output = {{ {field_inits} }};"

    ciface_call_args = ["&packed_output"] + [f"&in{p.idx}" for p in profiled_inputs if p.input_type == 'tensor']

    # After the ciface call, copy each output field back into the pre-allocated tensors.
    if len(return_types) == 1:
        r = return_types[0]
        copy_back = (
            f"memcpy(out0_ptr, packed_output.data, sizeof({r.dtype}) * {'*'.join(str(s) for s in r.sizes)});"
            f" free(packed_output.basePtr);"
        )
    else:
        copy_back = " ".join(
            f"memcpy(out{r.idx}_ptr, packed_output.f{r.idx}.data, sizeof({r.dtype}) * {'*'.join(str(s) for s in r.sizes)});"
            f" free(packed_output.f{r.idx}.basePtr);"
            for r in return_types
        )

    n_tensor_args = len(return_types) + sum(1 for p in profiled_inputs if p.input_type == 'tensor')
    format_str = "iiiOOOO" + "O" * n_tensor_args
    pyarg_list = ", ".join(f"&_arg{i}" for i in range(n_tensor_args))
    arg_decls  = " ".join(f"PyObject* _arg{i};" for i in range(n_tensor_args))
    ptr_checks = " ".join(
        f"DevicePtrInfo ptr{i} = getPointer(_arg{i}, {i}); if (!ptr{i}.valid) return NULL;"
        for i in range(n_tensor_args)
    )
    launch_call = f"_launch(gridX, gridY, gridZ, {', '.join(f'ptr{i}.dev_ptr' for i in range(n_tensor_args))});"

    streaming_attr = ' __attribute__((arm_streaming))' if is_streaming else ''
    launch_fn_attr = ' __attribute__((arm_locally_streaming, noinline))' if is_streaming else ''

    return f"""
#include <assert.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <Python.h>
#include "ExecutionEngine/CRunnerUtils.h"
#include "ExecutionEngine/CRunnerUtils.cpp"

{packed_output_struct_def}
extern "C" {{
  void _mlir_ciface_{func_name}({", ".join(decl_args)}){streaming_attr};
}}

static void{launch_fn_attr} _launch(int gridX, int gridY, int gridZ, {", ".join(launch_void_args)}) {{
  {"  ".join(memref_constructions)}
  {packed_output_init}
  _mlir_ciface_{func_name}({", ".join(ciface_call_args)});
  // _mlir_ciface_ stores results into packed_output. Copy each output field
  // into the pre-allocated output tensors and free the kernel's buffers.
  {copy_back}
}}

// --- everything below is identical to _generate_launcher in triton-shared/backend/driver.py ---
typedef struct _DevicePtrInfo {{ void *dev_ptr; bool valid; }} DevicePtrInfo;
static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info; ptr_info.dev_ptr = 0; ptr_info.valid = true;
  if (PyLong_Check(obj)) {{ ptr_info.dev_ptr = reinterpret_cast<void*>(PyLong_AsUnsignedLongLong(obj)); return ptr_info; }}
  if (obj == Py_None) {{ return ptr_info; }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if (ptr) {{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple); Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{ PyErr_SetString(PyExc_TypeError, "data_ptr must return int"); ptr_info.valid = false; return ptr_info; }}
    ptr_info.dev_ptr = reinterpret_cast<void*>(PyLong_AsUnsignedLongLong(ret));
    Py_DECREF(ret); return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "arg must be uint64 or have data_ptr"); return ptr_info;
}}
static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook, *launch_exit_hook, *kernel_metadata, *launch_metadata;
  {arg_decls}
  if (!PyArg_ParseTuple(args, "{format_str}", &gridX, &gridY, &gridZ,
                        &kernel_metadata, &launch_metadata, &launch_enter_hook, &launch_exit_hook, {pyarg_list}))
    return NULL;
  {ptr_checks}
  {launch_call}
  if (PyErr_Occurred()) return NULL;
  Py_INCREF(Py_None); return Py_None;
}}
static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "torch-mlir kernel launcher"}}, {{NULL, NULL, 0, NULL}}
}};
static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT, "__triton_shared_ref_cpu_kernel_launcher", NULL, -1, ModuleMethods
}};
PyMODINIT_FUNC PyInit___triton_shared_ref_cpu_kernel_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) return NULL;
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
