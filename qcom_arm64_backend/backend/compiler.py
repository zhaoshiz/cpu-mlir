# ===- compiler.py ----------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import os
import subprocess
import functools
from pathlib import Path

from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, llvm, qcom_arm64_backend

from triton.backends.qcom_arm64_backend.utils import get_work_dir, get_kernel_name, get_codegen_options, get_verbose_level, get_debug_mode

def _get_triton_shared_opt_path() -> str:
    path = os.getenv("TRITON_SHARED_OPT_PATH", "")
    if path == "":
        raise Exception("TRITON_SHARED_OPT_PATH is not set.")
    return path

def _get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        raise Exception("LLVM_BINARY_DIR is not set.")
    return os.path.join(path, bin_name)

def _ttir_to_ttsharedir(mod):
    # Get Triton-MLIR as string
    ttir_code = str(mod)
    kernel_name = get_kernel_name(ttir_code)
    ir_dir = get_work_dir()
    src_path = os.path.join(ir_dir, f"{kernel_name}-tt.mlir")
    dst_path = os.path.join(ir_dir, f"{kernel_name}-ttshared.mlir")
    Path(src_path).write_text(ttir_code)
    triton_shared_opt_path = _get_triton_shared_opt_path()
    cmd = [triton_shared_opt_path, src_path, "--triton-to-linalg-experimental", "-o", dst_path]
    if get_debug_mode():
        # required to preserve debug info in the output mlir text
        cmd.append("--mlir-print-debuginfo")
    subprocess.check_call(cmd)
    return Path(dst_path).read_text()

def _optimize_ttsharedir(ttsharedir: str, metadata, options):
    kernel_name = get_kernel_name(ttsharedir)
    context = ir.context()
    qcom_arm64_backend.load_dialects(context)
    optimized = qcom_arm64_backend.optimize_triton_shared_mlir(ttsharedir, context, options.vector_ext,
        get_verbose_level(), get_debug_mode())
    return optimized 

def _ttsharedir_to_llir(ttsharedir: str, options):
    kernel_name = get_kernel_name(ttsharedir)
    context = ir.context()
    qcom_arm64_backend.load_dialects(context)
    llmlir = qcom_arm64_backend.triton_linalg_to_llvm_mlir(ttsharedir, context, get_codegen_options(options),
        get_verbose_level(), get_debug_mode())
    ir_dir = get_work_dir()
    llmlir_path = os.path.join(ir_dir, f"{kernel_name}-llvm-dialect.mlir")
    Path(llmlir_path).write_text(llmlir)

    # LLVM-MLIR to LLVM-IR
    mlir_translate_path = _get_llvm_bin_path("mlir-translate")
    llir_path = os.path.join(ir_dir, f"{kernel_name}-llvm-ir.ll")
    subprocess.check_call([mlir_translate_path, llmlir_path,
        "--mlir-to-llvmir",
        "-o",
        llir_path])
    return Path(llir_path).read_text()

def _optimize_llir(llir: str):
    ir_dir = get_work_dir()
    name = get_kernel_name(llir)
    src_path = os.path.join(ir_dir, name + "-llvm-ir.ll")
    Path(src_path).write_text(llir)
    dst_path = os.path.join(ir_dir, name + "-opt.ll")
    opt_path = _get_llvm_bin_path("opt")
    subprocess.check_call([opt_path, src_path, "--mattr=+sve,+sve2,+sme", "-passes=default<O3>", "-S", "-o", dst_path])
    return Path(dst_path).read_text()


def _llir_to_bin(llir: str, metadata):
    name = get_kernel_name(llir)
    metadata["name"] = name
    ir_dir = get_work_dir()
    src_path = os.path.join(ir_dir, name + "-pre-llc.ll")
    dst_path = os.path.join(ir_dir, name + "-arm64.S")
    Path(src_path).write_text(llir)
    llc_path = _get_llvm_bin_path("llc")
    subprocess.check_call([llc_path, src_path, "-mtriple=aarch64", "-mattr=+v8.5a,+sve2,+sme", "-o", dst_path])
    # Actually it's text-format assembly. Use read_text().
    asm = Path(dst_path).read_text()
    return asm 



@dataclass(frozen=True)
class ARM64CPUOptions:
    debug: bool = False
    arch: str = None
    arch_triple: str = "armv8-a"
    arch_features: str = "+sve,+sve2,+sme"
    vectorize: int = 1
    vector_ext: str = "sme"
    base_vector_length: int = 32
    scalable_vector_length: int = 128
    num_threads: int = 4
    enable_fp_fusion: bool = False
    extern_libs = None
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    sanitize_overflow: bool = True
    instrumentation_mode: str = ""
    #data_layout: str = (...)
    # TODO: Next 7 options not used currently but kept to be consistent with other backends, remove later if not required
    num_warps: int = 0
    num_ctas: int = 0
    num_stages: int = 1
    shared: bool = False
    enable_warp_specialization: bool = False
    cluster_dims: tuple = (1, 1, 1)
    fpsan_homomorphic_casts: bool = False
    # TODO: Currently set name here temporarily. Need to set correctly to the name of the compiled kernel
    # when driver code is completed.
    name: str = "arm64"

    # Loops are instrumented by default, setting this to True disables this behavior
    prof_disable_loop: bool = False
    # By default, only outer loop of nested loops will be instrumented
    # Increasing this value will create more profiling overhead, lowering precision
    prof_loop_depth: int = 1

    def __post_init__(self):
        pass

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


class CPUBackend(BaseBackend):
    binary_ext = 'cpuasm'

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'arm64'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    def parse_options(self, opts) -> Any:
        args = {'vector_ext': self.target.arch}
        args.update({k: opts[k] for k in ARM64CPUOptions.__dataclass_fields__.keys() if k in opts})
        return ARM64CPUOptions(**args)

    def get_codegen_implementation(self, options):
        # We don't have limitations for minimum dot size for our architecture, so we just
        # add a placeholder with the minimum size values.
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        # Note: We actually don't need any of these except for the name which is
        # used in the launch function in driver.py. Putting these in so we're
        # consistent with other backends
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
            metadata.name
        )

    # Our compilation pipeline isn't in python like nvidia or amd, no need to load
    # dialects. See `triton_shared.cc`
    def load_dialects(self, ctx):
        return

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        return mod

    def add_stages(self, stages, options, language):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttsharedir"] = lambda src, metadata: _optimize_ttsharedir(_ttir_to_ttsharedir(src), metadata, options)
        stages["llir"] = lambda src, metadata: _optimize_llir(_ttsharedir_to_llir(src, options))
        stages["cpuasm"] = lambda src, metadata: _llir_to_bin(src, metadata)


    @functools.lru_cache()
    def hash(self):
        return self.target

    # No additional python modules used, hence returning an empty dictionary
    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}
