# ===- resnet18model.py -----------------------------------------------------===
# 
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  SPDX-License-Identifier: BSD-3-Clause.
#  For more license information:
#    https://github.com/qualcomm/cpu-mlir/LICENSE.txt
# 
# ===------------------------------------------------------------------------===

import sys, os
import torch
import torchvision.models as models
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType
from triton.backends.qcom_arm64_backend.torch_mlir_launcher import TorchMLIRLauncher
from triton.backends.qcom_arm64_backend.utils import get_work_dir

VECTOR_EXT_OPTIONS = {
    "neon": {"vector_ext": "neon", "arch_features": "+neon"},
    "sve":  {"vector_ext": "sve",  "arch_features": "+sve,+sve2"},
    "sme":  {"vector_ext": "sme",  "arch_features": "+sve,+sve2,+sme"},
}


def compile_to_linalg(model, input, dump_to_file=None, debug=False):
    linalg = fx.export_and_import(
        model,
        input,
        output_type=OutputType.LINALG_ON_TENSORS,
        func_name=model.__class__.__name__,
        enable_graph_printing=debug,
        enable_ir_printing=debug,
    )

    if dump_to_file:
        with open(dump_to_file, "w") as f:
            f.write(str(linalg))

    return linalg


def arm64_execution(module, func_name, inputs, options={}):
    linalg_filename = os.path.join(get_work_dir(), f"{func_name}.mlirbc")

    bytecode = module.operation.get_asm(binary=True)
    with open(linalg_filename, "wb") as f:
        f.write(bytecode)

    arm64_outputs = TorchMLIRLauncher().run_torch_mlir(
        str(linalg_filename), inputs, func_name, options
    )
    return arm64_outputs


def host_execution(model, input):
    with torch.no_grad():
        return model(input)


def compare(arm64_output, host_output, atol=1e-2, fail_on_mismatch=False):
    match = torch.allclose(arm64_output, host_output, atol=atol)
    print("Match: ", match)
    if not match:
        max_diff = (arm64_output - host_output).abs().max().item()
        print(f"Max absolute difference: {max_diff:.6f}")
        assert not fail_on_mismatch, \
            "Correctness issue: ARM64 and x86 outputs do not match"


def resnet18(options={}):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    func_name = model.__class__.__name__

    input = torch.randn(1, 3, 224, 224)

    module = compile_to_linalg(model, input)

    inputs = [input]

    print(f"Number of inputs: {len(inputs)}")
    #print(f"  Input shapes: {[inp.shape for inp in inputs]}")

    arm64_outputs = arm64_execution(module, func_name, inputs, options)

    host_output = host_execution(model, input)

    compare(arm64_outputs[0], host_output, fail_on_mismatch=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sme"
    if mode not in VECTOR_EXT_OPTIONS:
        print(f"Usage: {sys.argv[0]} [neon|sve|sme]")
        sys.exit(1)
    resnet18(VECTOR_EXT_OPTIONS[mode])
