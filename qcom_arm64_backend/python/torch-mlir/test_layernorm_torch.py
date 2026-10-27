# ===- test_layernorm_torch.py ----------------------------------------------===
# 
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  SPDX-License-Identifier: BSD-3-Clause.
#  For more license information:
#    https://github.com/qualcomm/cpu-mlir/LICENSE.txt
# 
# ===------------------------------------------------------------------------===

import torch
import torch.nn as nn
import utils
import pytest

# test_layernorm_torch.py
# This test creates a graph containing a single node - a LayerNorm op
# compiles it and runs for ARM64  using our mlir-based compiler,
# and compares the results obtained with Torch on x86


class LayerNorm(nn.Module):
    def __init__(self):
        super(LayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(normalized_shape=128)

    def forward(self, x):
        x = self.layernorm(x)
        return x

def test_layernorm_torch(options={}):
    model = LayerNorm()
    inp = torch.rand(128, 128, dtype=torch.float16)

    manager = utils.ModelManager(model, inp)
    manager.create_linalg_module()
    manager.write_bytecode_to_file()
    # Set rtol, atol as needed.
    # Pass dump_outputs=True to print the outputs.
    print("Input was: ", inp)
    # Note: we have quite poor numerical accuracy on layernorm at the moment
    manager.execute_and_compare(rtol=5e-01, atol=1e-02, options=options)


if __name__ == "__main__":
    test_layernorm_torch(utils.get_vector_ext_options_from_args())
