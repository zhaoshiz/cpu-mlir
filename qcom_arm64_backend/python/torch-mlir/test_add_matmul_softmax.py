# ===- test_add_matmul_softmax.py -------------------------------------------===
# 
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  SPDX-License-Identifier: BSD-3-Clause.
#  For more license information:
#    https://github.com/qualcomm/cpu-mlir/LICENSE.txt
# 
# ===------------------------------------------------------------------------===

import torch
import utils
import pytest
import subprocess
import os


# Define the computation graph
class AddMatmulSoftmax(torch.nn.Module):
    def __init__(self):
        super(AddMatmulSoftmax, self).__init__()

    def forward(self, x, y, z):
        add_result = x + y
        matmul_result = torch.matmul(add_result, z)
        softmax_result = torch.nn.functional.softmax(matmul_result, dim=-1)
        return softmax_result


def test_add_matmul_softmax_torch(options={}):
    # Create dummy 2D tensors
    matrix_size = 32
    shape = (matrix_size, matrix_size)

    # Instantiate the model and run the forward pass
    model = AddMatmulSoftmax()
    x = torch.randn(shape)
    y = torch.randn(shape)
    z = torch.randn(shape)

    manager = utils.ModelManager(model, x, y, z)
    manager.create_linalg_module()
    manager.write_bytecode_to_file()
    # Set rtol, atol as needed.
    # Pass dump_outputs=True to print the outputs.
    manager.execute_and_compare(rtol=1e-05, atol=1e-06, options=options)



if __name__ == "__main__":
    test_add_matmul_softmax_torch(utils.get_vector_ext_options_from_args())
