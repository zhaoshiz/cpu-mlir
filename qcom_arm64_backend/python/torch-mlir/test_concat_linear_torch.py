# ===- test_concat_linear_torch.py ------------------------------------------===
# 
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  SPDX-License-Identifier: BSD-3-Clause.
#  For more license information:
#    https://github.com/qualcomm/cpu-mlir/LICENSE.txt
# 
# ===------------------------------------------------------------------------===

import torch
import torch.nn as nn
import random
import numpy as np
import utils
import pytest
import subprocess
import os


class ConcatLinear(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim1, output_dim2):
        super(ConcatLinear, self).__init__()
        self.linear1 = nn.Linear(input_dim1 + input_dim2, output_dim1)
        self.linear2 = nn.Linear(input_dim1 + input_dim2, output_dim2)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        out1 = self.linear1(x)
        out2 = self.linear2(x)
        return out1, out2


def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def test_concat_linear(options={}):
    input_dim1 = 3
    input_dim2 = 3
    output_dim1 = 2
    output_dim2 = 2

    model = ConcatLinear(input_dim1, input_dim2, output_dim1, output_dim2)
    tensor1 = torch.randn(5, input_dim1)  # Batch of 5 samples, each with 3 features
    tensor2 = torch.randn(5, input_dim2)  # Batch of 5 samples, each with 3 features

    manager = utils.ModelManager(model, tensor1, tensor2)
    manager.create_linalg_module()
    manager.write_bytecode_to_file()
    # Set rtol, atol as needed.
    manager.execute_and_compare(options=options)


if __name__ == "__main__":
    test_concat_linear(utils.get_vector_ext_options_from_args())
