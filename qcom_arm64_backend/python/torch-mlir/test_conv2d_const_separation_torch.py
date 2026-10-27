# ===- test_conv2d_const_separation_torch.py --------------------------------===
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
class Conv2d(torch.nn.Module):
    def __init__(self):
        super(Conv2d, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


def test_conv2d_torch(options={}):
    # Instantiate the model and run the forward pass
    model = Conv2d()
    x = torch.randn(1, 3, 64, 64)

    manager = utils.ModelManager(model, x)
    manager.create_linalg_module()
    manager.write_bytecode_to_file()
    # Set rtol, atol as needed.
    # Pass dump_outputs=True to print the outputs.
    manager.execute_and_compare(rtol=1e-05, atol=1e-06, separate_constants=True, options=options)



if __name__ == "__main__":
    test_conv2d_torch(utils.get_vector_ext_options_from_args())
