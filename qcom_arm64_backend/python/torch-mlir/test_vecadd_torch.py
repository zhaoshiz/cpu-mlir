# ===- test_vecadd_torch.py -------------------------------------------------===
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


class VecAdd(nn.Module):
    def __init__(self):
        super(VecAdd, self).__init__()
        pass

    def forward(self, a, b):
        c = torch.add(a, b)
        return c

def test_vecadd_torch(options={}):
    model = VecAdd()

    a = torch.rand(1000, device="cpu")
    b = torch.rand(1000, device="cpu")

    manager = utils.ModelManager(model, a, b)
    manager.create_linalg_module()
    manager.write_bytecode_to_file()
    manager.execute_and_compare(options=options)

if __name__ == "__main__":
    test_vecadd_torch(utils.get_vector_ext_options_from_args())
