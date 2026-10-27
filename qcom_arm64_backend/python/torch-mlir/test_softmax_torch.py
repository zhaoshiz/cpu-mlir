# ===- test_softmax_torch.py ------------------------------------------------===
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


class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.softmax(x)
        return x

def test_softmax_torch(options={}):
    model = Softmax()
    inp = torch.rand(128, 128, device="cpu")

    manager = utils.ModelManager(model, inp)
    manager.create_linalg_module()
    manager.write_bytecode_to_file()
    manager.execute_and_compare(options=options)

if __name__ == "__main__":
    test_softmax_torch(utils.get_vector_ext_options_from_args())
