# ===- lit.cfg.py -----------------------------------------------------------===
# 
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  SPDX-License-Identifier: BSD-3-Clause.
#  For more license information:
#    https://github.com/qualcomm/cpu-mlir/LICENSE.txt
# 
# ===------------------------------------------------------------------------===

# -*- Python -*-

import os

import lit.formats
import lit.util
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# Configuration file for the 'lit' test runner

# name: The name of this test suite
config.name = "TRITON-ARM64"

config.test_format = lit.formats.ShTest(False)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# excludes: A list of directories to exclude from the testsuite.
config.excludes = [
    "CMakeLists.txt",
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.arm64_obj_root, "test")
config.triton_tools_dir = os.path.join(config.arm64_obj_root, "bin")
tool_dirs = [config.triton_tools_dir, config.llvm_tools_dir]

# Tweak the PATH to include the tools dir.
for d in tool_dirs:
    llvm_config.with_environment("PATH", d, append_path=True)
