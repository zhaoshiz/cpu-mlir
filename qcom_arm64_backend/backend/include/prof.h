//===- prof.h -------------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <string>

extern "C" void enable_perf_counter_reads();
extern "C" bool write_prof_output(const std::string &out_json);
