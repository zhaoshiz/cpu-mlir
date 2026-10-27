//===- prof.cpp ------------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include <asm/unistd.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <linux/perf_event.h>
#include <sstream>
#include <stdint.h>
#include <sys/ioctl.h>
#include <unistd.h>

// Number of unique ids allowed
#define PROF_COUNTER_SIZE 1000
// Number of entries allowed per id
#define NUM_PROF_ENTRIES 200
// Total buffer size. Each "entry" takes up 3 elements in the array
#define PROF_BUFFER_SIZE (PROF_COUNTER_SIZE * NUM_PROF_ENTRIES * 3)

// File descriptor for hardware event
int perf_fd;
// Buffer of counters for number of entries per id
uint32_t prof_counter[PROF_COUNTER_SIZE] = {0};
// Buffer for profiling entries
uint32_t prof_buffer[PROF_BUFFER_SIZE];
// Current position in prof_buffer
uint32_t prof_buffer_count = 0;

// In order to enable profiling, a few steps must first be taken. First, a
// perf_event_attr is configured to represent the performance cycle counter.
// The perf_event_open syscall generates a file descriptor for this event,
// which can later be read from
extern "C" void enable_perf_counter_reads() {
  // configure performance event that represents hardware counter
  struct perf_event_attr pe;
  memset(&pe, 0, sizeof(struct perf_event_attr));
  pe.type = PERF_TYPE_HARDWARE;
  pe.size = sizeof(struct perf_event_attr);
  pe.config = PERF_COUNT_HW_CPU_CYCLES;
  pe.disabled = 1;
  pe.exclude_kernel = 1;
  pe.exclude_hv = 1;

  // Get file descriptor for harware counter
  perf_fd = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
  if (perf_fd == -1) {
    std::perror("Error generating profiling file descriptor with perf_event_open\n");
    return;
  }

  // Restart and enable hardware counter
  ioctl(perf_fd, PERF_EVENT_IOC_RESET, 0);
  ioctl(perf_fd, PERF_EVENT_IOC_ENABLE, 0);
}

// Definition for function calls added by ARM64ProfInstrumentationPass
// Reads from performance counter file descriptor and makes an entry in the
// profile buffer
extern "C" void arm64_prof_func(int id) {
  // Verify id does not already have max number of entries and ensure that
  // prof_buffer is not full
  if (prof_counter[id] >= NUM_PROF_ENTRIES || prof_buffer_count >= PROF_BUFFER_SIZE) {
      return;
  }

  uint64_t value;
  if (read(perf_fd, &value, sizeof(uint64_t)) == -1) {
    std::perror("Error reading from profiling file descriptor\n");
    return;
  }

  prof_buffer[prof_buffer_count] = id;
  // Since this is ARM64, the counter value is 64-bit. However, for the
  // function/loop id, a 32-bit type is more than sufficient. In order to
  // save space, the 64-bit counter value is split into its halves and saved
  // to the buffer so that each "entry" can be just 96 bits (3, 32-bit values)
  // instead of 128 bits (2, 64-bit values).
  prof_buffer[prof_buffer_count + 1] = static_cast<uint32_t>(value & 0xFFFFFFFF);
  prof_buffer[prof_buffer_count + 2] = static_cast<uint32_t>(value >> 32);
  prof_counter[id]++;
  prof_buffer_count += 3;
}

// Write profiling data to a JSON file
extern "C" bool write_prof_output(const std::string &out_json) {
  std::ostringstream s;
  s << "{\n  \"entries\": [\n";

  for (size_t i = 0; i < prof_buffer_count; i += 3) {
    uint64_t cycles =
        (static_cast<uint64_t>(prof_buffer[i + 2]) << 32) | prof_buffer[i + 1];
    s << "    {\n"
      << "      \"id\": " << prof_buffer[i] << ",\n"
      << "      \"cyc\": " << cycles << "\n"
      << "    }";
    if (i + 3 < prof_buffer_count)
      s << ",";
    s << "\n";
  }

  s << "  ]\n}\n";

  std::ofstream out(out_json);
  if (!out.is_open()) {
    std::perror("Error opening prof_report.json\n");
    return false;
  }
  out << s.str();
  return out.good();
}
