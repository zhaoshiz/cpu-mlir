## Environmental variables related to compiling/building executable

`ARM64_TOOLCHAIN_DIR` : path to top dir of arm64 c++ toolchain (only needed for device standalone executable build)  
`TRITON_TEST_ROOT` : path to top dir of triton_test  
`TRITON_SHARED_ROOT` :  path to top dir of triton_shared  
`LLVM_BINARY_DIR` : path to dir containing LLVM binaries like clang++  
`LAUNCHER_LIBPATHS` : path to dirs containing necessary libs to build the launcher (i.e. libstdc++, libunwind, compiler-rt)  
`TRITON_ARM64_DO_SME` : 1 to generate SME, 0 to generate SVE (default)  
`TRITON_ARM64_STAGING_DIR` : path to desired stanging directory (default `/tmp/triton`)

## Environmental variables related to running standalone executable

`RUN_ON_SIM`:  
1 for using QEMU  
0 for running on device (default)

Required if using in QEMU:  
`QEMU_PATH` : path to binary 'qemu-aarch64'

Required if running on device:  
`ADB_PATH` : path to adb executable  
`DEV_HOST` : remote linux-x86 host the device is attached to  
`DEV_ID` : device id to be used with adb -s

## Optional environmental variables for debugging

`TRITON_ARM64_SAVE_TEMPS`:  
1 to dump mlir, llvm-ir, assembly, and c++ launcher in current directory, 0 is default

`TRITON_ARM64_VERBOSE`:  
0 - no output (default)  
1 - print progress on compiling and running  
2 - print progress and command on compiling and running  
3 and above, currently same to 2

TRITON_ARM64_DEBUG:
0 - no debugging info compiled into executable (default) 
1 - compile with debug info

## Environment variables for benchmarking 

`TRITON_ARM64_BENCHMARK`:  
0 - Disable benchmarking (default)  
1 - Enable benchmarking

`TRITON_ARM64_BENCHMARK_ITERATIONS` :  
Set to the number of iterations the workload should run.

Note that the benchmarking settings are only used for the case of running on target device (RUN_ON_SIM=0)
