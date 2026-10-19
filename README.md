# cpu-mlir
A MLIR-based compilation stack targeting CPUs for high performance AI/ML kernel and models.
(We only support ARM64 ISA at the moment, we may add RISCV-64 support in the future.)

test commit

Currently, this project supports lowering Triton kernels through Triton-Shared and PyTorch models through Torch-MLIR, ultimately generating ARM64 code in both cases. The resulting code can be launched on an android device via a standalone executable or executed directly on a host ARM64 machine inside of a python environment.

## Dependencies
[LLVM](https://github.com/llvm/llvm-project)
[Triton](https://github.com/triton-lang/triton)
[Triton-Shared](https://github.com/facebookincubator/triton-shared)
Simulator: [QEMU](https://www.qemu.org/download/)
GCC Toolchain: [Arm GNU Toolchain](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads)

This project includes Triton and Triton-Shared as submodules and tries to use the latest Triton-Shared.
Triton-Shared specifies a commit hash of Triton, which in turn specifies a commit hash of LLVM: cpu-mlir -> Triton-Shared -> Triton -> LLVM.

## Note on compilers used
There're two compilation/building tasks:
1) building Triton, Triton-Shared and the ARM64 backend to run on the host machine;
2) compiling Triton kernels, generating ARM64 target code, and producing executables that run on ARM64 device or emulation.

Task (1) is native-compiling while task (2) is possibly cross-compiling. In the rest of this file, `HOST_COMPILER` refers the compiler for task (1) and `TARGET_COMPILER` refers the compiler that does (2). They can be the same LLVM build/installation. But for now we use a recent official LLVM release for `HOST_COMPILER` and a custom build of LLVM (commit hash specificed by Triton) as the `TARGET_COMPILER`.

A build recipe for `TARGET_COMPILER` and be found in `scripts/build_llvm.sh`.

## Setup
All commands assume you are in the base `.../cpu-mlir` directory.
### System Package Dependencies
```sh
sudo apt update
sudo apt install libzstd-dev
sudo apt install qemu-user
```
Note: the statically linked `qemu-user-static` also works.
### ADB
When executing workloads on android devices, this backend assumes a pipeline with two machines: (1) the machine compiling the standalone executable, and (2) the host machine the android device is connected to. Most dependencies are for (1), but `adb` needs to be installed on (2).
```sh
sudo apt install android-sdk-platform-tools
```
Note: It is best practice for all users on (2) to use the same `adb`. When `adb` is fist invoked, it establishes a server which subsequent commands use. If a different `adb` binary is then invoked, a version mismatch is detected, the initial server is destroyed, and a new server is created, potentially interupting other users who were using the first server.
### Python Dependencies
After cloning this project, use below script to check out and patch Triton and Triton-Shared:
```sh
./scripts/get_dependencies.sh
```
Setup a python virtual environment:
```sh
python3 -m venv .venv --prompt triton
source .venv/bin/activate
```
Install dependent Python modules:
```sh
cd triton
pip install --upgrade pip
pip install -r python/requirements.txt
pip install pytest numpy
pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels \
    --pre \
    torch==2.8.0+cpu \
    torchvision==0.23.0+cpu \
    torch-mlir \
    torchao==0.10.0+cpu \
    torchtune==0.6.0+cpu \
    transformers==4.52.4
cd -
```
### Arm GNU Cross Toolchain (only needed on x86 machines)
Download Arm GNU cross toolchain, if needed. If you are building on an AArch64 system, this is NOT necessary, the required libraries will already exist on your system.
To download and extract the x86_64 Linux hosted cross toolchain for AArch64:
```sh
curl -LO https://developer.arm.com/-/media/Files/downloads/gnu/15.2.rel1/binrel/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu.tar.xz
tar -xf arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu.tar.xz
export ARM64_TOOLCHAIN_DIR=$PWD/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu
```
[All availible downloads](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads) (though most users should need the x86_64 Linux hosted toolchain)
### Building Custom LLVM
A pre-built LLVM will be provided in the future, for now users need to provide their own LLVM build.
If you do not already have a host C/C++ compiler, we recommend using [the latest LLVM release](https://github.com/llvm/llvm-project/releases).
To download and checkout the correct version of the LLVM source:
```sh
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout $(cat ../triton/cmake/llvm-hash.txt)
cd -
```
In addition to LLVM, this script builds ARM64 compiler-rt and c++ libraries, which enable cross-compilation on x86 devices. This requires the Arm GNU Toolchain that was downloaded above. Similar to the toolchain, if you are building on an AArch64 system, the compiler-rt and c++ libraries will already be on the system, so this step is skipped. This means that `ARM64_TOOLCHAIN_DIR` is only required when building on x86 systems.
To build custom LLVM using helper script:
```sh
./scripts/build-llvm.sh <HOST_COMPILER> <LLVM_SOURCE_DIR> <LLVM_BUILD_DIR> <LLVM_INSTALL_DIR> [<ARM64_TOOLCHAIN_DIR>]
```

## Build
### Setup envivronment variables needed by the build process:
```sh
# Set the locations of HOST_COMPILER and TARGET_COMPILER
export HOST_COMPILER=<HOST_COMPILER_TOP>
export TARGET_COMPILER=<TARGET_COMPILER_TOP>
```
### Build Triton, Triton-Shared, and this project:
```sh
./scripts/build.sh
```

## Usage

### Environment variables for IR lowering and codegen:
```sh
source ./scripts/setup_env.sh
```

### Standalone Mode vs Host Mode

This project provides two modes for executing workloads: standalone and host mode. In standalone mode, an aarch64 standalone executable is compiled (typically on an x86 machine) for execution and launched on either an android device or a simulator. In host mode, the workload is compiled, loaded into the local python environment, and invoked. Naturally, host mode is only supported on aarch64 systems.

### Common Environment Variables
Each mode has its own unique environment variables, but there are some used by both:
```sh
# Arm GNU Toolchain
# This should already be set if you downloaded the toolchain using this README
export ARM64_TOOLCHAIN_DIR=<ARM_GNU_TOOLCHAIN_PATH>
```
```sh
# Include paths
export ARM64_BACKEND_PATH=$PWD/qcom_arm64_backend
export TRITON_SHARED_ROOT=$PWD/triton-shared

# library path
CLANG_VERSION=$($TARGET_COMPILER/bin/clang --version | grep "clang version" | sed "s/.*version //" | sed "s/\.[0-9].*//")
export LAUNCHER_LIBPATHS="$TARGET_COMPILER/lib $TARGET_COMPILER/aarch64-linux-gnu/lib $TARGET_COMPILER/lib/clang/$CLANG_VERSION/lib/aarch64-linux-gnu"
```

### Standalone Mode
Examples for how to use standalone mode can be found in `qcom_arm64_backend/python`. Triton examples can be found in `examples/triton_shared_examples` and torch-mlir examples can be found in `torch-mlir`.

#### To run on simulation by QEMU:
```sh
export RUN_ON_SIM=1
export QEMU_PATH=/path/to/qemu-aarch64
```
#### To run on android device:
```sh
export RUN_ON_SIM=0

# Staging Directory
# directory accessible by both the compilation machine and device host machine,
# used to pass standalone executable and data between systems
export TRITON_ARM64_STAGING_DIR=<STAGING_DIR_PATH>

export ADB_PATH=<HOST_ADB_PATH> # path to the adb binary on the android device host machine
export DEV_HOST=<DEVICE_HOST_NAME> # hostname of the android device host machine
export DEV_ID=<DEVICE_ID> # id of the android device
```

See qcom_arm64_backend/README.md for full details of environment variables that can be used.

### Host Mode (Build and Execution)
Host mode is controlled by the environement variable `TRITON_ARM64_HOST_BUILD` and is disabled by default. See the notes below for workload specific considerations.

To enable: `export TRITON_ARM64_HOST_BUILD=1`
To disable: `export TRITON_ARM64_HOST_BUILD=0`

#### Triton workloads

To utilize host build for triton kernels, one should use Triton-Shared's `CPUDriver` and NOT `ARM64CPUDriver`. To alter ARM64 options, provide the options when invoking the kernel. The following example shows how to change the vector extension to NEON:

```python
@triton.jit
def foo_kernel(arg0, arg1):
    ...

foo_kernel[grid](arg0, arg1, vector_ext="neon")
```

#### Torch-mlir workloads

For torch-mlir models, ARM64 options may be provided to the `execute_and_compare` function inside a dictionary. The following example shows how to change the vector extension to NEON:

```python
class foo(nn.Module):
    ...

manager = utils.ModelManager(foo())
manager.create_linalg_module()
manager.write_bytecode_to_file()
manager.execute_and_compare(options={"vector_ext" : "neon"})
```


### Run a Triton-shared example:
```sh
pytest qcom_arm64_backend/python/examples/triton_shared_examples/test_vec_add.py
```

Or use convenience script:
```sh
cd qcom_arm64_backend/python/examples
./run_pytest.sh triton_shared_examples/test_vec_add.py
```

When invoking any example kernels tests in `qcom_arm64_backend/python/examples`, Triton will compile the kernel workload and then run the the generated executable on either simulator or target device. The kernel output will be compared to the expected output to determine test success.

If kernel is compiled/executed on device the user also has the option of enabling kernel "wall-clock" benchmarking. This capability is enabled by (this example runs benchmarking over 10 iterations of the workload)

```sh
export TRITON_ARM64_BENCHMARK=1
export TRITON_ARM64_BENCHMARK_ITERATIONS=10
```

### Run full set of Triton-shared examples

To invoke the full test suite use the `qcom_arm64_backend/python/example/run_pytest.sh` script

### Dumping IR

#### Dump of Triton IR stages

Triton workloads are lowered through MLIR/LLVM IR in several stages before being compiled into assembly. These stages are: Triton dialect MLIR -> Trtion-Shared dialect MLIR -> LLVM dialect MLIR -> LLVM IR. To save these to the intermediate files in the current working directory, use the following environment variable:
```sh
export TRITON_ARM64_SAVE_TEMPS = 1
```

#### MLIR IR dump of passes

The above method dumps IR at specific domain transitions but does not allow control or dump of output IR for each compiler pass.

To enable more granular ability to view IR after each pass use the following environment variables (as illustrated in the `run_pytest.sh` script)

`MLIR_ENABLE_DUMP`:
0 - Disable MLIR passes dump for all high level TTIR passes (default)
1 - Enable MLIR passes dump for all high level TTIR passes

`MLIR_ENABLE_DUMP_TRITON_SHARED`:
0 - Disable MLIR passes dump for all Triton-shared passes (default)
1 - Enable MLIR passes dump for all Triton-shared passes

`MLIR_DUMP_TRITON_SHARED_SPECIFIC_PASS`:
(used with `MLIR_ENABLE_DUMP_TRITON_SHARED=1`)
Dump IR output from specific Triton-shared pass
E.g. `MLIR_DUMP_TRITON_SHARED_SPECIFIC_PASS="ConvertVectorToLLVMPass"`

`MLIR_DUMP_ALL_IR_DATA`:
(used with `MLIR_ENABLE_DUMP_TRITON_SHARED=1`)
0 - Omits all constant data from dump (default)
1 - Dumps all constant data as part of IR

#### Dump with custom `arm64-mlir-opt`

It is also possible to dump IR while using our standlone MLIR optimization tool `arm64-mlir-opt`. This tool can be used to run our custom optimizations and uses MLIRS's print flags. For example:
```sh
arm64-mlir-opt --mlir-print-ir-before-all ...
arm64-mlir-opt --mlir-print-ir-before=triton-arm64-linalg-vectorization ...
```

## Triton-shared passes LIT tests

To run all Triton-share LIT tests for passes use the following command:
`lit -v $TRITON_BUILD_DIR/third_party/triton_shared/test`

To run a single LIT pass test use the following command:
`lit -a --filter addptr_bitcast $TRITON_BUILD_DIR/third_party/triton_shared/test`

## ARM64 backend passes LIT tests

To run all Qualcomm ARM64 backend LIT tests use the following command:
`lit $TRITON_BUILD_DIR/third_party/qcom_arm64_backend/test/`

## Profiling

cpu-mlir has builtin functionality for profiling functions and loops inside compiled kernels. Profiling can only be enabled when running on a device and is always disabled when running on simulator.

Profiling is enabled with `TRITON_ARM64_PROFILE`:
```sh
export TRITON_ARM64_PROFILE=1
```

With profiling enabled, three files will be generated: `prof_infodump.txt`, `prof_report.json`, and `prof_output.csv`. `prof_infodump.txt` contains information collected during compilation that is used for postprocessing, `prof_report.json` is the raw profiling report, and `prof_output.csv` is a nicely-formatted report, showing the number of processor cycles spent in each function/loop. The line numbers correspond to lines in the pre-profiling-pass MLIR, and the id's can be used to track functions/loops in post-profiling-pass IR.

Note: by default, `prof_output.csv` is automatically generated, but it can also be generated manually using `process_prof.py`
```sh
python qcom_arm64_backend/backend/process_prof.py path/to/prof_report.json path/to/prof_infodump.txt path/to/desired/output.csv
```

## License

Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.

