get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_executable(arm64-mlir-opt arm64-mlir-opt.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(arm64-mlir-opt)
target_link_libraries(arm64-mlir-opt PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}

    LLVMCore
    LLVMSupport
    LLVMOption

    # MLIR core
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRRegisterAllExtensions

    TritonARM64Common
    OptimizeTritonShared
    LowerTritonLinalgToLLVM
    TritonIR
)

mlir_check_all_link_libraries(arm64-mlir-opt)
