list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(ARM64_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(ARM64_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files
include_directories(${Python3_INCLUDE_DIR})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(bin)
add_subdirectory(test)
#add_subdirectory(tools)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

message(STATUS "Adding Qualcomm ARM64 Backend")

add_library(TritonARM64Common STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common.cc
)

add_triton_plugin(TritonARM64
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_qcom_arm64_backend.cc

    LINK_LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    TritonARM64Common
    OptimizeTritonShared
    LowerTritonLinalgToLLVM

    MLIRRegisterAllDialects
    MLIRRegisterAllExtensions
)
target_link_libraries(TritonARM64 PRIVATE Python3::Module nanobind-static)

#Add a symlink qcom_arm64_backend/python/examples/triton_shared_examples -> triton-shared/python/examples
cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH ROOT_DIR)
cmake_path(APPEND ROOT_DIR "triton-shared" "python" "examples" OUTPUT_VARIABLE TRITON_SHARED_EXAMPLES)
cmake_path(APPEND CMAKE_CURRENT_SOURCE_DIR "python" "examples" "triton_shared_examples" OUTPUT_VARIABLE ARM64_EXAMPLES)
add_symlink(${TRITON_SHARED_EXAMPLES} ${ARM64_EXAMPLES})
