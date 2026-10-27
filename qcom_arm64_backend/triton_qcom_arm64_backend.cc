//===- triton_qcom_arm64_backend.cc ---------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "Dialect/TmTensor/IR/TmTensorDialect.h"
#include "LowerTritonLinalgToLLVM/LowerTritonLinalgToLLVM.h"
#include "OptimizeTritonShared/OptimizeTritonShared.h"

#include "triton/Dialect/Triton/IR/Dialect.h" // added this for the loadDialect

#include "mlir/Conversion/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <unordered_map>

namespace nb = nanobind;

mlir::ModuleOp parse_mlir_module(std::string &module_str,
                                 mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> module_op =
      mlir::parseSourceString<mlir::ModuleOp>(module_str, &context);
  if (!module_op)
    throw std::runtime_error("ARM64 Parse MLIR string failed.");

  return module_op->clone();
}

mlir::ModuleOp parse_mlir_from_file(const std::string &path,
                                 mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(path, &context);
  if (!module)
    throw std::runtime_error("ARM64 Parse MLIR file failed.");
  return module->clone();
}

// Print a module to string with optional debug info (locations) enabled, matching the
// format produced by Python's str(module) via getOpPrintingFlags() in ir.cc.
static std::string moduleToStringWithDebugInfo(mlir::ModuleOp mod, int debug_mode) {
  std::string str;
  llvm::raw_string_ostream os(str);
  mlir::OpPrintingFlags flags;
  if(debug_mode) flags.enableDebugInfo();
  mod.print(os, flags);
  os.flush();
  return str;
}

void print_loaded_dialects(mlir::MLIRContext &context) {
  llvm::dbgs() << "Loaded dialects in context:\n";
  for (mlir::Dialect *d : context.getLoadedDialects()) {
    llvm::dbgs() << "  - " << d->getNamespace().str() << "\n";
  }
}

// Extract the return types (rank, dtype, and shape) of a function given its name.
std::vector<std::tuple<int, std::string, std::vector<int64_t>>>
get_return_list(mlir::MLIRContext &context, mlir::ModuleOp module_op, const std::string &fName) {
  std::vector<std::tuple<int, std::string, std::vector<int64_t>>> resultList;

  for (auto funcOp : module_op.getOps<mlir::func::FuncOp>()) {
    if (funcOp.getName() != fName)
      continue;
    for (auto resultType : funcOp.getResultTypes()) {
      int rank;
      mlir::Type dtype;
      std::vector<int64_t> shape;

      if (auto tensorType =
              mlir::dyn_cast<mlir::RankedTensorType>(resultType)) {
        rank = tensorType.getRank();
        dtype = tensorType.getElementType();
        shape = std::vector<int64_t>(tensorType.getShape().begin(), tensorType.getShape().end());
      } else {
        // Assuming if it's not RankedTensor, it's a scalar value.
        rank = 0;
        dtype = resultType;
        shape = {};
      }

      std::string typebuf;
      llvm::raw_string_ostream sdtype(typebuf);
      dtype.print(sdtype);

      resultList.push_back(std::make_tuple(rank, sdtype.str(), shape));
    }
  }
  // Return the list of return types.
  return resultList;
}

void init_triton_arm64_optimization(nb::module_ &m) {
  using ret = nb::rv_policy;

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);   // TODO: restrict
    mlir::registerAllExtensions(registry); // TODO: restrict

    registry.insert<mlir::tm_tensor::TmTensorDialect>();

    context.appendDialectRegistry(registry);
    context.loadDialect<mlir::triton::TritonDialect>();
    context.loadAllAvailableDialects();
    // print_loaded_dialects(context);
  });

  m.def(
      "optimize_triton_shared_mlir",
      [](std::string ttlinalg, mlir::MLIRContext &context, std::string vec_ext, int verbose_level, int debug_mode) {
        mlir::ModuleOp ttshared_module = parse_mlir_module(ttlinalg, context);

        auto optimized = ::mlir::triton::arm64OptimizeTritonShared(ttshared_module,
			     getVectorExtFromString(vec_ext), verbose_level, debug_mode);
        if (!optimized)
          throw std::runtime_error(
              "Failed to optimize Triton Shared.");

        return moduleToStringWithDebugInfo(ttshared_module, debug_mode);
      },
      ret::take_ownership);

  m.def(
      "parse_mlir_module_from_file",
      [](std::string &module_str, mlir::MLIRContext &context) {
        return parse_mlir_from_file(module_str, context);
      },
      nb::rv_policy::move);

  m.def("get_return_list", [](mlir::MLIRContext &context, mlir::ModuleOp &module_op, std::string fName) {
        return get_return_list(context, module_op, fName);
      });
}

void init_triton_linalg_to_mlir(nb::module_ &m) {
  using ret = nb::rv_policy;

  m.def(
      "triton_linalg_to_llvm_mlir",
      [](std::string ttlinalg, mlir::MLIRContext &context, nb::dict codegen_options, int verbose_level, int debug_mode) {
        mlir::ModuleOp ttshared_module = parse_mlir_module(ttlinalg, context);

        arm64VectorExtension vector_ext;
        std::unordered_map<std::string, std::string> prof_options_map;
        for (auto it : codegen_options) {
          if (nb::cast<std::string>(it.first) == "arm64_vector_ext")
            vector_ext=getVectorExtFromString(nb::cast<std::string>(it.second));
          else
            prof_options_map[nb::cast<std::string>(it.first)] = nb::cast<std::string>(it.second);
        }

        auto llvm_mlir = ::mlir::triton::arm64TritonLinalgToLLVMMLIR(
           ttshared_module, vector_ext, prof_options_map, verbose_level, debug_mode);

        if (!llvm_mlir)
          throw std::runtime_error(
              "Failed to convert to LLVM MLIR.");

        return moduleToStringWithDebugInfo(ttshared_module, debug_mode);
      },
      ret::take_ownership);

  m.def(
      "triton_linalg_to_llvm_mlir_separate_constants",
      [](std::string ttlinalg, mlir::MLIRContext &context, nb::dict codegen_options, int verbose_level, int debug_mode) {
        mlir::ModuleOp ttshared_module = parse_mlir_module(ttlinalg, context);

        arm64VectorExtension vector_ext;
        std::unordered_map<std::string, std::string> prof_options_map;
        for (auto it : codegen_options) {
          if (nb::cast<std::string>(it.first) == "arm64_vector_ext")
            vector_ext = getVectorExtFromString(nb::cast<std::string>(it.second));
          else
            prof_options_map[nb::cast<std::string>(it.first)] = nb::cast<std::string>(it.second);
        }

        std::vector<mlir::ModuleOp> modules =
            ::mlir::triton::arm64TritonLinalgToMultipleLLVMMLIRModules(
                ttshared_module, vector_ext, prof_options_map, verbose_level, debug_mode);

        if (modules.empty())
          throw std::runtime_error(
              "Failed to convert to LLVM MLIR with separate constants.");

        std::vector<std::string> result;
        result.reserve(modules.size());
        for (auto &mod : modules) {
          result.push_back(moduleToStringWithDebugInfo(mod, debug_mode));
        }
        return result;
      },
      ret::take_ownership);
}

void init_triton_qcom_arm64_backend(nb::module_ &m) {
  m.doc() = "Python bindings to the Qualcomm ARM64 Triton backend";
  init_triton_arm64_optimization(m);
  init_triton_linalg_to_mlir(m);
}
