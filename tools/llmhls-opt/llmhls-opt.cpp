//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The LLMHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llmhls/InitAllDialects.h"
#include "llmhls/InitAllPasses.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::llmhls::registerAllDialects(registry);
  mlir::llmhls::registerAllPasses();

  return mlir::failed(mlir::MlirOptMain(
      argc, argv, "LLMHLS Optimization Tool", registry, true));
}
