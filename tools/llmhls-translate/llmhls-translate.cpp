//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The LLMHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "llmhls/Translation/EmitHLSCpp.h"

int main(int argc, char **argv) {
  mlir::llmhls::registerEmitHLSCppTranslation();

  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "LLMHLS Translation Tool"));
}
