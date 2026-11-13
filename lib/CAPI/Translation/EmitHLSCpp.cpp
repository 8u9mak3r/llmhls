//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The LLMHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "llmhls-c/EmitHLSCpp.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "llmhls/Translation/EmitHLSCpp.h"

using namespace mlir;
using namespace llmhls;

MlirLogicalResult mlirEmitHlsCpp(MlirModule module, MlirStringCallback callback,
                                 void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(emitHLSCpp(unwrap(module), stream));
}
