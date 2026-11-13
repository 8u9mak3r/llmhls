//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The LLMHLS Authors.
//
//===----------------------------------------------------------------------===//

#ifndef LLMHLS_C_TRANSLATION_EMITHLSCPP_H
#define LLMHLS_C_TRANSLATION_EMITHLSCPP_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirLogicalResult mlirEmitHlsCpp(MlirModule module,
                                                    MlirStringCallback callback,
                                                    void *userData);

#ifdef __cplusplus
}
#endif

#endif // LLMHLS_C_TRANSLATION_EMITHLSCPP_H
