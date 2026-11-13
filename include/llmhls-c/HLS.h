//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The LLMHLS Authors.
//
//===----------------------------------------------------------------------===//

#ifndef LLMHLS_C_DIALECT_HLS_H
#define LLMHLS_C_DIALECT_HLS_H

#include "mlir-c/RegisterEverything.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(HLS, hls);

#ifdef __cplusplus
}
#endif

#endif // LLMHLS_C_DIALECT_HLS_H
