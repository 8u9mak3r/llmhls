//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The LLMHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "llmhls-c/HLS.h"
#include "mlir/CAPI/Registration.h"
#include "llmhls/Dialect/HLS/HLS.h"

using namespace mlir;
using namespace llmhls;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(HLS, hls, hls::HLSDialect)
