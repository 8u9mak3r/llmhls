//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The LLMHLS Authors.
//
//===----------------------------------------------------------------------===//

#ifndef LLMHLS_INITALLPASSES_H
#define LLMHLS_INITALLPASSES_H

#include "mlir/InitAllPasses.h"
#include "llmhls/Transforms/Passes.h"

namespace mlir {
namespace llmhls {

// Add all the LLMHLS passes.
inline void registerAllPasses() {
  llmhls::registerTransformsPasses();
  mlir::registerAllPasses();
}

} // namespace llmhls
} // namespace mlir

#endif // LLMHLS_INITALLPASSES_H
