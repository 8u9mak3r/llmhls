//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The LLMHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "llmhls/Transforms/Passes.h"
#include "llmhls/Transforms/Utils.h"

using namespace mlir;
using namespace llmhls;
using namespace hls;

/// Apply function pipelining to the input function, all contained loops are
/// automatically fully unrolled.
static bool applyFuncPipelining(func::FuncOp func, int64_t targetII) {
  if (auto directive = getFuncDirective(func))
    if (directive.getDataflow())
      return false;

  if (!func.getOps<func::CallOp>().empty())
    return false;

  if (!applyFullyLoopUnrolling(func.front()))
    return false;
  setFuncDirective(func, true, targetII, false);
  return true;
}

namespace {
struct FuncPipelining : public FuncPipeliningBase<FuncPipelining> {
  void runOnOperation() override {
    auto func = getOperation();

    if (func.getName() == targetFunc)
      applyFuncPipelining(func, targetII);
  }
};
} // namespace

std::unique_ptr<Pass> llmhls::createFuncPipeliningPass() {
  return std::make_unique<FuncPipelining>();
}
