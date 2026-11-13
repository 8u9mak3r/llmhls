//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The LLMHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "llmhls/Transforms/Passes.h"
#include "llmhls/Transforms/Utils.h"

using namespace mlir;
using namespace llmhls;
using namespace hls;

namespace {

/// Add duplicate outputs to a node and thread them through nested nodes and
/// schedules until reaching the stores that write the original output.
static LogicalResult duplicateOutputWrites(Value output,
                                           ArrayRef<Value> duplicateOutputs,
                                           IRRewriter &rewriter);

static LogicalResult appendNodeOutputs(NodeOp node, Value output,
                                       ArrayRef<Value> duplicateOutputs,
                                       IRRewriter &rewriter) {
  auto outputIt = llvm::find(node.getOutputs(), output);
  if (outputIt == node.getOutputs().end())
    return node.emitError("cannot find the output to duplicate");

  unsigned outputIdx = outputIt - node.getOutputs().begin();
  unsigned numInputs = node.getNumInputs();
  unsigned numOutputs = node.getNumOutputs();
  Block &body = node.getBody().front();
  Value outputArg = body.getArgument(numInputs + outputIdx);

  node.getOutputsMutable().append(duplicateOutputs);

  SmallVector<Value> duplicateOutputArgs;
  unsigned insertIdx = numInputs + numOutputs;
  for (Value duplicate : duplicateOutputs) {
    duplicateOutputArgs.push_back(body.insertArgument(
        insertIdx++, duplicate.getType(), duplicate.getLoc()));
  }

  return duplicateOutputWrites(outputArg, duplicateOutputArgs, rewriter);
}

static LogicalResult duplicateOutputWrites(Value output,
                                           ArrayRef<Value> duplicateOutputs,
                                           IRRewriter &rewriter) {
  bool foundWritePath = false;

  for (OpOperand &use : llvm::make_early_inc_range(output.getUses())) {
    Operation *owner = use.getOwner();

    if (auto schedule = dyn_cast<ScheduleOp>(owner)) {
      unsigned operandIdx = use.getOperandNumber();
      Block &body = schedule.getBody().front();
      Value outputArg = body.getArgument(operandIdx);

      schedule.operandsMutable().append(duplicateOutputs);
      SmallVector<Value> duplicateOutputArgs;
      for (Value duplicate : duplicateOutputs)
        duplicateOutputArgs.push_back(
            body.addArgument(duplicate.getType(), duplicate.getLoc()));

      if (failed(
              duplicateOutputWrites(outputArg, duplicateOutputArgs, rewriter)))
        return failure();
      foundWritePath = true;
      continue;
    }

    if (auto node = dyn_cast<NodeOp>(owner)) {
      if (node.getOperandKind(use) != OperandKind::OUTPUT)
        continue;
      if (failed(appendNodeOutputs(node, output, duplicateOutputs, rewriter)))
        return failure();
      foundWritePath = true;
      continue;
    }

    if (auto store = dyn_cast<AffineStoreOp>(owner)) {
      if (store.getMemRef() != output)
        continue;
      rewriter.setInsertionPointAfter(store);
      for (Value duplicate : duplicateOutputs)
        rewriter.create<AffineStoreOp>(store.getLoc(), store.getValueToStore(),
                                       duplicate, store.getAffineMap(),
                                       store.getMapOperands());
      foundWritePath = true;
      continue;
    }

    if (auto store = dyn_cast<memref::StoreOp>(owner)) {
      if (store.getMemref() != output)
        continue;
      rewriter.setInsertionPointAfter(store);
      for (Value duplicate : duplicateOutputs)
        rewriter.create<memref::StoreOp>(store.getLoc(), store.getValue(),
                                         duplicate, store.getIndices());
      foundWritePath = true;
    }
  }

  if (!foundWritePath) {
    if (Operation *definingOp = output.getDefiningOp())
      return definingOp->emitError(
          "cannot find a supported write path for duplicated output");
    return failure();
  }
  return success();
}

static LogicalResult duplicateMultiConsumerOutputs(NodeOp node,
                                                   IRRewriter &rewriter) {
  ScheduleOp schedule = node.getScheduleOp();
  SmallVector<Value> originalOutputs(node.getOutputs());

  for (Value output : originalOutputs) {
    auto buffer = output.getDefiningOp<BufferOp>();
    if (!buffer)
      continue;

    SmallVector<NodeOp> consumers;
    llvm::SmallDenseSet<Operation *> seenConsumers;
    for (OpOperand &use : output.getUses()) {
      auto consumer = dyn_cast<NodeOp>(use.getOwner());
      if (!consumer || consumer == node ||
          consumer.getScheduleOp() != schedule ||
          consumer.getOperandKind(use) != OperandKind::INPUT)
        continue;
      if (seenConsumers.insert(consumer).second)
        consumers.push_back(consumer);
    }

    if (consumers.size() < 2)
      continue;

    rewriter.setInsertionPoint(node);
    SmallVector<Value> duplicateBuffers;
    for (size_t i = 1; i < consumers.size(); ++i) {
      auto duplicate = cast<BufferOp>(rewriter.clone(*buffer));
      duplicateBuffers.push_back(duplicate.getMemref());
    }

    for (auto [consumer, duplicate] :
         llvm::zip(llvm::drop_begin(consumers), duplicateBuffers)) {
      for (OpOperand &use : llvm::make_early_inc_range(output.getUses()))
        if (use.getOwner() == consumer &&
            consumer.getOperandKind(use) == OperandKind::INPUT)
          use.set(duplicate);
    }

    if (failed(appendNodeOutputs(node, output, duplicateBuffers, rewriter)))
      return failure();
  }

  return success();
}

struct BufferDuplicate : public BufferDuplicateBase<BufferDuplicate> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(&getContext());

    SmallVector<ScheduleOp> outermostSchedules;
    func.walk([&](ScheduleOp schedule) {
      if (!schedule->getParentOfType<NodeOp>())
        outermostSchedules.push_back(schedule);
    });

    for (ScheduleOp schedule : outermostSchedules) {
      SmallVector<NodeOp> nodes(schedule.getOps<NodeOp>());
      for (NodeOp node : nodes)
        if (failed(duplicateMultiConsumerOutputs(node, rewriter))) {
          signalPassFailure();
          return;
        }
    }

    func.walk([&](ScheduleOp schedule) {
      schedule.setIsLegalAttr(UnitAttr::get(&getContext()));
    });
  }
};

} // namespace

std::unique_ptr<Pass> llmhls::createBufferDuplicatePass() {
  return std::make_unique<BufferDuplicate>();
}
