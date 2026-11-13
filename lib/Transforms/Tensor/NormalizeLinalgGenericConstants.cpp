//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The LLMHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llmhls/Transforms/Passes.h"

using namespace mlir;
using namespace llmhls;

namespace {

static TypedAttr getSplatValue(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return {};

  auto denseValue = constant.getValue().dyn_cast<DenseElementsAttr>();
  if (!denseValue || !denseValue.isSplat())
    return {};
  return denseValue.getSplatValue<TypedAttr>();
}

/// Materialize shaped constants used as linalg.generic output initializers as
/// tensor.empty + linalg.fill. This avoids carrying a constant tensor buffer
/// into bufferization.
struct MaterializeConstantOutputs : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    rewriter.startRootUpdate(genericOp);

    for (OpOperand *output : genericOp.getOutputOperands()) {
      auto tensorType = output->get().getType().dyn_cast<RankedTensorType>();
      TypedAttr splatValue = getSplatValue(output->get());
      if (!tensorType || !tensorType.hasStaticShape() || !splatValue)
        continue;

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(genericOp);
      auto scalarConstant = rewriter.create<arith::ConstantOp>(
          genericOp.getLoc(), splatValue, splatValue.getType());
      auto empty = rewriter.create<tensor::EmptyOp>(
          genericOp.getLoc(), tensorType.getShape(),
          tensorType.getElementType());
      auto fill = rewriter.create<linalg::FillOp>(
          genericOp.getLoc(), scalarConstant.getResult(), empty.getResult());
      output->set(fill.getResult(0));
      changed = true;
    }

    if (!changed) {
      rewriter.cancelRootUpdate(genericOp);
      return failure();
    }
    rewriter.finalizeRootUpdate(genericOp);
    return success();
  }
};

/// Replace one splat tensor input at a time with a scalar arith.constant that
/// is captured by the linalg.generic body. The corresponding input operand,
/// indexing map, and body block argument are removed together.
struct ScalarizeConstantInputs : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!genericOp.hasTensorSemantics())
      return failure();

    for (OpOperand *constantOperand : genericOp.getInputOperands()) {
      if (!constantOperand->get().getType().isa<RankedTensorType>())
        continue;
      TypedAttr splatValue = getSplatValue(constantOperand->get());
      if (!splatValue)
        continue;

      SmallVector<Value> newInputs;
      SmallVector<AffineMap> newIndexingMaps;
      SmallVector<Location> fusedLocations{genericOp.getLoc()};
      for (OpOperand *input : genericOp.getInputOperands()) {
        if (input == constantOperand)
          continue;
        newInputs.push_back(input->get());
        newIndexingMaps.push_back(genericOp.getMatchingIndexingMap(input));
        fusedLocations.push_back(input->get().getLoc());
      }
      for (OpOperand *output : genericOp.getOutputOperands())
        newIndexingMaps.push_back(genericOp.getMatchingIndexingMap(output));

      // Removing an operand must not make the loop bounds uncomputable.
      if (!inversePermutation(concatAffineMaps(newIndexingMaps)))
        continue;

      rewriter.setInsertionPoint(genericOp);
      auto scalarConstant = rewriter.create<arith::ConstantOp>(
          constantOperand->get().getLoc(), splatValue, splatValue.getType());
      auto newGenericOp = rewriter.create<linalg::GenericOp>(
          rewriter.getFusedLoc(fusedLocations), genericOp->getResultTypes(),
          newInputs, genericOp.getOutputs(), newIndexingMaps,
          genericOp.getIteratorTypesArray());

      // Preserve non-structural attributes such as "Actual Consumer".
      for (NamedAttribute attribute : genericOp->getAttrs())
        if (!newGenericOp->hasAttr(attribute.getName()))
          newGenericOp->setAttr(attribute.getName(), attribute.getValue());

      Block &oldBody = genericOp.getRegion().front();
      BlockAndValueMapping mapping;
      mapping.map(oldBody.getArgument(constantOperand->getOperandNumber()),
                  scalarConstant);
      rewriter.cloneRegionBefore(genericOp.getRegion(),
                                 newGenericOp.getRegion(),
                                 newGenericOp.getRegion().begin(), mapping);
      rewriter.replaceOp(genericOp, newGenericOp.getResults());
      return success();
    }
    return failure();
  }
};

struct NormalizeLinalgGenericConstants
    : public NormalizeLinalgGenericConstantsBase<
          NormalizeLinalgGenericConstants> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MaterializeConstantOutputs, ScalarizeConstantInputs>(
        &getContext());
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> llmhls::createNormalizeLinalgGenericConstantsPass() {
  return std::make_unique<NormalizeLinalgGenericConstants>();
}
