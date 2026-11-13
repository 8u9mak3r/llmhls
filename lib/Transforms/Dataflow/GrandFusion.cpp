#include "mlir/IR/Dominance.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llmhls/Transforms/Passes.h"
#include "llmhls/Transforms/Utils.h"
#include <cassert>

using namespace mlir;
using namespace llmhls;
using namespace hls;

namespace {
LogicalResult generalizeNamedOpPrecondition(linalg::LinalgOp linalgOp) {
  if (isa<linalg::MatmulOp, linalg::BatchMatmulOp, linalg::ReduceOp>(linalgOp)) return success();
  return failure();
}

struct LinalgGeneralization : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  /// This rewrite pattern is copied from a MLIR inherit pass named
  /// LinalgGeneralizationPass(llvm-project/mlir/lib/Dialect/Linalg/Transforms/Generalization.cpp)
  /// plus a self-changed judgement function(generalizeNamedOpPrecondition) and placing tags on new Generic ops
  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                 PatternRewriter& rewriter) const override {
    if (failed(generalizeNamedOpPrecondition(linalgOp)))
      return rewriter.notifyMatchFailure(linalgOp, "preconditions not met");

    SmallVector<Value> inputs = linalgOp.getInputOperands();
    SmallVector<Value> outputs = linalgOp.getOutputOperands();
    SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
    SmallVector<StringRef> iterators = linalgOp.getIteratorTypesArray();
    SmallVector<Type> resultTypes = linalgOp.hasTensorSemantics()
                                        ? TypeRange(ValueRange(outputs))
                                        : TypeRange{};

    // All named ops have a region attached that can be inlined.
    assert(linalgOp->getNumRegions() == 1 &&
          "expect named op to have one region attached");
    linalg::GenericOp genericOp = rewriter.create<linalg::GenericOp>(
        linalgOp.getLoc(), resultTypes, inputs, outputs, indexingMaps, iterators);
    genericOp->setAttr("Actual Consumer", rewriter.getBoolAttr(true)); // place tags on new Generic ops

    rewriter.inlineRegionBefore(linalgOp->getRegion(0), genericOp.getRegion(),
                                genericOp.getRegion().begin());
    rewriter.replaceOp(linalgOp, genericOp->getResults());
    return success();                                
  }
};
} // namespace

namespace {
bool isFusionRoot(linalg::GenericOp genericOp) {
  // Generic reductions produced directly by a frontend are already in the
  // form expected by GrandFusion and therefore never pass through
  // LinalgGeneralization, which is where named reductions receive the
  // "Actual Consumer" marker.
  return genericOp->hasAttr("Actual Consumer") ||
         isReductionTypeGenericOp(genericOp);
}

AffineMap getNewAffineMapForOutputFromExpand(AffineMap map,
                                              SmallVector<ReassociationIndices, 4> reassociationIndices,
                                              ArrayRef<int64_t> newResultShape,
                                              MLIRContext* ctx) {
  SmallVector<mlir::AffineExpr, 8> exprs;

  size_t exprIdx = SIZE_MAX;
  for (auto expr : map.getResults()) {
    exprIdx += 1;
    auto indices = *(reassociationIndices.begin() + exprIdx);

    if (indices.size() > 1) {
      int64_t scale[8] = {1, 1, 1, 1, 1, 1, 1, 1};
      int64_t sum = 1;
      for (int64_t idx : llvm::reverse(indices)) {
        sum *= newResultShape[idx];
        scale[idx] = sum;
      }

            
      for (int64_t idx : indices) {
        AffineExpr newExpr = expr;
        // no need for bound check as long as scale[] is big enough
        newExpr = newExpr.floorDiv(mlir::getAffineConstantExpr(scale[idx + 1], ctx)); 
        newExpr = newExpr % mlir::getAffineConstantExpr(newResultShape[idx], ctx);
        exprs.push_back(newExpr);
      }
    } else {
      exprs.push_back(expr);
    }
  }

  return AffineMap::get(map.getNumDims(), map.getNumSymbols(), exprs, ctx);
}

AffineMap getNewAffineMapForOutputFromCollapse(AffineMap map,
                                                SmallVector<ReassociationIndices, 4> reassociationIndices,
                                                ArrayRef<int64_t> oldResultShape,
                                                MLIRContext* ctx) {
  SmallVector<mlir::AffineExpr, 8> exprs;
  auto mapExprsForOutputs = map.getResults();

  for (auto indices : reassociationIndices) {
    if (indices.size() == 1) {
      int64_t idx = indices[0];
      exprs.push_back(mapExprsForOutputs[idx]);
      continue;
    }

    int64_t scale[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t sum = 1;
    for (int64_t idx : llvm::reverse(indices)) {
      sum *= oldResultShape[idx];
      scale[idx] = sum;
    }

    AffineExpr newExpr = mlir::getAffineConstantExpr(0, ctx);
    for (int64_t idx : indices) 
      // no need for bound check as long as scale[] is big enough
      newExpr = newExpr + mapExprsForOutputs[idx] * mlir::getAffineConstantExpr(scale[idx + 1], ctx); 
    exprs.push_back(newExpr);
  }

  return AffineMap::get(map.getNumDims(), map.getNumSymbols(), exprs, ctx);
}

struct BackwardFuse : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                 PatternRewriter& rewriter) const override {
    if (!isFusionRoot(genericOp)) return failure();
    bool fail = true;
    for (auto& _ : genericOp->getUses()) if (!isa<hls::YieldOp>(_.getOwner())) fail = false;
    if (fail) return failure();

    MLIRContext* ctx = rewriter.getContext();
    auto loc = genericOp.getLoc();
    SmallVector<Type> resultTypes;
    SmallVector<Value> inputs = genericOp.getInputOperands();
    SmallVector<Value> outputs;

    SmallVector<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();
    SmallVector<AffineMap> indexingMapsForOutputs(
      indexingMaps.end() - genericOp.getOutputs().size(), indexingMaps.end()
    );
    indexingMaps.erase(indexingMaps.end() - genericOp.getOutputs().size(), indexingMaps.end());

    auto iteratorTypes = genericOp.getIteratorTypesArray();
    rewriter.setInsertionPoint(genericOp);

    for (auto result : genericOp.getResults()) {
      unsigned int resultIdx = result.getResultNumber();
      for (auto& use : llvm::make_early_inc_range(result.getUses())) {
        auto user = use.getOwner();
        auto output = genericOp.getOutputs()[resultIdx];

        if (isa<hls::YieldOp>(user)) {
          auto resultType = result.getType().cast<RankedTensorType>();
          resultTypes.push_back(resultType);

          outputs.push_back(output);
          indexingMaps.push_back(indexingMapsForOutputs[resultIdx]);
          use.set(output);
          continue;
        }

        auto resultType = user->getResult(0).getType().cast<RankedTensorType>();
        resultTypes.push_back(resultType);
        
        auto emptyOp = rewriter.create<tensor::EmptyOp>(
          loc, resultType.getShape(), resultType.getElementType()
        );

        Operation* fillOp = nullptr;
        if (isa<linalg::FillOp>(output.getDefiningOp())) {
          auto value = dyn_cast<arith::ConstantOp>(output.getDefiningOp()->getOperand(0).getDefiningOp()).getValue();
          auto cst = rewriter.create<arith::ConstantOp>(loc, value);
          fillOp = rewriter.create<linalg::FillOp>(loc, cst.getResult(), emptyOp.getResult());
        } else {
          fillOp = emptyOp.getOperation();
        }
        outputs.push_back(fillOp->getResult(0));

        auto map = indexingMapsForOutputs[resultIdx];
        // the only difference lies in the indexing map for the output
        if (isa<tensor::ExpandShapeOp>(user)) {
          auto castUser = dyn_cast<tensor::ExpandShapeOp>(user);
          auto reassociationIndices = castUser.getReassociationIndices();
        
          auto newResultShape = resultType.getShape();
          indexingMaps.push_back(
            getNewAffineMapForOutputFromExpand(map, reassociationIndices, newResultShape, ctx)
          );
        } else if (isa<linalg::TransposeOp>(user)) {
          auto castUser = dyn_cast<linalg::TransposeOp>(user);
          auto perm = castUser.getPermutation();

          auto mapExprsForOutputs = map.getResults();

          SmallVector<mlir::AffineExpr, 8> exprs;
          for (int64_t dim : perm) exprs.push_back(mapExprsForOutputs[dim]);
          indexingMaps.push_back(
            AffineMap::get(map.getNumDims(), map.getNumSymbols(), exprs, ctx)
          );
        } else if (isa<tensor::CollapseShapeOp>(user)) {
          auto castUser = dyn_cast<tensor::CollapseShapeOp>(user);
          auto reassociationIndices = castUser.getReassociationIndices();

          auto oldResultShape = dyn_cast<RankedTensorType>(genericOp->getResult(0).getType()).getShape();
          indexingMaps.push_back(
            getNewAffineMapForOutputFromCollapse(map, reassociationIndices, oldResultShape, ctx)
          );
        } else {
          user->emitError(
            "Operations to be fused backwards are only limited to: tensor.expand_shape, tensor.collapse_shape and linalg.transpose!"
          );
          return failure();
        }
        
        user->getResult(0).replaceAllUsesWith(fillOp->getResult(0));
        rewriter.eraseOp(user);
        
      }
    }

    auto newGenericOp = rewriter.create<linalg::GenericOp>(
      loc, resultTypes, inputs, outputs, indexingMaps, iteratorTypes
    );
    rewriter.inlineRegionBefore(genericOp->getRegion(0), newGenericOp.getRegion(), newGenericOp.getRegion().begin());

    Block& block = newGenericOp.getRegion().front();
    auto terminator = block.getTerminator();

    for (size_t i = 0; i < newGenericOp.getOutputs().size() - genericOp.getOutputs().size(); i += 1)
      block.addArgument(terminator->getOperand(0).getType(), terminator->getLoc());

    SmallVector<Value> yields;
    for (auto _ : outputs) yields.push_back(terminator->getOperand(0));
    rewriter.setInsertionPoint(terminator);
    rewriter.create<linalg::YieldOp>(terminator->getLoc(), ValueRange(yields));
    rewriter.eraseOp(terminator);

    newGenericOp->setAttr("Actual Consumer", rewriter.getBoolAttr(true));
    rewriter.eraseOp(genericOp);

    for (size_t i = 0; i < outputs.size(); i += 1) {
      auto definingOp = outputs[i].getDefiningOp();
      definingOp->getResult(0).replaceAllUsesExcept(newGenericOp.getResult(i), newGenericOp.getOperation());
    }

    return success();
  }
};
} // namespace

namespace {
AffineMap getNewAffineMapsForForwardFusedOp(AffineMap map,
                                            AffineMap indexingMapOfInput) {
  // `map` maps the producer loops to one of its inputs, while
  // `indexingMapOfInput` maps the consumer loops to the producer result.
  // Compose them instead of restricting the consumer map to dimensions and
  // constants: valid linalg indexing maps may also contain expressions such
  // as `mod` (RoPE) and `floordiv` (GQA).
  return map.compose(indexingMapOfInput);
}

struct ForwardFuse : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                 PatternRewriter& rewriter) const override {
    if (!isFusionRoot(genericOp)) return failure();

    MLIRContext* ctx = rewriter.getContext();
    auto loc = genericOp.getLoc();
    auto resultType = genericOp.getResult(0).getType().cast<RankedTensorType>();
    SmallVector<Value> inputs = genericOp.getInputOperands();
    SmallVector<Value> outputs = genericOp.getOutputOperands();
    SmallVector<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();
    SmallVector<StringRef> iteratorTypes = genericOp.getIteratorTypesArray();
    auto& region = genericOp.getRegion();

    SmallVector<AffineMap> indexingMapsOfOutputs(indexingMaps.end() - outputs.size(), indexingMaps.end());
    indexingMaps.erase(indexingMaps.end() - outputs.size(), indexingMaps.end());
    assert(inputs.size() == indexingMaps.size());

    SmallVector<Value> newInputs;
    SmallVector<AffineMap> newIndexingMaps;
    size_t argIdxToBeReplaced = 0;
    bool fused = false;
    for (auto [input, indexingMapOfInput] : llvm::zip(genericOp.getInputs(), indexingMaps)) {
      auto definingOp = input.getDefiningOp();
      if (definingOp == nullptr) goto _else;  // this means the input is a func arg, just ignore and jump to the else branch

      if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(definingOp)) {
        auto srcTypeOfDefiningOp = definingOp->getOperand(0).getType().cast<RankedTensorType>();
        auto dstTypeOfDefiningOp = definingOp->getResult(0).getType().cast<RankedTensorType>();
        assert(!isReshaped(srcTypeOfDefiningOp, dstTypeOfDefiningOp));

        newInputs.push_back(definingOp->getOperand(0));

        auto srcShapeOfDefiningOp = srcTypeOfDefiningOp.getShape();
        auto dstShapeOfDefiningOp = dstTypeOfDefiningOp.getShape();
        auto mapExprsOfInput = indexingMapOfInput.getResults();
        assert(dstShapeOfDefiningOp.size() == mapExprsOfInput.size());

        SmallVector<AffineExpr> newExprs;
        for (int64_t dim : srcShapeOfDefiningOp) {
          if (dim == 1) {
            newExprs.push_back(mlir::getAffineConstantExpr(0, ctx));
            continue;
          }

          size_t idx = llvm::find(dstShapeOfDefiningOp, dim) - dstShapeOfDefiningOp.begin();
          newExprs.push_back(mapExprsOfInput[idx]);
        }

        newIndexingMaps.push_back(
          AffineMap::get(indexingMapOfInput.getNumDims(), indexingMapOfInput.getNumSymbols(), newExprs, ctx)
        );

        argIdxToBeReplaced += 1;
        fused = true;
      } else if (isa<linalg::TransposeOp>(definingOp)) {
        auto castDefiningOp = dyn_cast<linalg::TransposeOp>(definingOp);
        auto perm = castDefiningOp.getPermutation();
        auto mapExprsOfInput = indexingMapOfInput.getResults();

        newInputs.push_back(definingOp->getOperand(0));

        SmallVector<AffineExpr> newExprs;
        for (int64_t dim : perm) newExprs.push_back(mapExprsOfInput[dim]);

        newIndexingMaps.push_back(
          AffineMap::get(indexingMapOfInput.getNumDims(), indexingMapOfInput.getNumSymbols(), newExprs, ctx)
        );

        argIdxToBeReplaced += 1;
        fused = true;
      } else if (
          definingOp->getName().getDialectNamespace() ==
              math::MathDialect::getDialectNamespace() &&
          OpTrait::hasElementwiseMappableTraits(definingOp) &&
          definingOp->getNumResults() == 1 &&
          llvm::all_of(definingOp->getOperandTypes(), [](Type type) {
            return type.isa<RankedTensorType>();
          })) {
        // Replace the tensor-level math op with its tensor operands and clone
        // the equivalent scalar op into the consumer's linalg.generic body.
        // ElementwiseMappable guarantees that all tensor operands and results
        // have the same shape, so each operand composes with the consumer
        // through the same indexing map.
        size_t firstInsertedArg = argIdxToBeReplaced;
        newInputs.append(definingOp->operand_begin(), definingOp->operand_end());
        for (Value operand : definingOp->getOperands()) {
          newIndexingMaps.push_back(indexingMapOfInput);
          auto operandType = operand.getType().cast<RankedTensorType>();
          region.front().insertArgument(argIdxToBeReplaced,
                                        operandType.getElementType(), loc);
          argIdxToBeReplaced += 1;
        }

        auto& block = region.front();
        SmallVector<Value> scalarOperands;
        for (size_t i = firstInsertedArg; i < argIdxToBeReplaced; ++i)
          scalarOperands.push_back(block.getArgument(i));

        auto resultElementType =
            input.getType().cast<RankedTensorType>().getElementType();
        rewriter.setInsertionPointToStart(&block);
        auto* scalarOp = rewriter.create(
            definingOp->getLoc(), definingOp->getName().getIdentifier(),
            scalarOperands, TypeRange{resultElementType},
            definingOp->getAttrs());

        auto argToBeReplaced = block.getArgument(argIdxToBeReplaced);
        argToBeReplaced.replaceAllUsesWith(scalarOp->getResult(0));
        block.eraseArgument(argIdxToBeReplaced);
        fused = true;
      } else if (isa<linalg::GenericOp>(definingOp)) {
        auto castDefiningOp = dyn_cast<linalg::GenericOp>(definingOp);
        SmallVector<AffineMap> indexingMapsOfDefiningOp = castDefiningOp.getIndexingMapsArray();
        assert(
          indexingMapsOfDefiningOp.back().isIdentity() &&
          indexingMapsOfDefiningOp.back().getNumDims() == indexingMapOfInput.getResults().size()
        );

        SmallVector<Value> inputsOfDefiningOp = castDefiningOp.getInputOperands();
        indexingMapsOfDefiningOp.pop_back();
        assert(indexingMapsOfDefiningOp.size() == inputsOfDefiningOp.size());
    
        auto& block = region.front();
        auto& blockToBeFused = castDefiningOp.getRegion().front();

        auto argTypesToBeFused = blockToBeFused.getArgumentTypes();
        assert(argTypesToBeFused.size() - 1 == inputsOfDefiningOp.size());

        BlockAndValueMapping mapper;
        auto argsToBeFused = blockToBeFused.getArguments();
        assert(argsToBeFused.size() - 1 == inputsOfDefiningOp.size());
        for (size_t i = 0; i < inputsOfDefiningOp.size(); i += 1) {
          auto newIndexingMap = getNewAffineMapsForForwardFusedOp(
              indexingMapsOfDefiningOp[i], indexingMapOfInput);

          // The same tensor may be exposed repeatedly while fusing a producer
          // generic. It is only the same linalg input when its composed access
          // map is also identical.
          size_t existingInputIdx = SIZE_MAX;
          for (size_t j = 0; j < newInputs.size(); ++j) {
            if (newInputs[j] == inputsOfDefiningOp[i] &&
                newIndexingMaps[j] == newIndexingMap) {
              existingInputIdx = j;
              break;
            }
          }

          if (existingInputIdx != SIZE_MAX) {
            mapper.map(argsToBeFused[i],
                       block.getArgument(existingInputIdx));
            continue;
          }

          newInputs.push_back(inputsOfDefiningOp[i]);
          newIndexingMaps.push_back(newIndexingMap);
          block.insertArgument(argIdxToBeReplaced, argTypesToBeFused[i], loc);
          mapper.map(argsToBeFused[i],
                     block.getArgument(argIdxToBeReplaced));
          argIdxToBeReplaced += 1;
        }

        rewriter.setInsertionPointToStart(&block);
        Operation* replacer;
        for (auto& bodyOp : blockToBeFused.without_terminator()) replacer = rewriter.clone(bodyOp, mapper);

        auto argToBeReplaced = block.getArgument(argIdxToBeReplaced);
        argToBeReplaced.replaceAllUsesWith(replacer->getResult(0));
        block.eraseArgument(argIdxToBeReplaced);
        fused = true;
      } else {
_else:
        newInputs.push_back(input);
        newIndexingMaps.push_back(indexingMapOfInput);
        argIdxToBeReplaced += 1;
      }
    }

    if (!fused) return failure();

    newIndexingMaps.append(indexingMapsOfOutputs);

    rewriter.setInsertionPointAfter(genericOp);
    auto newGenericOp = rewriter.create<linalg::GenericOp>(
      loc, TypeRange(resultType), ValueRange(newInputs), ValueRange(outputs), newIndexingMaps, iteratorTypes
    );
    
    rewriter.inlineRegionBefore(region, newGenericOp.getRegion(), newGenericOp.getRegion().begin());
    // genericOp.getResult(0).replaceAllUsesWith(newGenericOp.getResult(0));
    for (auto _ : genericOp.getResults()) _.replaceAllUsesWith(newGenericOp.getResult(_.getResultNumber()));
    newGenericOp->setAttr("Actual Consumer", rewriter.getBoolAttr(true));
    rewriter.eraseOp(genericOp);
    
    return success();                                
  }
};
} // namespace

namespace {
struct GrandFusion
    : public GrandFusionBase<GrandFusion> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<LinalgGeneralization>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns), {false, true, 0L});
    patterns.clear();
    patterns.add<BackwardFuse>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns), {false, true, 0L});
    patterns.clear();
    patterns.add<ForwardFuse>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns), {false, true, 0L});
    patterns.clear();
  }
};
} // namespace

std::unique_ptr<Pass> llmhls::createGrandFusionPass() {
  return std::make_unique<GrandFusion>();
}
