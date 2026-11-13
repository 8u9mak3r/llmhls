//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The LLMHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llmhls/Transforms/Passes.h"

using namespace mlir;
using namespace llmhls;

namespace {
#define GEN_PASS_REGISTRATION
#include "llmhls/Transforms/Passes.h.inc"
} // namespace

namespace {
struct LLMHLSDSEPipelineOptions
    : public PassPipelineOptions<LLMHLSDSEPipelineOptions> {
  Option<std::string> hlsTopFunc{
      *this, "top-func", llvm::cl::init("main"),
      llvm::cl::desc("Specify the top function of the design")};

  Option<std::string> dseTargetSpec{
      *this, "target-spec", llvm::cl::init("./config.json"),
      llvm::cl::desc(
          "File path: target backend specifications and configurations")};
};
} // namespace

void llmhls::registerLLMHLSDSEPipeline() {
  PassPipelineRegistration<LLMHLSDSEPipelineOptions>(
      "llmhls-dse-pipeline",
      "Launch design space exploration for C/C++ kernel",
      [](OpPassManager &pm, const LLMHLSDSEPipelineOptions &opts) {
        // Legalize the input program.
        pm.addPass(llmhls::createFuncPreprocessPass(opts.hlsTopFunc));
        pm.addPass(llmhls::createMaterializeReductionPass());
        pm.addPass(bufferization::createBufferLoopHoistingPass());

        // Apply the automatic design space exploration to the top function.
        pm.addPass(llmhls::createDesignSpaceExplorePass(opts.dseTargetSpec));

        // Finally, estimate the QoR of the DSE result.
        pm.addPass(llmhls::createQoREstimationPass(opts.dseTargetSpec));
      });
}

void llmhls::addCreateSubviewPasses(OpPassManager &pm,
                                      CreateSubviewMode mode) {
  pm.addPass(llmhls::createCreateMemrefSubviewPass(mode));
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

void llmhls::addSimplifyAffineLoopPasses(OpPassManager &pm) {
  pm.addPass(mlir::createAffineLoopNormalizePass());
  pm.addPass(mlir::createSimplifyAffineStructuresPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

namespace {
struct ScaleFlowPyTorchPipelineOptions
    : public PassPipelineOptions<ScaleFlowPyTorchPipelineOptions> {
  Option<std::string> hlsTopFunc{
      *this, "top-func", llvm::cl::init("forward"),
      llvm::cl::desc("Specify the top function of the design")};

  // Option<unsigned> optLevel{
  //     *this, "opt-level", llvm::cl::init(1),
  //     llvm::cl::desc("Optimization level from 0 to 7 (default level is 1)")};

  // Option<unsigned> dataflowGranularity{
  //     *this, "dataflow-granularity",
  //     llvm::cl::desc("The granularity of dataflow (set 0 to disable)")};

  Option<double> fusionTolerance{
      *this, "fusion-tolerance", llvm::cl::init(100.0),
      llvm::cl::desc("Additional computation tolerated while loop fusing "
                     "(default is 100.0)")};

  Option<unsigned> loopTileSize{
      *this, "loop-tile-size", llvm::cl::init(2),
      llvm::cl::desc("The tile size of each loop (must larger equal to 1)")};

  Option<unsigned> loopUnrollFactor{
      *this, "loop-unroll-factor", llvm::cl::init(0),
      llvm::cl::desc("The overall loop unrolling factor (set 0 to disable)")};

  Option<bool> greedyFuse{
      *this, "greedy-fuse", llvm::cl::init(true),
      llvm::cl::desc("Whether to fuse ops into a dataflow task in a greedy pattern")};

  Option<bool> complexityAware{
      *this, "complexity-aware", llvm::cl::init(true),
      llvm::cl::desc("Whether to consider node complexity in the transform")};

  Option<bool> correlationAware{
      *this, "correlation-aware", llvm::cl::init(true),
      llvm::cl::desc("Whether to consider node correlation in the transform")};

  Option<bool> placeExternalBuffer{
      *this, "place-external-buffer", llvm::cl::init(false),
      llvm::cl::desc("Place buffers in external memories")};

  Option<bool> balanceDataflow{
      *this, "balance-dataflow", llvm::cl::init(false),
      llvm::cl::desc("Whether to balance the dataflow")};

  Option<bool> axiInterface{*this, "axi-interface", llvm::cl::init(false),
                            llvm::cl::desc("Create AXI interface")};

  Option<bool> vectorize{*this, "vectorize", llvm::cl::init(false),
                         llvm::cl::desc("Vectorize with factor of 2")};

  Option<bool> tosaInput{*this, "tosa-input", llvm::cl::init(false),
                         llvm::cl::desc("Inidicate the input IR is TOSA")};

  Option<bool> fakeQuantize{
      *this, "fake-quantize", llvm::cl::init(false),
      llvm::cl::desc("Trigger the fake quantization (just for testing use)")};

  Option<unsigned> startPoint{
      *this, "start-point", llvm::cl::init(0),
      llvm::cl::desc("Start the pipeline at the given debug point")};

  Option<unsigned> debugPoint{
      *this, "debug-point", llvm::cl::init(0),
      llvm::cl::desc("Stop the pipeline at the given debug point")};
};
} // namespace

void llmhls::registerScaleFlowPyTorchPipeline() {
  PassPipelineRegistration<ScaleFlowPyTorchPipelineOptions>(
      "scaleflow-pytorch-pipeline",
      "Compile TOSA (from Torch-MLIR) to HLS C++ with ScaleFlow",
      [](OpPassManager &pm, const ScaleFlowPyTorchPipelineOptions &opts) {
        switch (opts.startPoint)
        {
          case 1:
            goto entry_1;
            break;

          case 2:
            goto entry_2;
            break;

          case 3:
            goto entry_3;
            break;

          case 4:
            goto entry_4;
            break;
          
          case 5:
            goto entry_5;
            break;
          
          case 6:
            goto entry_6;
            break;

          case 7:
            goto entry_7;
            break;

          case 8:
            goto entry_8;
            break;

          case 9:
            goto entry_9;
            break;

          case 10:
            goto entry_10;
            break;

          case 11:
            goto entry_11;
            break;

          case 12:
            goto entry_12;
            break;

          case 13:
            goto entry_13;
            break;

          case 14:
            goto entry_14;
            break;
        
          default:
            break;
        }

entry_1:
        if (opts.tosaInput) {
          // TOSA optimization.
          pm.addPass(llmhls::createTosaSimplifyGraphPass());
          pm.addPass(llmhls::createCreateDataflowFromTosaPass());
          pm.addPass(mlir::createCanonicalizerPass());

          // TOSA to Linalg conversion.
          tosa::addTosaToLinalgPasses(pm);
          pm.addPass(tosa::createTosaToArith());
          pm.addPass(tosa::createTosaToTensor());
        }

        // Linalg fake quantization.
        if (opts.fakeQuantize)
          pm.addPass(llmhls::createLinalgFakeQuantizePass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 1)
          return;

entry_2:         
        // Linalg optimization.
        // pm.addPass(mlir::createLinalgElementwiseOpFusionPass());
        pm.addPass(llmhls::createCreateDataflowFromLinalgPass(opts.greedyFuse));
        // pm.addPass(mlir::createLinalgElementwiseOpFusionPass());
        pm.addPass(llmhls::createUpliftReshapesBetweenTasksPass());
        pm.addPass(mlir::createConvertTensorToLinalgPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(llmhls::createGrandFusionPass());
        pm.addPass(llmhls::createNormalizeLinalgGenericConstantsPass());
        if (opts.debugPoint == 2)
          return;
          
entry_3:
        // Bufferization.
        pm.addPass(mlir::createLinalgBufferizePass());
        pm.addPass(arith::createArithBufferizePass());
        pm.addPass(mlir::createTensorBufferizePass());
        pm.addPass(func::createFuncBufferizePass());
        pm.addPass(bufferization::createBufferResultsToOutParamsPass());
        pm.addPass(llmhls::createBufferizeDataflowPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 3)
          return;

entry_4:
        // Linalg to Affine conversion.
        pm.addPass(mlir::createLinalgGeneralizationPass());
        pm.addPass(llmhls::createSimplifyCopyPass());
        pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
        pm.addPass(llmhls::createLowerCopyToAffinePass());
        pm.addPass(memref::createFoldMemRefAliasOpsPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 4)
          return;

entry_5:
        // Affine loop fusion.
        pm.addPass(llmhls::createFuncPreprocessPass(opts.hlsTopFunc));
        pm.addPass(llmhls::createAffineLoopFusionPass(opts.fusionTolerance));
        llmhls::addSimplifyAffineLoopPasses(pm);
        llmhls::addCreateSubviewPasses(pm);
        pm.addPass(llmhls::createRaiseAffineToCopyPass());
        pm.addPass(llmhls::createSimplifyCopyPass());
        pm.addPass(llmhls::createLowerCopyToAffinePass());
        pm.addPass(memref::createFoldMemRefAliasOpsPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 5)
          return;

entry_6:
        // Place dataflow buffers.
        pm.addPass(
            llmhls::createPlaceDataflowBufferPass(opts.placeExternalBuffer, opts.axiInterface));

        // if (opts.vectorize) {
        //   pm.addPass(mlir::createSuperVectorizePass({2}));
        //   pm.addPass(mlir::createCanonicalizerPass());
        // }

        if (opts.debugPoint == 6)
          return;

entry_7:
        // Affine loop tiling.
        pm.addPass(llmhls::createFuncPreprocessPass(opts.hlsTopFunc));
        pm.addPass(bufferization::createBufferLoopHoistingPass());
        pm.addPass(llmhls::createAffineLoopPerfectionPass());
        pm.addPass(llmhls::createAffineLoopOrderOptPass());
        pm.addPass(llmhls::createAffineLoopTilePass(opts.loopTileSize));
        pm.addPass(mlir::createSimplifyAffineStructuresPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 7)
          return;

entry_8:
        // Local buffer allocation.
        llmhls::addCreateSubviewPasses(pm);
        pm.addPass(llmhls::createCreateLocalBufferPass(/*externalBufferOnly=*/false));
        pm.addPass(llmhls::createLowerCopyToAffinePass());
        pm.addPass(memref::createFoldMemRefAliasOpsPass());
        pm.addPass(mlir::createSimplifyAffineStructuresPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 8)
          return;

entry_9:
        // Affine loop dataflowing.
        pm.addPass(llmhls::createCollapseMemrefUnitDimsPass());
        pm.addPass(llmhls::createAffineStoreForwardPass());
        pm.addPass(llmhls::createCreateDataflowFromAffinePass());
        pm.addPass(llmhls::createStreamDataflowTaskPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 9)
          return;

entry_10:
        // Lower and optimize dataflow.
        pm.addPass(llmhls::createLowerDataflowPass());
        pm.addPass(llmhls::createEliminateMultiProducerPass());
        pm.addPass(llmhls::createEliminateMultiConsumerPass());
        pm.addPass(llmhls::createScheduleDataflowNodePass());
        if (opts.balanceDataflow.getValue())
          pm.addPass(llmhls::createBalanceDataflowNodePass());
        pm.addPass(llmhls::createLowerCopyToAffinePass());
        pm.addPass(llmhls::createAffineStoreForwardPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 10)
          return;

entry_11:
        // Parallelize dataflow.
        pm.addPass(llmhls::createParallelizeDataflowNodePass(
            opts.loopUnrollFactor, /*unrollPointLoopOnly=*/true,
            opts.complexityAware, opts.correlationAware));
        pm.addPass(mlir::createSimplifyAffineStructuresPass());
        // pm.addPass(llmhls::createLegalizeDataflowPass());
        pm.addPass(llmhls::createBufferDuplicatePass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 11)
          return;

entry_12:
        // Memory optimization.
        pm.addPass(llmhls::createSimplifyAffineIfPass());
        pm.addPass(llmhls::createAffineStoreForwardPass());
        pm.addPass(llmhls::createReduceInitialIntervalPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 12)
          return;

entry_13:
        // Convert dataflow to func.
        pm.addPass(llmhls::createCreateTokenStreamPass());
        pm.addPass(llmhls::createConvertDataflowToFuncPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 13)
          return;

entry_14:
        // Directive-level optimization.
        if (opts.axiInterface)
          pm.addPass(llmhls::createCreateAxiInterfacePass(opts.hlsTopFunc));
        pm.addPass(llmhls::createLoopPipeliningPass());
        pm.addPass(llmhls::createArrayPartitionPass());
        pm.addPass(llmhls::createCreateHLSPrimitivePass());
        pm.addPass(mlir::createCanonicalizerPass());
      });
}

/// FIXME: Don't use! This is just for testing purpose.
void llmhls::registerScaleFlowPyTorchPipelinePost() {
  PassPipelineRegistration<ScaleFlowPyTorchPipelineOptions>(
      "scaleflow-pytorch-pipeline-post",
      "Compile TOSA (from Torch-MLIR) to HLS C++ with ScaleFlow",
      [](OpPassManager &pm, const ScaleFlowPyTorchPipelineOptions &opts) {
        // Parallelize dataflow.
        if (opts.loopUnrollFactor) {
          pm.addPass(llmhls::createParallelizeDataflowNodePass(
              opts.loopUnrollFactor, /*unrollPointLoopOnly=*/true));
          // pm.addPass(llmhls::createAffineLoopUnrollJamPass(
          //     opts.loopUnrollFactor, /*unrollPointLoopOnly=*/true));
          pm.addPass(mlir::createSimplifyAffineStructuresPass());
          pm.addPass(mlir::createCanonicalizerPass());
        }

        if (opts.debugPoint == 11)
          return;

        // Memory optimization.
        pm.addPass(llmhls::createSimplifyAffineIfPass());
        pm.addPass(llmhls::createAffineStoreForwardPass());
        pm.addPass(llmhls::createReduceInitialIntervalPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 12)
          return;

        // Convert dataflow to func.
        pm.addPass(llmhls::createCreateTokenStreamPass());
        pm.addPass(llmhls::createConvertDataflowToFuncPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 13)
          return;

        // Directive-level optimization.
        if (opts.axiInterface)
          pm.addPass(llmhls::createCreateAxiInterfacePass(opts.hlsTopFunc));
        pm.addPass(llmhls::createLoopPipeliningPass());
        pm.addPass(llmhls::createArrayPartitionPass());
        pm.addPass(llmhls::createCreateHLSPrimitivePass());
        pm.addPass(mlir::createCanonicalizerPass());
      });
}

void llmhls::registerScaleFlowCppPipeline() {
  PassPipelineRegistration<ScaleFlowPyTorchPipelineOptions>(
      "scaleflow-cpp-pipeline", "Compile C++ to optimized C++",
      [](OpPassManager &pm, const ScaleFlowPyTorchPipelineOptions &opts) {
        // // Affine loop dataflowing.
        // pm.addPass(llmhls::createCreateDataflowFromAffinePass());
        // pm.addPass(llmhls::createStreamDataflowTaskPass());
        // pm.addPass(mlir::createCanonicalizerPass());

        // if (opts.debugPoint == 4)
        //   return;

        // Affine loop fusion.
        pm.addPass(llmhls::createFuncPreprocessPass(opts.hlsTopFunc));
        pm.addPass(llmhls::createAffineLoopFusionPass(opts.fusionTolerance));
        llmhls::addSimplifyAffineLoopPasses(pm);
        llmhls::addCreateSubviewPasses(pm);
        pm.addPass(llmhls::createRaiseAffineToCopyPass());
        pm.addPass(llmhls::createSimplifyCopyPass());
        pm.addPass(llmhls::createLowerCopyToAffinePass());
        pm.addPass(memref::createFoldMemRefAliasOpsPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 5)
          return;

        // Place dataflow buffers.
        pm.addPass(
            llmhls::createPlaceDataflowBufferPass(opts.placeExternalBuffer));

        // if (opts.vectorize) {
        //   pm.addPass(mlir::createSuperVectorizePass({2}));
        //   pm.addPass(mlir::createCanonicalizerPass());
        // }

        if (opts.debugPoint == 6)
          return;

        // Affine loop tiling.
        pm.addPass(llmhls::createFuncPreprocessPass(opts.hlsTopFunc));
        // pm.addPass(bufferization::createBufferLoopHoistingPass());
        pm.addPass(llmhls::createAffineLoopPerfectionPass());
        pm.addPass(llmhls::createRemoveVariableBoundPass());
        pm.addPass(llmhls::createAffineLoopOrderOptPass());
        // pm.addPass(llmhls::createAffineLoopTilePass(opts.loopTileSize));
        pm.addPass(mlir::createSimplifyAffineStructuresPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 7)
          return;

        // // Local buffer allocation.
        // llmhls::addCreateSubviewPasses(pm);
        // pm.addPass(llmhls::createCreateLocalBufferPass());
        // pm.addPass(llmhls::createLowerCopyToAffinePass());
        // pm.addPass(memref::createFoldMemRefAliasOpsPass());
        // pm.addPass(mlir::createSimplifyAffineStructuresPass());
        // pm.addPass(mlir::createCanonicalizerPass());

        // if (opts.debugPoint == 8)
        //   return;

        // Affine loop dataflowing.
        pm.addPass(llmhls::createCollapseMemrefUnitDimsPass());
        pm.addPass(llmhls::createAffineStoreForwardPass());
        pm.addPass(llmhls::createCreateDataflowFromAffinePass());
        pm.addPass(llmhls::createStreamDataflowTaskPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 9)
          return;

        // Lower and optimize dataflow.
        pm.addPass(llmhls::createLowerDataflowPass());
        pm.addPass(llmhls::createEliminateMultiProducerPass());
        pm.addPass(llmhls::createEliminateMultiConsumerPass());
        // pm.addPass(llmhls::createScheduleDataflowNodePass());
        // pm.addPass(llmhls::createBalanceDataflowNodePass());
        // pm.addPass(llmhls::createLowerCopyToAffinePass());
        // pm.addPass(llmhls::createAffineStoreForwardPass());
        // pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 10)
          return;

        // Parallelize dataflow.
        if (opts.loopUnrollFactor) {
          pm.addPass(llmhls::createParallelizeDataflowNodePass(
              opts.loopUnrollFactor, /*unrollPointLoopOnly=*/true,
              opts.complexityAware, opts.correlationAware));
          pm.addPass(mlir::createSimplifyAffineStructuresPass());
          pm.addPass(mlir::createCanonicalizerPass());
        }

        if (opts.debugPoint == 11)
          return;

        // Memory optimization.
        pm.addPass(llmhls::createSimplifyAffineIfPass());
        pm.addPass(llmhls::createAffineStoreForwardPass());
        pm.addPass(llmhls::createReduceInitialIntervalPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 12)
          return;

        // Convert dataflow to func.
        pm.addPass(llmhls::createCreateTokenStreamPass());
        pm.addPass(llmhls::createConvertDataflowToFuncPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 13)
          return;

        // Directive-level optimization.
        if (opts.axiInterface)
          pm.addPass(llmhls::createCreateAxiInterfacePass(opts.hlsTopFunc));
        pm.addPass(llmhls::createLoopPipeliningPass());
        pm.addPass(llmhls::createArrayPartitionPass());
        pm.addPass(llmhls::createCreateHLSPrimitivePass());
        pm.addPass(mlir::createCanonicalizerPass());
      });
}

void llmhls::registerTransformsPasses() {
  registerLLMHLSDSEPipeline();
  registerScaleFlowPyTorchPipeline();
  registerScaleFlowPyTorchPipelinePost();
  registerScaleFlowCppPipeline();
  registerPasses();
}
