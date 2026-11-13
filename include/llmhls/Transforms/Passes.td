//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The LLMHLS Authors.
//
//===----------------------------------------------------------------------===//

#ifndef LLMHLS_TRANSFORMS_PASSES_TD
#define LLMHLS_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def DesignSpaceExplore : Pass<"llmhls-dse", "ModuleOp"> {
  let summary = "Optimize HLS design at multiple abstraction level";
  let description = [{
    This pass will automatically conduct the design space exploration (DSE)
    across multiple abstraction levels. By calling methods provided by
    qor-estimation, this pass is able to rapidly obtain the QoR estimation of
    the current design point, and feed it back to the DSE engine for an
    efficient optimization convergence.
  }];
  let constructor = "mlir::llmhls::createDesignSpaceExplorePass()";

  let options = [
    Option<"outputPath", "output-path", "std::string",
           /*default=*/"\"./\"",
           "File path: the path for dumping the MLIR of pareto design points">,
    Option<"csvPath", "csv-path", "std::string",
           /*default=*/"\"./\"",
           "File path: the path for dumping the CSV of design spaces">,

    Option<"targetSpec", "target-spec", "std::string",
           /*default=*/"\"./config.json\"",
           "File path: target backend specifications and configurations">
  ];
}

def FuncDuplication : Pass<"llmhls-func-duplication", "mlir::ModuleOp"> {
  let summary = "Duplicate function for each function call";
  let constructor = "mlir::llmhls::createFuncDuplicationPass()";
}

def FuncPreprocess : Pass<"llmhls-func-preprocess", "func::FuncOp"> {
  let summary = "Preprocess the functions subsequent LLMHLS optimizations";
  let constructor = "mlir::llmhls::createFuncPreprocessPass()";

  let options = [
    Option<"topFunc", "top-func", "std::string", /*default=*/"\"main\"",
           "The top function for HLS synthesis">
  ];
}

//===----------------------------------------------------------------------===//
// Dataflow-related Passes
//===----------------------------------------------------------------------===//

def BalanceDataflowNode :
      Pass<"llmhls-balance-dataflow-node", "func::FuncOp"> {
  let summary = "Balance dataflow nodes";
  let constructor = "mlir::llmhls::createBalanceDataflowNodePass()";
}

def BufferizeDataflow : Pass<"llmhls-bufferize-dataflow", "func::FuncOp"> {
  let summary = "Bufferize dataflow operations";
  let constructor = "mlir::llmhls::createBufferizeDataflowPass()";
}

def ConvertDataflowToFunc :
      Pass<"llmhls-convert-dataflow-to-func", "ModuleOp"> {
  let summary = "Convert dataflow to function dialect";
  let constructor = "mlir::llmhls::createConvertDataflowToFuncPass()";

  let options = [
    Option<"splitExternalAccess", "split-external-access", "bool",
           /*default=*/"true", "whether split external memory accesses">
  ];
}

def CreateDataflowFromAffine :
      Pass<"llmhls-create-dataflow-from-affine", "func::FuncOp"> {
  let summary = "Create dataflow hierarchy from affine loops";
  let constructor = "mlir::llmhls::createCreateDataflowFromAffinePass()";
}

def CreateDataflowFromLinalg :
      Pass<"llmhls-create-dataflow-from-linalg", "func::FuncOp"> {
  let summary = "Create dataflow hierarchy from linalg";
  let constructor = "mlir::llmhls::createCreateDataflowFromLinalgPass()";
  let options = [
    Option<"greedyFuse", "greedy-fuse", "bool",
           /*default=*/"true", "Whether to fuse ops into a dataflow task in a greedy pattern">
  ];
}

def CreateDataflowFromTosa :
      Pass<"llmhls-create-dataflow-from-tosa", "func::FuncOp"> {
  let summary = "Create dataflow hierarchy from tosa";
  let constructor = "mlir::llmhls::createCreateDataflowFromTosaPass()";
}

def CreateTokenStream : Pass<"llmhls-create-token-stream", "func::FuncOp"> {
  let summary = "Create token stream channels for DRAM buffers";
  let constructor = "mlir::llmhls::createCreateTokenStreamPass()";
}

def BufferDuplicate : Pass<"llmhls-buffer-duplicate", "func::FuncOp"> {
  let summary = "Duplicate output buffers for every dataflow node consumer";
  let description = [{
    This pass eliminates multi-consumer and bypass dependencies between the
    outermost dataflow nodes by giving each consumer a distinct buffer.
  }];
  let constructor = "mlir::llmhls::createBufferDuplicatePass()";
}

def EliminateMultiConsumer :
      Pass<"llmhls-eliminate-multi-consumer", "func::FuncOp"> {
  let summary = "Eliminate multi-consumer violations";
  let description = [{
    This pass will insert explicit fork nodes to eliminate multi-consumers.
  }];
  let constructor = "mlir::llmhls::createEliminateMultiConsumerPass()";
}

def EliminateMultiProducer :
      Pass<"llmhls-eliminate-multi-producer", "func::FuncOp"> {
  let summary = "Try to eliminate multi-producer violations";
  let constructor = "mlir::llmhls::createEliminateMultiProducerPass()";
}

def LegalizeDataflow : Pass<"llmhls-legalize-dataflow", "func::FuncOp"> {
  let summary = "Legalize dataflow by merging dataflow nodes";
  let constructor = "mlir::llmhls::createLegalizeDataflowPass()";
}

def LowerDataflow : Pass<"llmhls-lower-dataflow", "func::FuncOp"> {
  let summary = "Lower dataflow from task level to node level";
  let constructor = "mlir::llmhls::createLowerDataflowPass()";

  let options = [
    Option<"splitExternalAccess", "split-external-access", "bool",
           /*default=*/"true", "whether split external memory accesses">
  ];
}

def ParallelizeDataflowNode :
      Pass<"llmhls-parallelize-dataflow-node", "func::FuncOp"> {
  let summary = "Unroll affine loop nests based on the dataflow structure";
  let description = [{
    This pass calculates the overall loop unroll factor of each dataflow node
    based on the amount of associated computations. Then, unroll and jam from
    the outermost loop until the overall unroll factor reaches the caculated
    factor. Optionally, optimize the loop order after the unrolling.
  }];
  let constructor = "mlir::llmhls::createParallelizeDataflowNodePass()";

  let options = [
    Option<"maxUnrollFactor", "max-unroll-factor", "unsigned", /*default=*/"1",
           "Positive number: the maximum factor of unrolling">,
    Option<"pointLoopOnly", "point-loop-only", "bool", /*default=*/"false",
           "Only apply unroll and jam to point loop band">,
    Option<"complexityAware", "complexity-aware", "bool", /*default=*/"true",
           "Whether to consider node complexity in the transform">,
    Option<"correlationAware", "correlation-aware", "bool", /*default=*/"true",
           "Whether to consider node correlation in the transform">
  ];
}

def PlaceDataflowBuffer :
      Pass<"llmhls-place-dataflow-buffer", "func::FuncOp"> {
  let summary = "Place dataflow buffers";
  let constructor = "mlir::llmhls::createPlaceDataflowBufferPass()";

  let options = [
    Option<"placeExternalBuffer", "place-external-buffer", "bool",
           /*default=*/"false", "Place buffers in external buffers">,
    Option<"placeTopFuncArgsOnDram", "place-top-func-args-on-dram", "bool",
           /*default=*/"false", "Whether to place top func args on DRAM">
  ];
}

def ScheduleDataflowNode :
      Pass<"llmhls-schedule-dataflow-node", "func::FuncOp"> {
  let summary = "Schedule dataflow nodes";
  let constructor = "mlir::llmhls::createScheduleDataflowNodePass()";

  let options = [
    Option<"ignoreViolations", "ignore-violations", "bool",
           /*default=*/"false", "Ignore multi-consumer or producer violations">
  ];
}

def StreamDataflowTask : Pass<"llmhls-stream-dataflow-task", "func::FuncOp"> {
  let summary = "Stream dataflow tasks";
  let constructor = "mlir::llmhls::createStreamDataflowTaskPass()";
}

def UpliftReshapesBetweenTasks :
      Pass<"llmhls-uplift-reshapes-between-tasks", "func::FuncOp"> {
  let summary = "Uplift reshape-type operations between dataflow tasks";
  let constructor = "mlir::llmhls::createUpliftReshapesBetweenTasksPass()";
}

def GrandFusion :
      Pass<"llmhls-grand-fusion", "func::FuncOp"> {
  let summary = "Fuse all trivial ops within each task into one linalg.generic op";
  let constructor = "mlir::llmhls::createGrandFusionPass()";
}

//===----------------------------------------------------------------------===//
// Tensor-related Passes
//===----------------------------------------------------------------------===//

def ConvertTensorToLinalg :
      Pass<"llmhls-convert-tensor-to-linalg", "func::FuncOp"> {
  let summary = "Lower tosa::ReshapeOp and tensor::PadOp";
  let constructor = "mlir::llmhls::createConvertTensorToLinalgPass()";
}

def LinalgAnalyzeModel : Pass<"llmhls-linalg-analyze-model", "func::FuncOp"> {
  let summary = "Analyze the operation number of a linalg model";
  let constructor = "mlir::llmhls::createLinalgAnalyzeModelPass()";
}

def LinalgFakeQuantize : Pass<"llmhls-linalg-fake-quantize", "ModuleOp"> {
  let summary = "Convert to quantized model (only for testing use)";
  let constructor = "mlir::llmhls::createLinalgFakeQuantizePass()";

  let options = [
    Option<"quanBits", "quan-bits", "unsigned", /*default=*/"8",
           "the number of bits for quantization">
  ];
}

def NormalizeLinalgGenericConstants :
      Pass<"llmhls-normalize-linalg-generic-constants", "func::FuncOp"> {
  let summary = "Normalize shaped constants around linalg.generic operations";
  let description = [{
    Replace shaped output constants with tensor.empty plus linalg.fill, and
    inline splat tensor inputs as scalar constants in linalg.generic bodies.
  }];
  let constructor =
      "mlir::llmhls::createNormalizeLinalgGenericConstantsPass()";
}

def TosaFakeQuantize : Pass<"llmhls-tosa-fake-quantize", "ModuleOp"> {
  let summary = "Convert to 8-bits quantized model (only for testing use)";
  let constructor = "mlir::llmhls::createTosaFakeQuantizePass()";
}

def TosaSimplifyGraph : Pass<"llmhls-tosa-simplify-graph", "func::FuncOp"> {
  let summary = "Remove redundant TOSA operations";
  let description = [{
    This simplify-tosa-graph pass will try to remove redundant transpose ops
    through pattern matching.
  }];
  let constructor = "mlir::llmhls::createTosaSimplifyGraphPass()";
}

//===----------------------------------------------------------------------===//
// Loop-related Passes
//===----------------------------------------------------------------------===//

def AffineLoopFusion : Pass<"llmhls-affine-loop-fusion", "func::FuncOp"> {
  let summary = "Fuse affine loop nests";
  let description = [{
    This pass performs fusion of loop nests using a slicing-based approach. It
    combines two fusion strategies: producer-consumer fusion and sibling fusion.
    Producer-consumer fusion is aimed at fusing pairs of loops where the first
    one writes to a memref that the second reads. Sibling fusion targets pairs
    of loops that share no dependences between them but that load from the same
    memref. The fused loop nests, when possible, are rewritten to access
    significantly smaller local buffers instead of the original memref's, and
    the latter are often either completely optimized away or contracted. This
    transformation leads to enhanced locality and lower memory footprint through
    the elimination or contraction of temporaries/intermediate memref's. These
    benefits are sometimes achieved at the expense of redundant computation
    through a cost model that evaluates available choices such as the depth at
    which a source slice should be materialized in the designation slice.
  }];
  let constructor = "mlir::llmhls::createAffineLoopFusionPass()";

  let options = [
    Option<"computeToleranceThreshold", "fusion-compute-tolerance", "double",
           /*default=*/"0.30f", "Fractional increase in additional computation "
                                "tolerated while fusing">,
    Option<"fastMemorySpace", "fusion-fast-mem-space", "unsigned",
           /*default=*/"0",
           "Faster memory space number to promote fusion buffers to">,
    Option<"localBufSizeThreshold", "fusion-local-buf-threshold", "uint64_t",
           /*default=*/"0", "Threshold size (KiB) for promoting local buffers "
                            "to fast memory space">,
    Option<"maximalFusion", "fusion-maximal", "bool", /*default=*/"false",
           "Enables maximal loop fusion">,
    Option<"affineFusionMode", "mode", "enum AffineFusionMode",
           "AffineFusionMode::Greedy", "fusion mode to attempt",
           "llvm::cl::values(clEnumValN(AffineFusionMode::Greedy,"
           " \"greedy\", \"Perform greedy fusion\"), "
           "clEnumValN( AffineFusionMode::ProducerConsumer, "
           "\"producer\", \"Perform only producer-consumer fusion\"), "
           "clEnumValN( AffineFusionMode::Sibling, "
           "\"sibling\", \"Perform only sibling fusion\"))">,
    ];
}

def AffineLoopOrderOpt :
      Pass<"llmhls-affine-loop-order-opt", "func::FuncOp"> {
  let summary = "Optimize the order of affine loop nests";
  let description = [{
    This pass will optimize the order of perfect affine loop nests through
    polyhedral-based dependency analysis. The objective of this pass is to
    maximize the distance of loop-carried dependencies.
  }];
  let constructor = "mlir::llmhls::createAffineLoopOrderOptPass()";
}

def AffineLoopPerfection :
      Pass<"llmhls-affine-loop-perfection", "func::FuncOp"> {
  let summary = "Try to perfect a nested loop";
  let description = [{
    This pass will try to perfect all affine loops. Specifically, this pass will
    move operations under non-innermost loops into innermost loop and create if
    statements to ensure the correctness of the transformation.
  }];
  let constructor = "mlir::llmhls::createAffineLoopPerfectionPass()";
}

def AffineLoopTile : Pass<"llmhls-affine-loop-tile", "func::FuncOp"> {
  let summary = "Tile affine loop nests and annotate point loops";
  let description = [{
    Apply the same "tile-size" to each affine loop in the nests.
  }];
  let constructor = "mlir::llmhls::createAffineLoopTilePass()";

  let options = [
    Option<"tileSize", "tile-size", "unsigned", /*default=*/"1",
           "Use this tile size for all loops">
  ];
}

def AffineLoopUnrollJam :
      Pass<"llmhls-affine-loop-unroll-jam", "func::FuncOp"> {
  let summary = "Unroll and jam affine loop nests";
  let description = [{
    Unroll and jam from the outermost loop until the overall unroll factor
    reaches the specified "unroll-factor". Optionally, optimize the loop order
    after the unrolling.
  }];
  let constructor = "mlir::llmhls::createAffineLoopUnrollJamPass()";

  let options = [
    Option<"unrollFactor", "unroll-factor", "unsigned", /*default=*/"1",
           "Positive number: the factor of unrolling">,
    Option<"pointLoopOnly", "point-loop-only", "bool", /*default=*/"false",
           "Only apply unroll and jam to point loop band">
  ];
}

def MaterializeReduction :
      Pass<"llmhls-materialize-reduction", "func::FuncOp"> {
  let summary = "Materialize loop reductions";
  let description = [{
    This pass will materialize loop reductions with local buffer read/writes in
    order to expose more optimization opportunities targeting HLS.
  }];
  let constructor = "mlir::llmhls::createMaterializeReductionPass()";
}

def RemoveVariableBound :
      Pass<"llmhls-remove-variable-bound", "func::FuncOp"> {
  let summary = "Try to remove variable loop bounds";
  let description = [{
    This pass will try to remove the variable loop bounds. Specifically, this is
    only feasible when the loop bound is an affine expression of induction
    variables of other loops with constant lower and upper bound.
  }];
  let constructor = "mlir::llmhls::createRemoveVariableBoundPass()";
}

//===----------------------------------------------------------------------===//
// Memory-related Passes
//===----------------------------------------------------------------------===//

def AffineStoreForward : Pass<"llmhls-affine-store-forward", "func::FuncOp"> {
  let summary = "Forward store to load, including conditional stores";
  let description = [{
    This pass is similar to scal-repl, but supports to forward stores located in
    if statements.
  }];
  let constructor = "mlir::llmhls::createAffineStoreForwardPass()";
}

def CollapseMemrefUnitDims :
      Pass<"llmhls-collapse-memref-unit-dims", "func::FuncOp"> {
  let summary = "Collapse memref's unit dimensions";
  let constructor = "mlir::llmhls::createCollapseMemrefUnitDimsPass()";
}

def CreateLocalBuffer : Pass<"llmhls-create-local-buffer", "func::FuncOp"> {
  let summary = "Promote external buffer to on-chip buffer";
  let constructor = "mlir::llmhls::createCreateLocalBufferPass()";

  let options = [
    Option<"externalBufferOnly", "external-buffer-only", "bool",
           /*default=*/"true", "only handle external buffers">,
    Option<"registerOnly", "register-only", "bool",
           /*default=*/"false", "only registers or single-element buffers">
  ];
}

def CreateMemrefSubview :
      Pass<"llmhls-create-memref-subview", "func::FuncOp"> {
  let summary = "Create subviews based on loop analysis";
  let description = [{
    Through loop analysis, this pass can identify the memory partition that each
    sub-function is accessing. Then, by creating subview operations, the program
    in each sub-function can access the memory subview rather than the original
    memory.
  }];
  let constructor = "mlir::llmhls::createCreateMemrefSubviewPass()";

  let options = [
    Option<"createSubviewMode", "mode", "enum CreateSubviewMode",
           /*default=*/"CreateSubviewMode::Point",
           "loop band mode to create subviews", "llvm::cl::values("
           "clEnumValN(CreateSubviewMode::Point, \"point\", "
           "\"Create subviews on point loop band\"), "
           "clEnumValN(CreateSubviewMode::Reduction, \"reduction\", "
           "\"Create subviews on reduction loop band\"))">
  ];
}

def LowerCopyToAffine : Pass<"llmhls-lower-copy-to-affine", "func::FuncOp"> {
  let summary = "Convert copy and assign to affine loops";
  let constructor = "mlir::llmhls::createLowerCopyToAffinePass()";

  let options = [
    Option<"internalCopyOnly", "internal-copy-only", "bool",
           /*default=*/"false", "only convert copy between internal buffers">
  ];
}

def RaiseAffineToCopy : Pass<"llmhls-raise-affine-to-copy", "func::FuncOp"> {
  let summary = "Raise copy in affine loops to memref.copy";
  let constructor = "mlir::llmhls::createRaiseAffineToCopyPass()";
}

def ReduceInitialInterval :
      Pass<"llmhls-reduce-initial-interval", "func::FuncOp"> {
  let summary = "Try to reduce the intiail interval";
  let description = [{
    This pass try to reduce the II by optimizing the commutative operator trees
    and iteration variables.
  }];
  let constructor = "mlir::llmhls::createReduceInitialIntervalPass()";
}

def SimplifyAffineIf : Pass<"llmhls-simplify-affine-if", "func::FuncOp"> {
  let summary = "Simplify affine if operations";
  let description = [{
    This pass will eliminate all redundant affine if operations which will
    always take then or else branch, and merge all affine if operations that
    having the same statement.
  }];
  let constructor = "mlir::llmhls::createSimplifyAffineIfPass()";
}

def SimplifyCopy : Pass<"llmhls-simplify-copy", "func::FuncOp"> {
  let summary = "Simplify memref copy ops";
  let constructor = "mlir::llmhls::createSimplifyCopyPass()";
}

//===----------------------------------------------------------------------===//
// Directive-related Passes
//===----------------------------------------------------------------------===//

def ArrayPartition : Pass<"llmhls-array-partition", "ModuleOp"> {
  let summary = "Apply optimized array partition strategy";
  let description = [{
    This pass will automatically search for the best array partition solution
    for each on-chip memory instance and apply the solution through changing the
    layout of the corresponding memref.
  }];
  let constructor = "mlir::llmhls::createArrayPartitionPass()";
}

def CreateHLSPrimitive : Pass<"llmhls-create-hls-primitive", "func::FuncOp"> {
  let summary = "Create HLS C++ multiplification primitives";
  let description = [{
    This pass will convert 8-bits multiplifications to HLS C++ primitives in
    order to utilize DSP instances in FPGA.
  }];
  let constructor = "mlir::llmhls::createCreateHLSPrimitivePass()";
}

def CreateAxiInterface : Pass<"llmhls-create-axi-interface", "ModuleOp"> {
  let summary = "Create AXI interfaces for the top function";
  let description = [{
    This pass will create a new "main" function calling the original top
    function. All constant tensors are instantiated in the new "main" function
    and passed into the original top function as arguments after the transform.
  }];
  let constructor = "mlir::llmhls::createCreateAxiInterfacePass()";

  let options = [
    Option<"topFunc", "top-func", "std::string", /*default=*/"\"main\"",
           "The top function for HLS synthesis">
  ];
}

def FuncPipelining : Pass<"llmhls-func-pipelining", "func::FuncOp"> {
  let summary = "Apply function pipelining";
  let description = [{
    This pass will insert pipeline pragma to the specified function, all
    contained loops will be automatically unrolled.
  }];
  let constructor = "mlir::llmhls::createFuncPipeliningPass()";

  let options = [
    Option<"targetFunc", "target-func", "std::string",
           /*default=*/"\"main\"", "The target function to be pipelined">,
    Option<"targetII", "target-ii", "unsigned", /*default=*/"1",
           "Positive number: the targeted II to achieve">
  ];
}

def LoopPipelining : Pass<"llmhls-loop-pipelining", "func::FuncOp"> {
  let summary = "Apply loop pipelining";
  let description = [{
    This pass will insert pipeline pragma to the target loop level, and
    automatically unroll all inner loops.
  }];
  let constructor = "mlir::llmhls::createLoopPipeliningPass()";

  let options = [
    Option<"pipelineLevel", "pipeline-level", "unsigned", /*default=*/"0",
           "Positive number: loop level to be pipelined (from innermost)">,
    Option<"targetII", "target-ii", "unsigned", /*default=*/"1",
           "Positive number: the targeted II to achieve">
  ];
}

def LowerAffine : Pass<"llmhls-lower-affine", "func::FuncOp"> {
  let summary = "Lower AffineSelectOp and AffineForOp";
  let constructor = "mlir::llmhls::createLowerAffinePass()";
}

def QoREstimation : Pass<"llmhls-qor-estimation", "ModuleOp"> {
  let summary = "Estimate the performance and resource utilization";
  let description = [{
    This pass will analyze the input IR and estimate the latency and resource
    utilization of HLS C++ synthesis. This pass will take all dependency and
    resource constraints and pragma settings into consideration, and conduct the
    estimation through an ALAP scheduling.
  }];
  let constructor = "mlir::llmhls::createQoREstimationPass()";

  let options = [
    Option<"targetSpec", "target-spec", "std::string",
           /*default=*/"\"./config.json\"",
           "File path: target backend specifications and configurations">
  ];
}

#endif // LLMHLS_TRANSFORMS_PASSES_TD
