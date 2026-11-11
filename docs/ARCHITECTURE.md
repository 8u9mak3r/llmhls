# Compiler Architecture

The compiler is organized into 14 major stages. Seen in `lib/Transforms/Passes.cpp`, Line 145~338.
The whole compiler pipeline still needs perfection, this doc only serves as an overview.

## Stage 1: IR Lowering

Input IRs are lowered from TOSA into Linalg.

Typical IR:

TOSA
  ->
Linalg

Since the frontend of the compiler `buddy-mlir` has guaranteed the output IRs are all linalg ops, this compiler stage can be simply skipped.

## Stage 2: Constructions of HLS Dataflows

HLS Dataflows are built during this stage, represented as `hls.dataflow`, which will be dealt as basic units of a high-performance HLS Kernel.

Major transformations:

- Dataflow Construction
- Shape Canonicalization
- Op fusion into a single `linalg.generic` within a dataflow.

## Stage 3: Bufferize

Linalg operations in tensor are converted to linalg operations in memref.

The compiler determines:

- loop order
- tile sizes
- buffer allocation
- data reuse
- pipeline structure

## Stage 4: Linalg to Affine

All the linalg ops are converted to affine for loops.

## Stage 5: Affine Loop Fusion

Since I fused every op within a HLS Dataflow into a huge `linalg.generic` op, this compiler stage is actually out of function (In fact, fusion passes in this stage have some problems, that's why I choose to generalize everything earlier).

## Stage 6: Set Dataflow Buffers

Choose where to put intermediate buffers between HLS Dataflows(BRAM/URAM/External DRAM).

## Stage 7: Affine Loop Tiling

This stage determines:

- Loop Order
- Tile Sizes

## Stage 8: Local Buffer Allocation

This stage allocates buffers for every tile blocks generated in Stage 7.

## Stage 9: Affine Loop Dataflowing.

For every local buffer allocated in stage 8, generate a load-compute-store dataflow within affine loops.

## Stage 10: Dataflow Lowering and Optimization

In this stage, original HLS dataflows `hls.dataflow` will be lowered and represented as `hls.dataflow.schedule` and `hls.dataflow.node`, which are inserted in each other. In most cases, the whole IR will be fit in a huge `hls.dataflow.schedule` and all original HLS Dataflows will be converted to `hls.dataflow.node`. Within each `hls.dataflow.node` lies a `hls.dataflow.schedule` to contain load-compute-store nodes, whose corresponding dataflow is formed in Stage 9. Moreover, this stage will eliminate all multi-producers and multi-consumer dependencies to meet the inherit restrictions of Vitis HLS C++, and set a execution level for each `hls.dataflow.node` within a certain `hls.dataflow.schedule`. A node with a larger execution level will run earlier, nodes with the same level can usually run in parallel.

## Stage 11: Dataflow Parallelization and Legalization

This stage performs loop unroll on each dataflow nodes(disabled for default) and legalize all dataflow nodes. Legalizing a dataflow node means eliminating all multi-consumers and bypass-nodes, this will be illustrated in detail in `docs/COMPILER_PIPELINE.md`.

## Stage 12: Memory Optimization.

This stage will optimize load/store ops within dataflow nodes.

## Stage 13: Convert Dataflow to Func

`hls.dataflow.schedule` and `hls.dataflow.nodes` will be converted to `func.func`.

## Stage 14: Directive-level Optimization

Codegen of high-performance HLS C++ code.

## IR Examples

Can be seen in `samples/pytorch/llms/bert/debug-w8only`, under this path lies IR files named as `debugpoint-x.mlir`(x ranges from 0 to 13) and a `directive-level.mlir`, which is the output of the last compiler stage. To inspect the lowering and optimization effects on IR of a certain compiler stage or several contiguous stages, you can run them by setting `start-point` and `debug-point` opts in shell cmds. For example, cmd `scalehls-opt debug-w8only/debugpoint-1.mlir -scaleflow-pytorch-pipeline="top-func=forward start-point=2 debug-point=10" > debug-w8only/hls-1.mlir 2> debug-w8only/dbg.log` will perform compiler stage 2~10 on the specified IR, and dump the generated IR in `debug-w8only/hls-1.mlir` and/or errors in `debug-w8only/dbg.log`.




