# Project Overview

This repository implements an MLIR-based compiler and automatic
HLS code generation framework for high-performance FPGA acceleration.

The main research targets are:

1. Operator fusion for HLS code generation.
2. Adaptive tiling for different neural network layers.
3. Dynamic load/compute/store pipeline construction.
4. Resource-performance co-optimization.
5. MLIR-to-HLS C++ lowering.

Target platform:
- AMD/Xilinx Alveo U55C
- Vitis HLS
- AXI4 / AXI-Lite interfaces

# Agent Workflow

For non-trivial compiler changes:

1. Understand the relevant compiler pipeline.
2. Locate the corresponding IR transformation or codegen implementation.
3. Inspect existing tests and similar implementations.
4. Establish the expected IR/code transformation.
5. Make the smallest reasonable change.
6. Build the affected target only.
7. Run relevant regression tests.
8. Inspect generated HLS C++ when code generation is affected.
9. Never claim success without verification.

# Documentation

Before modifying compiler architecture, For lowering pipeline:
- Read ARCHITECTURE.md

For pass details:
- Read docs/PASS_DETAILS.md

For HLS C++ generation:
- Read docs/HLS_CODEGEN.md

For compiler optimizations:
- Read docs/OPTIMIZATION_GUIDE.md

For performance/resource decisions:
- Read docs/PERFORMANCE_MODEL.md

For build and testing:
- Read docs/BUILD_AND_TEST.md

# General Rules

- Do not modify generated files under build/.
- Prefer MLIR/LLVM existing infrastructure over custom implementations.
- Preserve SSA dominance and type correctness.
- Do not rebuild all LLVM/MLIR unless necessary.
- Changes to transformations should include regression tests.
- Performance optimizations must consider both latency and FPGA resources.

# Some Other Temporary Rules

- Ignore regression tests of this repo, which means ignore the errors triggered by `check-llmhls` because of all the modifications I have made contaminating the original compiler pipeline. Just wait for me restoring all the passes I modified, as to the new passes I added/modified, I'll set up some opts to manually trigger them.
