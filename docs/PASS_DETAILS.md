# Pass Detail
Based on `docs/ARCHITECTURE.md`, This doc will further illustrate details of several key passes. Details of other more passes will be documented here if needed.

## llmhls-uplift-reshapes-between-tasks
Located at Stage 2, codes can be seen in `lib/Transforms/Dataflow/UpliftReshapesBetweenTasks.cpp`.

This pass is dedicated to eliminating shape conversion between HLS dataflow tasks, since it's hard and performance-unfriendly to code shape conversion in HLS C++.

For example, here's an HLS Dataflow Task IR pattern:
```
%1 = hls.dataflow.tasks {
    %21 = linalg.matmul (...) {...} -> <16x768xf16>
    hls.dataflow.yield %21 : tensor<16x768xf16>
}

%2 = hls.dataflow.tasks {
    %22 = linalg.matmul (...) {...} -> <16x768xf16>
    hls.dataflow.yield %22 : tensor<16x768xf16>
}

%3 = hls.dataflow.tasks {
    %23 = tensor.expand_shape %1 [[0], [1, 2]] : tensor<16x768xf16> into tensor<16x12x64xf16>
    %24 = tensor.expand_shape %2 [[0], [1, 2]] : tensor<16x768xf16> into tensor<16x12x64xf16>
    %empty = tensor.empty() : tensor<12x16x64xf16>
    %25 = linalg.transpose ins(%23 : tensor<16x12x64xf16>) outs(%empty : tensor<12x16x64xf16>) permutation = [1, 0, 2]
    %empty_1 = tensor.empty() : tensor<12x64x16xf16>
    %26 = linalg.transpose ins(%24 : tensor<16x12x64xf16>) outs(%empty : tensor<12x64x16xf16>) permutation = [1, 2, 0]
    %27 = linalg.batch_matmul (%25, %26) {...} -> tensor<12x16x16xf16>
}
```
As you can see, `%1` and `%2` respectively produces a result of shape `16x768` while `%3` consumes these two tensors in shape `12x16x64` and `12x64x16`. Since I don't want to carry these shape conversions all the way down to HLS C++ Codegen, this pass is designed to eliminate these conversions by uplift all shape conversion ops(including transpose) to the tail of theirs corresponding producer tasks.
```
%1 = hls.dataflow.tasks {
    %21 = linalg.matmul (...) {...} -> <16x768xf16>
    %23 = tensor.expand_shape %21 [[0], [1, 2]] : tensor<16x768xf16> into tensor<16x12x64xf16>
    %empty = tensor.empty() : tensor<12x16x64xf16>
    %25 = linalg.transpose ins(%23 : tensor<16x12x64xf16>) outs(%empty : tensor<12x16x64xf16>) permutation = [1, 0, 2]
    hls.dataflow.yield %25 : tensor<12x16x64xf16>
}

%2 = hls.dataflow.tasks {
    %22 = linalg.matmul (...) {...} -> <16x768xf16>
    %24 = tensor.expand_shape %22 [[0], [1, 2]] : tensor<16x768xf16> into tensor<16x12x64xf16>
    %empty_1 = tensor.empty() : tensor<12x64x16xf16>
    %26 = linalg.transpose ins(%24 : tensor<16x12x64xf16>) outs(%empty_1 : tensor<12x64x16xf16>) permutation = [1, 2, 0]
    hls.dataflow.yield %26 : tensor<12x64x16xf16>
}

%3 = hls.dataflow.tasks {
    %27 = linalg.batch_matmul (%1, %2) {...} -> tensor<12x16x16xf16>
}
```
Finally, by setting correct indexing maps, these conversions can be fused into a `linalg.generic` op with their matmul-like parents(Ops like matmul and conv can be flexibly represented by `linalg.generic` and shape conversions are mostly about setting correct indexing maps).
```
%1 = hls.dataflow.tasks {
    %21 = linalg.generic  {indexing_maps = [...], iterator_types = [...]} ins(...) outs(...) 
    {
        ...
    }  -> tensor<12x16x64xf16>
    hls.dataflow.yield %21 : tensor<12x16x64xf16>
}

%2 = hls.dataflow.tasks {
    %22 = linalg.generic  {indexing_maps = [...], iterator_types = [...]} ins(...) outs(...) 
    {
        ...
    }  -> tensor<12x16x64xf16>
    hls.dataflow.yield %22 : tensor<12x64x16xf16>
}

%3 = hls.dataflow.tasks {
    %27 = linalg.batch_matmul (%1, %2) {...} -> tensor<12x16x16xf16>
}
```

## llmhls-grand-fusion
Located at Stage 2 right after `llmhls-uplift-reshapes-between-tasks` pass. Typically during this pass, a HLS Dataflow task is centered at a matmul-like op(matmul, conv, reduction type op, etc.), where most of the computation overhead lies. Before this matmul-like op go a series of element-wise or broadcast-type ops, after it follow shape conversion ops uplifted during `llmhls-uplift-reshapes-between-tasks` pass. Within a dataflow task, they can all be fused into a `linalg.generic` op(also, can be represented as a nested loop). With this fusion at linalg level, another buggy `affine.for` level loop fusion int stage 5 can be avoided.

## llmhls-legalize-dataflow
Located at Stage 11, codes can be seen in `lib/Transforms/Dataflow/LegalizeDataflow.cpp`.

A legal HLS dataflow should guarantee that there's no multi-consumer nodes in it. For HLS C++, a multi-consumer node can show up in 2 forms:
Form 1:
```
node4(..., output: x)
node3(x, ..., output: y)
node2(x, ..., output: z)
node1(x, ..., output: w)
```
In this case, the result of `node4` has 3 consumers `node1~3`, Vitis HLS compiler will refuse to generate corresponding RTL design if there's any multi-consumer nodes. Thus, this pass will merge `node1~3` into one:
```
node0 (x, ..., output: y, z, w) {
    node3(x, ..., output: y)
    node2(x, ..., output: z)
    node1(x, ..., output: w)
}

node4(..., output: x)
node0(x, ..., output: y, z, w)
```

Form 2:
```
node4(..., output: x)
node3(x, ..., output: y)
node2(y, ..., output: z)
node1(x, y, z, ..., output: w)
```
In this case, `node1` and `node3` are defined as bypass nodes for value x. In the previous case, `node1~3` are on the same execution level and can be run in parallel. But in this one, there is data dependency between `node1` and `node3`, plus here's another node standing between them. so under this case, the pass will firstly merge `node1~3` even though `node2` is not consumer of `x`:
```
node4(..., x)
node0(x, output: w) {
    node3(x, ..., output: y)
    node2(y, ..., output: z)
    node1(x, y, z, ..., output: w)
}
```
Plus, `y` is used by `node2` and `node1`, so in merged `node0` another merge will happen:
```
node4(..., x)
node0(x, output: w) {
    node3(x, ..., output: y)
    node00(x, y, ..., output: w)
    {
        node2(y, ..., output: z)
        node1(x, y, z, ..., output: w)
    }
}
```
The `llmhls-legalize-dataflow` pass is to eliminate every multi-consumer or bypass nodes across the whole HLS workload. By the way, `llmhls-eliminate-multi-consumer` pass in Stage 10 only generates multiple copies for multi-consumer nodes instead of actually eliminating them. Furthermore, it does nothing to bypass nodes.
