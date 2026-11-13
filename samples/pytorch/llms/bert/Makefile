#!/bin/bash
BUDDY_BUILD_DIR := /home/dahaoming/buddy-mlir/build/
BUDDY_OPT := ${BUDDY_BUILD_DIR}/bin/buddy-opt
LLMHLS_BUILD_DIR := /home/dahaoming/llmhls-mlir/build/
LLMHLS_OPT := ${BUDDY_BUILD_DIR}/bin/llmhls-opt

linalg-w8only:
	@${BUDDY_OPT} ./subgraph0.mlir \
		-pass-pipeline \
			"builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith, canonicalize))" -o ./bertlayer-w8only.mlir

linalg-w4only:
	@${BUDDY_OPT} ./subgraph0.mlir \
		-pass-pipeline \
			"builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" -o ./bertlayer-w4only.mlir

linalg-w8a8:
	@${BUDDY_OPT} ./subgraph0.mlir \
		-pass-pipeline \
			"builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" -o ./bertlayer-w8a8.mlir

clear:
	rm *.cpp hls.mlir subgraph0.mlir
