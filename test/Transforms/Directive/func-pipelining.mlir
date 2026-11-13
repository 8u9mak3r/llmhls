// RUN: llmhls-opt -llmhls-func-pipelining="target-func=test" %s | FileCheck %s

// CHECK-LABEL: func.func @test
func.func @test() {
  return
}
