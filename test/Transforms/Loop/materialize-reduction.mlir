// RUN: llmhls-opt -llmhls-materialize-reduction %s | FileCheck %s

// CHECK-LABEL: func.func @test
func.func @test() {
  return
}
