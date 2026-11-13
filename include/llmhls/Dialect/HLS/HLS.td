//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The LLMHLS Authors.
//
//===----------------------------------------------------------------------===//

#ifndef LLMHLS_DIALECT_HLS_HLS_TD
#define LLMHLS_DIALECT_HLS_HLS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def HLSDialect : Dialect {
  let name = "hls";
  let summary = "An HLS out-of-tree MLIR dialect";
  let description = [{
    This dialect contains dedicated operations, types, and attributes designed
    for representing HLS C++ specific structures and components in MLIR, while
    enabling comprehensive optimization for both performance and area.
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let cppNamespace = "::mlir::llmhls::hls";
}

class HLSType<string name> : TypeDef<HLSDialect, name>;

class HLSAttr<string name> : AttrDef<HLSDialect, name>;

class HLSOp<string mnemonic, list<Trait> traits = []> :
    Op<HLSDialect, mnemonic, traits>;

include "llmhls/Dialect/HLS/Attributes.td"
include "llmhls/Dialect/HLS/Types.td"
include "llmhls/Dialect/HLS/Ops.td"

#endif // LLMHLS_DIALECT_HLS_HLS_TD
