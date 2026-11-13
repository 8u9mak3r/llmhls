get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

################################################################################
# Set up Python binding tools
################################################################################

include(AddMLIRPython)

################################################################################
# Declare native Python extension
################################################################################

declare_mlir_python_sources(LLMHLSBindingsPythonExtension)

declare_mlir_python_extension(LLMHLSBindingsPythonExtension.Core
  MODULE_NAME _llmhls
  ADD_TO_PARENT LLMHLSBindingsPythonExtension
  SOURCES
    LLMHLSModule.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRLLMHLSCAPIHLS
    MLIRLLMHLSCAPIEmitHLSCpp
    MLIRCAPITransforms
  PRIVATE_LINK_LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${translation_libs}
    LLVMSupport
    MLIRLLMHLSTransforms
)

################################################################################
# Declare Python sources
################################################################################

declare_mlir_python_sources(LLMHLSBindingsPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    llmhls/__init__.py
    llmhls/dialects/_ods_common.py
)

################################################################################
# Declare dialect-specific bindings.
################################################################################

# Ensure the build directory for generated Python files exists. Ninja is able to
# generate this, but make does not and the build fails.
file(MAKE_DIRECTORY ${LLMHLS_BINARY_DIR}/lib/Bindings/Python/llmhls/dialects)

declare_mlir_python_sources(LLMHLSBindingsPythonSources.Dialects
  ADD_TO_PARENT LLMHLSBindingsPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT LLMHLSBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE llmhls/dialects/HLSOps.td
  SOURCES
    llmhls/dialects/hls.py
  DIALECT_NAME hls)

################################################################################
# Build composite binaries
################################################################################

# Bundle our own, self-contained CAPI library with all of our deps.
add_mlir_python_common_capi_library(LLMHLSBindingsPythonCAPI
  INSTALL_COMPONENT LLMHLSBindingsPythonModules
  INSTALL_DESTINATION python_packages/llmhls_core/mlir/_mlir_libs
  # NOTE: When the MLIR API is relocated under llmhls, this would change to
  # .../llmhls/_mlir_libs
  OUTPUT_DIRECTORY "${LLMHLS_PYTHON_PACKAGES_DIR}/llmhls_core/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects
    MLIRPythonSources.ExecutionEngine
    LLMHLSBindingsPythonSources
    LLMHLSBindingsPythonExtension
)

# Bundle the MLIR python sources into our package.
# The MLIR API is position independent, so we explicitly output it to the mlir/
# folder as a temporary measure. It will eventually migrate under the llmhls/
# folder and be accessible under the unified "import llmhls..." namespace.
add_mlir_python_modules(LLMHLSMLIRPythonModules
  ROOT_PREFIX "${LLMHLS_PYTHON_PACKAGES_DIR}/llmhls_core/mlir"
  INSTALL_PREFIX "python_packages/llmhls_core/mlir"
  DECLARED_SOURCES
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects
    MLIRPythonSources.ExecutionEngine
    # We need the llmhls extensions co-located with the MLIR extensions. When
    # the namespace is unified, this moves to the below.
    LLMHLSBindingsPythonExtension
  COMMON_CAPI_LINK_LIBS
    LLMHLSBindingsPythonCAPI
)

# Bundle the LLMHLS python sources into our package.
add_mlir_python_modules(LLMHLSPythonModules
  ROOT_PREFIX "${LLMHLS_PYTHON_PACKAGES_DIR}/llmhls_core"
  INSTALL_PREFIX "python_packages/llmhls_core"
  DECLARED_SOURCES
    LLMHLSBindingsPythonSources
  COMMON_CAPI_LINK_LIBS
    LLMHLSBindingsPythonCAPI
)
