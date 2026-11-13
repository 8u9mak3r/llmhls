llvm_canonicalize_cmake_booleans(
  LLMHLS_ENABLE_BINDINGS_PYTHON
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(LLMHLS_TEST_DEPENDS
  FileCheck count not
  pyllmhls
  llmhls-opt
  llmhls-translate
  )

if(LLMHLS_ENABLE_BINDINGS_PYTHON)
  list(APPEND LLMHLS_TEST_DEPENDS LLMHLSMLIRPythonModules)
  list(APPEND LLMHLS_TEST_DEPENDS LLMHLSPythonModules)
endif()

add_lit_testsuite(check-llmhls "Running the llmhls regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${LLMHLS_TEST_DEPENDS}
  )
set_target_properties(check-llmhls PROPERTIES FOLDER "Tests")

add_lit_testsuites(LLMHLS
  ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${LLMHLS_TEST_DEPENDS}
  )
