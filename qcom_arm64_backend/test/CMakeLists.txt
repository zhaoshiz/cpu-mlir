configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TRITON_TEST_DEPENDS
  arm64-mlir-opt
)

add_lit_testsuite(check-arm64-mlir "Running the arm64 regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TRITON_TEST_DEPENDS}
  )

set_target_properties(check-arm64-mlir PROPERTIES FOLDER "Tests")
