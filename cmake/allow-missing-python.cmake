include_guard(GLOBAL)

# kaldifst and kaldi-decoder require a Python interpreter at configure time
# even with their bindings and tests off, though without them they run one
# only on Apple. Call this macro in a dependency's download function, before
# its add_subdirectory(), so a host without Python (e.g. Termux) can build
# it; being a macro, the placeholder stays local to that function.
macro(sherpa_onnx_allow_missing_python)
  if(NOT PYTHON_EXECUTABLE AND NOT Python_EXECUTABLE AND NOT Python3_EXECUTABLE)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(NOT Python3_FOUND AND NOT APPLE)
      set(PYTHON_EXECUTABLE "python3-unused")
    endif()
  endif()
endmacro()
