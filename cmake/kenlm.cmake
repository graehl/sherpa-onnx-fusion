# KenLM (https://github.com/kpu/kenlm) is licensed LGPL-2.1-or-later.
#
# Only its query sources (util/, util/double-conversion/, lm/ without the
# builder, filter, interpolate and wrapper subdirectories, and without any
# *main.cc or *test.cc) are built here, as a static library named `kenlm`.
# KenLM's own CMakeLists.txt is not used because it requires Boost; the
# query sources do not. Compressed ARPA input (HAVE_ZLIB, HAVE_BZLIB,
# HAVE_XZLIB) is deliberately left disabled so no extra libraries are needed.
include_guard(GLOBAL)

function(download_kenlm)
  include(FetchContent)

  set(kenlm_COMMIT "4cb443e60b7bf2c0ddf3c745378f76cb59e254e5")
  set(kenlm_URL  "https://github.com/kpu/kenlm/archive/${kenlm_COMMIT}.tar.gz")
  set(kenlm_HASH "SHA256=11df4f929b175f0b3bd26be7a5a83b3c17cdefa83baecab0b6e69f77430184ea")

  # If you don't have access to the Internet,
  # please pre-download kenlm
  set(possible_file_locations
    $ENV{HOME}/Downloads/kenlm-${kenlm_COMMIT}.tar.gz
    ${CMAKE_SOURCE_DIR}/kenlm-${kenlm_COMMIT}.tar.gz
    ${CMAKE_BINARY_DIR}/kenlm-${kenlm_COMMIT}.tar.gz
    /tmp/kenlm-${kenlm_COMMIT}.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(kenlm_URL  "${f}")
      file(TO_CMAKE_PATH "${kenlm_URL}" kenlm_URL)
      message(STATUS "Found local downloaded kenlm: ${kenlm_URL}")
      break()
    endif()
  endforeach()

  FetchContent_Declare(kenlm
    URL
      ${kenlm_URL}
    URL_HASH          ${kenlm_HASH}
  )

  FetchContent_GetProperties(kenlm)
  if(NOT kenlm_POPULATED)
    message(STATUS "Downloading kenlm from ${kenlm_URL}")
    FetchContent_Populate(kenlm)
  endif()
  message(STATUS "kenlm is downloaded to ${kenlm_SOURCE_DIR}")

  file(GLOB kenlm_srcs
    ${kenlm_SOURCE_DIR}/util/double-conversion/*.cc
    ${kenlm_SOURCE_DIR}/util/*.cc
    ${kenlm_SOURCE_DIR}/lm/*.cc
  )
  list(FILTER kenlm_srcs EXCLUDE REGEX "(main|test)\\.cc$")

  add_library(kenlm STATIC ${kenlm_srcs})

  set_target_properties(kenlm PROPERTIES POSITION_INDEPENDENT_CODE ON)

  target_include_directories(kenlm PUBLIC ${kenlm_SOURCE_DIR})
  target_compile_definitions(kenlm PUBLIC KENLM_MAX_ORDER=10)
  target_compile_definitions(kenlm PRIVATE NDEBUG)

  if(MSVC)
    target_compile_options(kenlm PRIVATE /w)
  else()
    target_compile_options(kenlm PRIVATE -w)
  endif()

  # libc++ removed std::unary_function/binary_function in C++17 mode.
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_definitions(kenlm
      PRIVATE
        _LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION
    )
  endif()

  if(CMAKE_SYSTEM_NAME STREQUAL Linux AND NOT ANDROID)
    target_link_libraries(kenlm PUBLIC rt)
  endif()

  if(NOT BUILD_SHARED_LIBS)
    install(TARGETS kenlm DESTINATION lib)
  endif()
endfunction()
