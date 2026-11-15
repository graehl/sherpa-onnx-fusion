# LiteRT (https://github.com/google-ai-edge/LiteRT), Apache-2.0, runs .tflite
# acoustic models; on a Pixel its Google Tensor dispatch library runs a graph
# compiled ahead of time on the phone's TPU.
#
# This defines an imported shared library target `litert` from three pinned
# downloads:
#   - LiteRT's C/C++ SDK release zip, for the C API headers;
#   - the litert Android archive (AAR), for jni/<abi>/libLiteRt.so;
#   - the Google Tensor NPU runtime AAR, for libLiteRtDispatch_GoogleTensor.so,
#     which is only loaded at run time (see --litert-dispatch-dir).
# Both libraries are installed to lib/.
#
# Off Android there is no pinned download of libLiteRt.so: point
# SHERPA_ONNX_LITERT_LIBRARY at one, for instance the libLiteRt.so of the
# ai-edge-litert Python wheel, to run portable graphs on the host CPU.
include_guard(GLOBAL)

set(SHERPA_ONNX_LITERT_LIBRARY "" CACHE FILEPATH
  "libLiteRt.so to link off Android (the Android build downloads its own)")

# Fetches NAME from URL with HASH, preferring a pre-downloaded copy of FILE in
# the usual places, and sets NAME_SOURCE_DIR in the caller.
function(litert_fetch name url hash file)
  include(FetchContent)
  set(possible_file_locations
    $ENV{HOME}/Downloads/${file}
    ${CMAKE_SOURCE_DIR}/${file}
    ${CMAKE_BINARY_DIR}/${file}
    /tmp/${file}
  )
  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(url "${f}")
      file(TO_CMAKE_PATH "${url}" url)
      message(STATUS "Found local downloaded ${file}: ${url}")
      break()
    endif()
  endforeach()

  FetchContent_Declare(${name} URL ${url} URL_HASH ${hash})
  FetchContent_GetProperties(${name})
  if(NOT ${name}_POPULATED)
    message(STATUS "Downloading ${name} from ${url}")
    FetchContent_Populate(${name})
  endif()
  message(STATUS "${name} is downloaded to ${${name}_SOURCE_DIR}")
  set(${name}_SOURCE_DIR ${${name}_SOURCE_DIR} PARENT_SCOPE)
endfunction()

function(download_litert)
  set(litert_VERSION "2.3.0")
  set(maven "https://dl.google.com/android/maven2/com/google/ai/edge/litert")

  litert_fetch(litert_sdk
    "https://github.com/google-ai-edge/LiteRT/releases/download/v${litert_VERSION}/litert_cc_sdk.zip"
    "SHA256=44ebb188ae1a510f233639fa499928fdaaba46436a153a2dbfbb9d87d59728d2"
    "litert_cc_sdk-${litert_VERSION}.zip")

  # litert_common.h includes this header, which LiteRT's own CMake build
  # generates; both accelerator kinds stay enabled.
  set(generated ${CMAKE_BINARY_DIR}/litert-generated)
  file(WRITE ${generated}/litert/build_common/build_config.h
"#ifndef LITERT_BUILD_COMMON_BUILD_CONFIG_H_
#define LITERT_BUILD_COMMON_BUILD_CONFIG_H_
#define LITERT_BUILD_CONFIG_DISABLE_GPU 0
#define LITERT_BUILD_CONFIG_DISABLE_NPU 0
#endif  // LITERT_BUILD_COMMON_BUILD_CONFIG_H_
")

  if(ANDROID)
    litert_fetch(litert_aar
      "${maven}/litert/${litert_VERSION}/litert-${litert_VERSION}.aar"
      "SHA256=9508c6e9142646f0f2bc1fe68b3925fb470ee7ddd29114540339a04a8ab189dc"
      "litert-${litert_VERSION}.aar")
    litert_fetch(litert_npu_google_tensor
      "${maven}/litert-npu-runtime-google-tensor/${litert_VERSION}/litert-npu-runtime-google-tensor-${litert_VERSION}.aar"
      "SHA256=5f38ab3de4b99ef75edf13c595cf0c090b10a82d9697f5b55703409757e786f7"
      "litert-npu-runtime-google-tensor-${litert_VERSION}.aar")
    set(library ${litert_aar_SOURCE_DIR}/jni/${ANDROID_ABI}/libLiteRt.so)
    set(dispatch
      ${litert_npu_google_tensor_SOURCE_DIR}/jni/${ANDROID_ABI}/libLiteRtDispatch_GoogleTensor.so)
    if(NOT EXISTS ${dispatch})
      set(dispatch "")
    endif()
  else()
    set(library ${SHERPA_ONNX_LITERT_LIBRARY})
    set(dispatch "")
  endif()
  if(NOT EXISTS "${library}")
    message(FATAL_ERROR
      "SHERPA_ONNX_ENABLE_LITERT needs libLiteRt.so; not found: '${library}'. "
      "Off Android, set SHERPA_ONNX_LITERT_LIBRARY.")
  endif()

  add_library(litert SHARED IMPORTED GLOBAL)
  set_target_properties(litert PROPERTIES
    IMPORTED_LOCATION ${library}
    IMPORTED_NO_SONAME ON
    INTERFACE_INCLUDE_DIRECTORIES "${litert_sdk_SOURCE_DIR};${generated}"
  )
  install(FILES ${library} ${dispatch} DESTINATION lib)
endfunction()
