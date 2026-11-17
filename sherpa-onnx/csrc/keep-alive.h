// sherpa-onnx/csrc/keep-alive.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_KEEP_ALIVE_H_
#define SHERPA_ONNX_CSRC_KEEP_ALIVE_H_

#include <memory>

namespace sherpa_onnx {

// Shares ownership of "object" and keeps "owners" alive until it is
// destroyed: for an object that a stream keeps and that uses what its
// decoder owns, such as a search over the decoder's scorers, so that a
// stream may outlive its recognizer.
template <typename T, typename... Owners>
std::shared_ptr<T> KeepingAlive(std::unique_ptr<T> object,
                                std::shared_ptr<Owners>... owners) {
  return std::shared_ptr<T>(object.release(), [owners...](T *p) { delete p; });
}

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_KEEP_ALIVE_H_
