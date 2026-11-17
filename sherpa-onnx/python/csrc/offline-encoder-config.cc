// sherpa-onnx/python/csrc/offline-encoder-config.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/python/csrc/offline-encoder-config.h"

#include <string>

#include "sherpa-onnx/csrc/offline-encoder-window.h"

namespace sherpa_onnx {

static constexpr const char *kOfflineEncoderWindowConfigInitDoc = R"doc(
How an offline encoder runs an utterance longer than a window: in
overlapping windows whose outputs are blended per output frame. With
``seconds=0``, the default, every utterance is encoded whole.

Args:
  seconds:
    Window length. 0: no windows. A LiteRT model of several fixed lengths
    takes its shortest length of at least this, and 0 means its longest.
  overlap:
    The least overlap of neighbouring windows, in seconds.
  slack:
    How far, in seconds, an interior window may move to put its edges
    where the filterbank energy is lowest; 0 keeps the windows evenly
    spaced.
  normalize:
    Scope of a normalization that uses statistics of its input:
    ``window`` or ``utterance``.
  weight:
    ``center``: a frame is the average of the windows that cover it,
    weighted by distance from the nearer edge; ``cut``: a frame is taken
    from the window in which it is farthest from an edge.
  blend:
    For CTC outputs, what ``center`` averages: ``log`` (log probabilities)
    or ``prob`` (probabilities).
  center_power:
    Exponent of the center weight, ``(1 + distance from the nearer retained
    edge) ** center_power``: 0 is uniform, 1 is linear.
  context_frames:
    Context-only output frames per interior window edge. At most one of
    ``context_frames`` and ``context_fraction`` may be nonzero.
  context_fraction:
    Context-only output frames per interior edge as a fraction of each
    window's output length.
)doc";

static constexpr const char *kOfflineEncoderConfigInitDoc = R"doc(
The options of running an offline model's encoder.

Args:
  window:
    An :class:`OfflineEncoderWindowConfig`.
  lengths:
    The input lengths an encoder call may have, in seconds, comma
    separated: every input is padded to the shortest that holds it, else to
    a multiple of ``pad_to`` seconds, else not at all. Both need a model
    that takes a padded batch.
  pad_to:
    See ``lengths``.
)doc";

static void PybindOfflineEncoderWindowConfig(py::module *m) {
  using PyClass = OfflineEncoderWindowConfig;
  py::class_<PyClass>(*m, "OfflineEncoderWindowConfig")
      .def(py::init([](float seconds, float overlap, float slack,
                       const std::string &normalize, const std::string &weight,
                       const std::string &blend, float center_power,
                       int32_t context_frames, float context_fraction) {
             PyClass w;
             w.seconds = seconds;
             w.overlap = overlap;
             w.slack = slack;
             w.normalize = normalize;
             w.weight = weight;
             w.blend = blend;
             w.center_power = center_power;
             w.context_frames = context_frames;
             w.context_fraction = context_fraction;
             return w;
           }),
           py::arg("seconds") = 0.0, py::arg("overlap") = 3.0,
           py::arg("slack") = 1.0, py::arg("normalize") = "window",
           py::arg("weight") = "center", py::arg("blend") = "log",
           py::arg("center_power") = 1.0, py::arg("context_frames") = 0,
           py::arg("context_fraction") = 0.0,
           kOfflineEncoderWindowConfigInitDoc)
      .def_readwrite("seconds", &PyClass::seconds)
      .def_readwrite("overlap", &PyClass::overlap)
      .def_readwrite("slack", &PyClass::slack)
      .def_readwrite("normalize", &PyClass::normalize)
      .def_readwrite("weight", &PyClass::weight)
      .def_readwrite("blend", &PyClass::blend)
      .def_readwrite("center_power", &PyClass::center_power)
      .def_readwrite("context_frames", &PyClass::context_frames)
      .def_readwrite("context_fraction", &PyClass::context_fraction)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

void PybindOfflineEncoderConfig(py::module *m) {
  PybindOfflineEncoderWindowConfig(m);

  using PyClass = OfflineEncoderConfig;
  py::class_<PyClass>(*m, "OfflineEncoderConfig")
      .def(py::init([](const OfflineEncoderWindowConfig &window,
                       const std::string &lengths, float pad_to) {
             PyClass e;
             e.window = window;
             e.lengths = lengths;
             e.pad_to = pad_to;
             return e;
           }),
           py::arg("window") = OfflineEncoderWindowConfig{},
           py::arg("lengths") = "", py::arg("pad_to") = 0.0,
           kOfflineEncoderConfigInitDoc)
      .def_readwrite("window", &PyClass::window)
      .def_readwrite("lengths", &PyClass::lengths)
      .def_readwrite("pad_to", &PyClass::pad_to)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
