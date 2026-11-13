// sherpa-onnx/csrc/option-collection.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_OPTION_COLLECTION_H_
#define SHERPA_ONNX_CSRC_OPTION_COLLECTION_H_

#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

// Largest element count an indexed option may create.
constexpr size_t kMaxOptionCollectionSize = 256;

// Splits the part of an indexed option after the collection name, e.g.
// "1.weight" into index 1 and field "weight". Without a leading index the
// whole path is the field of element 0. Returns false for an index that is
// too large or a missing field.
inline bool SplitIndexedOptionPath(const std::string &path, size_t *index,
                                   std::string *field) {
  size_t digits = 0;
  while (digits != path.size() &&
         std::isdigit(static_cast<unsigned char>(path[digits]))) {
    ++digits;
  }
  bool has_index =
      digits != 0 && (digits == path.size() || path[digits] == '.');
  *index = 0;
  if (has_index) {
    if (digits > 4) {
      return false;
    }
    *index = std::stoul(path.substr(0, digits));
    if (*index >= kMaxOptionCollectionSize) {
      return false;
    }
    *field = digits == path.size() ? "" : path.substr(digits + 1);
  } else {
    *field = path;
  }
  return !field->empty();
}

// Indexed options for a vector of option structs: --name.N.field=value
// assigns "field" of element N, creating elements up to N as needed, and
// --name.field=value assigns element 0. Options is any struct with the usual
// void Register(ParseOptions *). "collection" must outlive "po".
// See topics/json-option-collections.md.
template <class Options>
void RegisterOptionCollection(ParseOptions *po, const std::string &name,
                              std::vector<Options> *collection,
                              const std::string &doc) {
  Options defaults;
  ParseOptions defaults_po("");
  defaults.Register(&defaults_po);
  std::string usage = "  --" + name + ".N.<option> : " + doc +
                      " N counts from 0 and may be omitted for the first." +
                      " Options:\n" + defaults_po.OptionsUsage(name + ".N.");

  po->RegisterHandler(
      name,
      [collection](const std::string &rest, const std::string &value,
                   bool has_equal_sign) {
        size_t index = 0;
        std::string field;
        if (!SplitIndexedOptionPath(rest, &index, &field)) {
          return false;
        }
        if (collection->size() <= index) {
          collection->resize(index + 1);
        }
        // A parser local to this assignment keeps no pointer into the vector.
        ParseOptions element_po("");
        (*collection)[index].Register(&element_po);
        return element_po.Assign(field, value, has_equal_sign);
      },
      usage);
}

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OPTION_COLLECTION_H_
