// sherpa-onnx/csrc/option-json.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/option-json.h"

#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

namespace {

using Json = nlohmann::json;

// Parses, rejecting an object that repeats a key: the DOM would silently keep
// only the last value.
bool ParseJsonRejectingDuplicateKeys(const std::string &text, Json *json,
                                     std::string *error) {
  std::vector<std::unordered_set<std::string>> keys_of_open_objects;
  std::string duplicate;
  Json::parser_callback_t track_keys =
      [&](int /*depth*/, Json::parse_event_t event, Json &parsed) {
        if (event == Json::parse_event_t::object_start) {
          keys_of_open_objects.emplace_back();
        } else if (event == Json::parse_event_t::object_end) {
          keys_of_open_objects.pop_back();
        } else if (event == Json::parse_event_t::key) {
          const auto &key = parsed.get_ref<const std::string &>();
          if (!keys_of_open_objects.back().insert(key).second &&
              duplicate.empty()) {
            duplicate = key;
          }
        }
        return true;
      };
  try {
    *json = Json::parse(text, track_keys);
  } catch (const std::exception &e) {
    *error = std::string("invalid JSON: ") + e.what();
    return false;
  }
  if (!duplicate.empty()) {
    *error = "duplicate JSON key \"" + duplicate + "\"";
    return false;
  }
  return true;
}

bool Fail(const std::string &path, const std::string &what,
          std::string *error) {
  *error = (path.empty() ? std::string("top level") : path) + ": " + what;
  return false;
}

bool ApplyJsonAt(const Json &json, const std::string &path, ParseOptions *po,
                 std::string *error) {
  if (json.is_object() || json.is_array()) {
    // A string option takes a structured value whole, as JSON text.
    if (!path.empty() && po->Assign(path, json.dump())) {
      return true;
    }
    if (json.is_array()) {
      if (path.empty()) {
        return Fail(path, "expected a JSON object", error);
      }
      size_t index = 0;
      for (const auto &element : json) {
        if (!ApplyJsonAt(element, path + "." + std::to_string(index), po,
                         error)) {
          return false;
        }
        ++index;
      }
      return true;
    }
    for (const auto &member : json.items()) {
      const std::string &key = member.key();
      if (key.empty() || key.find('.') != std::string::npos) {
        return Fail(path, "invalid JSON key \"" + key + "\"", error);
      }
      if (!ApplyJsonAt(member.value(), path.empty() ? key : path + "." + key,
                       po, error)) {
        return false;
      }
    }
    return true;
  }

  if (path.empty()) {
    return Fail(path, "expected a JSON object", error);
  }
  if (json.is_null()) {
    return Fail(path, "null is not a value", error);
  }
  std::string value = json.is_string() ? json.get<std::string>() : json.dump();
  if (!po->Assign(path, value)) {
    return Fail(path, "no such option", error);
  }
  return true;
}

}  // namespace

bool ApplyJsonOptions(const std::string &json_text, ParseOptions *po,
                      std::string *error) {
  Json json;
  if (!ParseJsonRejectingDuplicateKeys(json_text, &json, error)) {
    return false;
  }
  return ApplyJsonAt(json, "", po, error);
}

void RegisterJsonOptionsFile(ParseOptions *po, const std::string &name,
                             const std::string &doc) {
  std::string usage = "  --" + name + "=<file.json> : " + doc + "\n";
  po->RegisterHandler(
      name,
      [po, name](const std::string &rest, const std::string &filename,
                 bool /*has_equal_sign*/) {
        if (!rest.empty()) {
          return false;
        }
        std::ifstream is(filename);
        if (!is) {
          SHERPA_ONNX_LOGE("--%s: cannot open '%s'", name.c_str(),
                           filename.c_str());
          SHERPA_ONNX_EXIT(-1);
        }
        std::ostringstream text;
        text << is.rdbuf();
        std::string error;
        if (!ApplyJsonOptions(text.str(), po, &error)) {
          SHERPA_ONNX_LOGE("--%s=%s: %s", name.c_str(), filename.c_str(),
                           error.c_str());
          SHERPA_ONNX_EXIT(-1);
        }
        return true;
      },
      usage);
}

}  // namespace sherpa_onnx
