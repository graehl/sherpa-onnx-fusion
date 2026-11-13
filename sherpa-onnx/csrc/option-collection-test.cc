// sherpa-onnx/csrc/option-collection-test.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/option-collection.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/option-json.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {
namespace {

struct Limits {
  bool strict = false;
  int32_t depth = 3;

  void Register(ParseOptions *po) {
    po->Register("strict", &strict, "Strict limits.");
    po->Register("depth", &depth, "Depth limit.");
  }
};

struct Part {
  std::string label;

  void Register(ParseOptions *po) {
    po->Register("label", &label, "Part label.");
  }
};

struct Item {
  std::string name;
  float weight = 1.0f;
  std::string extra = "{}";
  Limits limits;
  std::vector<Part> parts;

  void Register(ParseOptions *po) {
    po->Register("name", &name, "Item name.");
    po->Register("weight", &weight, "Item weight.");
    po->Register("extra", &extra, "JSON text for the item.");
    ParseOptions limits_po("limits", po);
    limits.Register(&limits_po);
    RegisterOptionCollection(po, "part", &parts, "Parts of the item.");
  }
};

struct Config {
  int32_t width = 8;
  std::vector<Item> items;

  void Register(ParseOptions *po) {
    po->Register("width", &width, "Width.");
    RegisterOptionCollection(po, "item", &items, "Items.");
    RegisterJsonOptionsFile(po, "items-config", "JSON file of options.");
  }
};

void Read(const std::vector<std::string> &options, Config *config) {
  std::vector<const char *> argv = {"option-collection-test"};
  for (const auto &option : options) {
    argv.push_back(option.c_str());
  }
  ParseOptions po("Test.");
  config->Register(&po);
  po.Read(static_cast<int32_t>(argv.size()), argv.data());
}

bool ApplyJson(const std::string &json, Config *config, std::string *error) {
  ParseOptions po("Test.");
  config->Register(&po);
  return ApplyJsonOptions(json, &po, error);
}

TEST(OptionCollection, SplitsIndexAndField) {
  size_t index = 9;
  std::string field;
  EXPECT_TRUE(SplitIndexedOptionPath("2.limits.depth", &index, &field));
  EXPECT_EQ(index, 2u);
  EXPECT_EQ(field, "limits.depth");

  EXPECT_TRUE(SplitIndexedOptionPath("weight", &index, &field));
  EXPECT_EQ(index, 0u);
  EXPECT_EQ(field, "weight");

  EXPECT_FALSE(SplitIndexedOptionPath("2", &index, &field));
  EXPECT_FALSE(SplitIndexedOptionPath("", &index, &field));
  EXPECT_FALSE(SplitIndexedOptionPath("256.weight", &index, &field));
  EXPECT_FALSE(SplitIndexedOptionPath("99999.weight", &index, &field));
}

TEST(OptionCollection, UnindexedOptionsSetTheFirstElement) {
  Config config;
  Read({"--item.name=a", "--item.weight=0.5", "--width=4"}, &config);
  ASSERT_EQ(config.items.size(), 1u);
  EXPECT_EQ(config.items[0].name, "a");
  EXPECT_FLOAT_EQ(config.items[0].weight, 0.5f);
  EXPECT_EQ(config.items[0].extra, "{}");
  EXPECT_EQ(config.width, 4);
}

TEST(OptionCollection, IndexedOptionsCreateElementsInAnyOrder) {
  Config config;
  Read({"--item.2.name=c", "--item.0.name=a", "--item.2.weight=3"}, &config);
  ASSERT_EQ(config.items.size(), 3u);
  EXPECT_EQ(config.items[0].name, "a");
  EXPECT_EQ(config.items[1].name, "");
  EXPECT_FLOAT_EQ(config.items[1].weight, 1.0f);
  EXPECT_EQ(config.items[2].name, "c");
  EXPECT_FLOAT_EQ(config.items[2].weight, 3.0f);
}

TEST(OptionCollection, LaterAssignmentWinsAndBothSpellingsMeanElementZero) {
  Config config;
  Read({"--item.0.name=a", "--item.name=b"}, &config);
  ASSERT_EQ(config.items.size(), 1u);
  EXPECT_EQ(config.items[0].name, "b");
}

TEST(OptionCollection, ReachesNestedStructsAndNestedCollections) {
  Config config;
  Read({"--item.1.limits.depth=7", "--item.1.limits.strict",
        "--item.1.part.1.label=x", "--item.1.part.label=w"},
       &config);
  ASSERT_EQ(config.items.size(), 2u);
  EXPECT_EQ(config.items[1].limits.depth, 7);
  EXPECT_TRUE(config.items[1].limits.strict);
  EXPECT_EQ(config.items[0].limits.depth, 3);
  ASSERT_EQ(config.items[1].parts.size(), 2u);
  EXPECT_EQ(config.items[1].parts[0].label, "w");
  EXPECT_EQ(config.items[1].parts[1].label, "x");
}

TEST(OptionCollection, UnknownFieldIsAnInvalidOption) {
  Config config;
  EXPECT_DEATH(Read({"--item.0.nonesuch=1"}, &config), "Invalid option");
  EXPECT_DEATH(Read({"--item.300.name=a"}, &config), "Invalid option");
}

TEST(OptionJson, ArrayOfObjectsAssignsEachElement) {
  Config config;
  std::string error;
  ASSERT_TRUE(ApplyJson(
      R"({"width": 2, "item": [
            {"name": "a", "weight": 0.25, "limits": {"depth": 5}},
            {"name": "b", "limits": {"strict": true},
             "part": [{"label": "x"}, {"label": "y"}]}]})",
      &config, &error))
      << error;
  EXPECT_EQ(config.width, 2);
  ASSERT_EQ(config.items.size(), 2u);
  EXPECT_EQ(config.items[0].name, "a");
  EXPECT_FLOAT_EQ(config.items[0].weight, 0.25f);
  EXPECT_EQ(config.items[0].limits.depth, 5);
  EXPECT_EQ(config.items[1].name, "b");
  EXPECT_FLOAT_EQ(config.items[1].weight, 1.0f);
  EXPECT_TRUE(config.items[1].limits.strict);
  ASSERT_EQ(config.items[1].parts.size(), 2u);
  EXPECT_EQ(config.items[1].parts[1].label, "y");
}

TEST(OptionJson, SingleObjectIsAListOfOne) {
  Config config;
  std::string error;
  ASSERT_TRUE(ApplyJson(R"({"item": {"name": "only", "weight": "0.5"}})",
                        &config, &error))
      << error;
  ASSERT_EQ(config.items.size(), 1u);
  EXPECT_EQ(config.items[0].name, "only");
  EXPECT_FLOAT_EQ(config.items[0].weight, 0.5f);
}

TEST(OptionJson, StructuredValueForAStringOptionIsPassedAsJsonText) {
  Config config;
  std::string error;
  ASSERT_TRUE(
      ApplyJson(R"({"item": [{"extra": {"layout": "gru", "sizes": [1, 2]}}]})",
                &config, &error))
      << error;
  ASSERT_EQ(config.items.size(), 1u);
  EXPECT_EQ(config.items[0].extra, R"({"layout":"gru","sizes":[1,2]})");
}

TEST(OptionJson, ReportsProblemsWithTheirPath) {
  Config config;
  std::string error;

  EXPECT_FALSE(ApplyJson(R"({"item": [{"nonesuch": 1}]})", &config, &error));
  EXPECT_EQ(error, "item.0.nonesuch: no such option");

  EXPECT_FALSE(ApplyJson(R"({"width": 1, "width": 2})", &config, &error));
  EXPECT_EQ(error, "duplicate JSON key \"width\"");

  EXPECT_FALSE(ApplyJson(R"({"item": [{"name": null}]})", &config, &error));
  EXPECT_EQ(error, "item.0.name: null is not a value");

  EXPECT_FALSE(ApplyJson(R"({"item.0.name": "a"})", &config, &error));
  EXPECT_EQ(error, "top level: invalid JSON key \"item.0.name\"");

  EXPECT_FALSE(ApplyJson(R"([{"name": "a"}])", &config, &error));
  EXPECT_EQ(error, "top level: expected a JSON object");

  EXPECT_FALSE(ApplyJson(R"({"width": )", &config, &error));
  EXPECT_EQ(error.rfind("invalid JSON: ", 0), 0u) << error;
}

TEST(OptionJson, FileOptionAppliesInCommandLineOrder) {
  std::string filename = ::testing::TempDir() + "option-collection-test.json";
  {
    std::ofstream os(filename);
    os << R"({"width": 2, "item": [{"name": "a"}, {"name": "b"}]})";
  }

  Config config;
  Read({"--width=9", "--items-config=" + filename, "--item.1.name=z"}, &config);
  EXPECT_EQ(config.width, 2);
  ASSERT_EQ(config.items.size(), 2u);
  EXPECT_EQ(config.items[0].name, "a");
  EXPECT_EQ(config.items[1].name, "z");

  EXPECT_EQ(std::remove(filename.c_str()), 0);
}

}  // namespace
}  // namespace sherpa_onnx
