// sherpa-onnx/csrc/online-ctc-fst-decoder-test.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/online-ctc-fst-decoder.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "fst/fstlib.h"
#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/endpoint.h"
#include "sherpa-onnx/csrc/fusion-config.h"
#include "sherpa-onnx/csrc/online-ctc-prefix-beam-search-decoder.h"
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {
namespace {

class OnlineCtcFstDecoderTest : public ::testing::Test {
 protected:
  void SetUp() override {
#if defined(_WIN32)
    char temp_path[MAX_PATH];
    char temp_file[MAX_PATH];
    DWORD length = GetTempPathA(MAX_PATH, temp_path);
    ASSERT_GT(length, 0);
    ASSERT_LT(length, MAX_PATH);
    ASSERT_NE(GetTempFileNameA(temp_path, "ctc", 0, temp_file), 0);
    config_.graph = temp_file;
#else
    char path[] = "/tmp/sherpa_onnx_ctc_test_XXXXXX";
    int fd = mkstemp(path);
    ASSERT_NE(fd, -1);
    close(fd);
    config_.graph = path;
#endif
    fst::VectorFst<fst::StdArc> graph;
    graph.AddState();
    graph.SetStart(0);
    graph.SetFinal(0, fst::TropicalWeight::One());
    // FST input labels are token IDs + 1; blank emits no word.
    graph.AddArc(0, fst::StdArc(1, 0, fst::TropicalWeight::One(), 0));
    graph.AddArc(0, fst::StdArc(2, 1, fst::TropicalWeight::One(), 0));
    ASSERT_TRUE(graph.Write(config_.graph));
  }

  void TearDown() override {
    if (!config_.graph.empty()) {
      EXPECT_EQ(std::remove(config_.graph.c_str()), 0);
    }
  }

  OnlineCtcFstDecoderConfig config_;
};

TEST_F(OnlineCtcFstDecoderTest, KnownBugAllBlankChunksOvercountSilence) {
  OnlineCtcFstDecoder decoder(config_, 0);
  OnlineStream chunked;
  chunked.SetFasterDecoder(decoder.CreateFasterDecoder());
  OnlineStream *streams[] = {&chunked};
  std::vector<OnlineCtcDecoderResult> results(1);
  const float scores[] = {0, -100, 0, -100, 0, -100, 0, -100, 0, -100, 0, -100};
  const std::vector<int32_t> correct_counts = {2, 4, 6};
  const std::vector<int32_t> known_bug_counts = {2, 6, 12};
  std::vector<int32_t> observed_counts;
  for (int32_t chunk = 0; chunk != 3; ++chunk) {
    decoder.Decode(scores + chunk * 4, 1, 2, 2, &results, streams, 1);
    observed_counts.push_back(results[0].num_trailing_blanks);
    EXPECT_EQ(chunked.GetFasterDecoderProcessedFrames(), 2 * (chunk + 1));
    EXPECT_TRUE(results[0].tokens.empty());
    EXPECT_TRUE(results[0].timestamps.empty());
    EXPECT_TRUE(results[0].words.empty());
  }

  // Characterize the defect until the separate decoder fix changes this oracle:
  // gaps/streaming-ctc-trailing-blank-overcount.md.
  EXPECT_EQ(observed_counts, known_bug_counts)
      << "Correct trailing-blank counts are 2, 4, 6; update this test when "
         "fixed.";
  if (observed_counts == known_bug_counts) {
    std::cerr
        << "[WARNING] CTC trailing-blank defect reproduced: got 2, 6, 12; "
           "the correct output is 2, 4, 6. Earlier silence is recounted "
           "across chunks and can trigger a premature endpoint.\n";
  }

  OnlineStream whole;
  whole.SetFasterDecoder(decoder.CreateFasterDecoder());
  OnlineStream *whole_streams[] = {&whole};
  std::vector<OnlineCtcDecoderResult> whole_results(1);
  decoder.Decode(scores, 1, 6, 2, &whole_results, whole_streams, 1);
  ASSERT_EQ(whole_results[0].num_trailing_blanks, correct_counts.back());
  EXPECT_EQ(whole_results[0].tokens, results[0].tokens);

  EndpointConfig endpoint_config;
  endpoint_config.rule1.min_trailing_silence = 0.3f;
  Endpoint endpoint(endpoint_config);
  // Six 40-ms frames are 240 ms: the 300-ms threshold must not fire.
  EXPECT_FALSE(endpoint.IsEndpoint(6, correct_counts.back(), 0.04f));
  EXPECT_FALSE(
      endpoint.IsEndpoint(6, whole_results[0].num_trailing_blanks, 0.04f));
  EXPECT_TRUE(endpoint.IsEndpoint(6, results[0].num_trailing_blanks, 0.04f));

  // A nonblank in the full traceback resets the stale count.
  const float speech_then_blank[] = {-100, 0, 0, -100};
  decoder.Decode(speech_then_blank, 1, 2, 2, &results, streams, 1);
  EXPECT_EQ(results[0].num_trailing_blanks, 1);
  EXPECT_EQ(results[0].tokens, std::vector<int64_t>({1}));
  EXPECT_EQ(results[0].timestamps, std::vector<int32_t>({6}));
}

// The same input through the streaming prefix search, which keeps its search
// across chunks instead of re-reading a traceback: the counts are the correct
// ones, chunked or whole, and the endpoint rule does not fire early.
TEST(OnlineCtcPrefixBeamSearchDecoderTrailingSilence,
     AllBlankChunksCountSilenceOnce) {
  SymbolTable symbol_table("<blk> 0\na 1\n", false);
  FusionConfig fusion;
  OnlineCtcPrefixBeamSearchDecoder decoder(fusion, 4, symbol_table, 0);
  OnlineStream chunked;
  OnlineStream *streams[] = {&chunked};
  std::vector<OnlineCtcDecoderResult> results(1);
  const float scores[] = {0, -100, 0, -100, 0, -100, 0, -100, 0, -100, 0, -100};
  std::vector<int32_t> observed_counts;
  for (int32_t chunk = 0; chunk != 3; ++chunk) {
    decoder.Decode(scores + chunk * 4, 1, 2, 2, &results, streams, 1);
    observed_counts.push_back(results[0].num_trailing_blanks);
    EXPECT_TRUE(results[0].tokens.empty());
    EXPECT_TRUE(results[0].timestamps.empty());
  }
  EXPECT_EQ(observed_counts, std::vector<int32_t>({2, 4, 6}));

  OnlineStream whole;
  OnlineStream *whole_streams[] = {&whole};
  std::vector<OnlineCtcDecoderResult> whole_results(1);
  decoder.Decode(scores, 1, 6, 2, &whole_results, whole_streams, 1);
  EXPECT_EQ(whole_results[0].num_trailing_blanks, 6);

  EndpointConfig endpoint_config;
  endpoint_config.rule1.min_trailing_silence = 0.3f;
  Endpoint endpoint(endpoint_config);
  // Six 40-ms frames are 240 ms: the 300-ms threshold must not fire.
  EXPECT_FALSE(endpoint.IsEndpoint(6, results[0].num_trailing_blanks, 0.04f));

  const float speech_then_blank[] = {-100, 0, 0, -100};
  decoder.Decode(speech_then_blank, 1, 2, 2, &results, streams, 1);
  EXPECT_EQ(results[0].num_trailing_blanks, 1);
  EXPECT_EQ(results[0].tokens, std::vector<int64_t>({1}));
  EXPECT_EQ(results[0].timestamps, std::vector<int32_t>({6}));
}

}  // namespace
}  // namespace sherpa_onnx
