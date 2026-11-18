// sherpa-onnx/csrc/scored-prefix-trie.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_SCORED_PREFIX_TRIE_H_
#define SHERPA_ONNX_CSRC_SCORED_PREFIX_TRIE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "sherpa-onnx/csrc/scorer.h"

// The part of a prefix beam search that does not depend on the recognizer's
// model family: one utterance's output prefixes as a trie of records, each
// holding the search's own state for that prefix followed by every scorer's
// subtotal and state bytes, with the scorers run in batched waves.
// See topics/beam-search-and-scorer-fusion.md.

namespace sherpa_onnx {

// What a search needs to know about one recognizer token.
struct SearchToken {
  // False for a token the search never emits, such as an unknown-word token.
  bool emit = true;
  // The token's text begins a new word (its piece begins with a space).
  bool starts_word = false;
  // Unicode scalars of the token's text, without that leading space.
  std::vector<UnitId> chars;
};

struct SearchVocabulary {
  int32_t blank_id = 0;
  std::vector<SearchToken> tokens;  // Indexed by token ID.
};

// Binds a decoder to its scorers: gives each scorer every unit a search over
// "vocabulary" can send it (Scorer::Prepare). Call it once the vocabulary is
// built and before any search uses the scorers.
void PrepareScorers(const SearchVocabulary &vocabulary,
                    const std::vector<Scorer *> &scorers,
                    UnitId word_boundary_char = ' ');

enum class LengthBonusUnit { kToken, kWord };

class ScoredPrefixTrie {
 public:
  static constexpr uint32_t kRoot = 0;
  static constexpr uint32_t kNone = 0xffffffffu;

  // The model-independent part of a prefix's record.
  struct Node {
    uint32_t parent;
    int32_t token;   // Last token; -1 for the empty prefix.
    int32_t length;  // Tokens.
    int32_t words;   // Completed words.
    int32_t num_children;
    // Distinguishes a record from an earlier prefix that used its index.
    uint32_t serial;
    float scorer_total;  // Weighted sum of the scorers' subtotals.
    bool word_open;      // Text has been emitted since the last word boundary.
    bool scored;         // Subtotals are valid, and states unless released.
    bool rejected;       // A scorer rejected this prefix.
    // Some hypothesis may still extend this prefix (the search's beam, or
    // an arrival waiting for it); never reclaimed while set.
    bool pinned;
    // Unpinned with live descendants: its scorer states were destroyed, and
    // are scored again from its parent if a search reaches it again.
    bool states_released;
    bool restoring;  // Queued to have its released states scored again.
    bool live;
  };

  // "search_bytes" is the size of the search's own state per prefix, which
  // the search initializes for each node it creates. "scorers" and
  // "vocabulary" must outlive the trie. One weight per scorer.
  ScoredPrefixTrie(std::shared_ptr<const SearchVocabulary> vocabulary,
                   std::vector<Scorer *> scorers, std::vector<float> weights,
                   size_t search_bytes, UnitId word_boundary_char);
  ~ScoredPrefixTrie();

  ScoredPrefixTrie(const ScoredPrefixTrie &) = delete;
  ScoredPrefixTrie &operator=(const ScoredPrefixTrie &) = delete;

  Node *At(uint32_t index) const;
  // The search's own state for the prefix.
  std::byte *SearchState(Node *node) const;
  // Raw, unweighted log score of each scorer so far.
  float *Subtotals(Node *node) const;
  std::byte *ScorerState(Node *node, int32_t scorer) const;

  const SearchVocabulary &Vocabulary() const { return *vocabulary_; }
  const std::vector<Scorer *> &Scorers() const { return scorers_; }
  const std::vector<float> &Weights() const { return weights_; }
  size_t NumScorers() const { return scorers_.size(); }

  // The prefix "parent" extended by "token": the existing node, or a new one
  // queued for scoring, in which case *created is set. A new node's search
  // state is uninitialized. An existing node whose scorer states were
  // released is queued to have them scored again; its subtotals and
  // scorer_total stay as first computed.
  uint32_t FindOrCreateChild(uint32_t parent, int32_t token, bool *created);

  // Scoring of the nodes created (or reached again) since StartScoring. A
  // token of several scorer units takes one wave per unit.
  void StartScoring();
  // Appends the next extensions for scorer "scorer" and returns true, or
  // returns false when that scorer has no work left.
  bool GatherScorerWave(int32_t scorer,
                        std::vector<ScorerExtension> *extensions);
  // "results" are for the extensions the last GatherScorerWave appended.
  void ApplyScorerWave(int32_t scorer, const ScorerAdvanceResult *results);
  // Completes the new nodes: sets scored, or rejected.
  void FinishScoring();
  const std::vector<uint32_t> &NewNodes() const { return new_nodes_; }

  // Call when a prefix stops being pinned. Frees it if it has no
  // descendants, then any ancestors that this leaves in the same condition;
  // an unpinned prefix that keeps descendants keeps its record (for their
  // tokens and scores) but has its scorer states released, since no search
  // extends it until FindOrCreateChild reaches it again. A release during
  // scoring waits for FinishScoring.
  void ReleaseIfUnused(uint32_t index);

  // End-of-input scores of "nodes": (*final_scores)[i * NumScorers() + s],
  // raw, and whether every scorer accepts node i.
  void FinalScores(const std::vector<uint32_t> &nodes,
                   std::vector<float> *final_scores,
                   std::vector<bool> *accepted) const;
  // The same for several tries created with the same scorers, with one
  // call per scorer serving all of them. Entry i of each argument belongs
  // to tries[i].
  static void FinalScoresTogether(
      const std::vector<const ScoredPrefixTrie *> &tries,
      const std::vector<std::vector<uint32_t>> &nodes,
      std::vector<std::vector<float>> *final_scores,
      std::vector<std::vector<bool>> *accepted);

  // Length bonus units of the prefix; "at_end" counts an unfinished word.
  int32_t LengthIn(const Node &node, LengthBonusUnit unit, bool at_end) const;

  // Whether everything scored from here on is the same for both prefixes as
  // far as the scorers and the word state are concerned, and a hash that is
  // equal when it is. Meaningful when every scorer's states are canonical.
  bool SameScorerFuture(const Node &a, const Node &b) const;
  uint64_t HashScorerFuture(const Node &node) const;
  // True when every scorer's ScorerInfo::recombines is set.
  bool ScorersRecombine() const { return scorers_recombine_; }

  void TokensOf(uint32_t index, std::vector<int32_t> *tokens) const;

  int32_t NumLiveNodes() const { return num_live_; }
  int64_t NumNodesScored() const { return num_scored_; }
  // Prefixes reached again after their scorer states were released.
  int64_t NumStatesRestored() const { return num_restored_; }

 private:
  struct Pending;

  uint32_t NewNode(uint32_t parent, int32_t token);
  void FreeNode(uint32_t index);
  void DestroyScorerStates(Node *node);
  void ReleaseScorerStates(Node *node);
  // Queues the scoring of "index" from its parent.
  void QueueScoring(uint32_t index);

  std::shared_ptr<const SearchVocabulary> vocabulary_;
  std::vector<Scorer *> scorers_;
  std::vector<float> weights_;
  std::vector<std::unique_ptr<ScorerUtterance>> utterances_;
  UnitId word_boundary_char_;
  bool scorers_recombine_ = true;

  // Record layout: Node, the search's state, one float subtotal per scorer,
  // then each scorer's state bytes, all 8-byte aligned.
  size_t search_offset_ = 0;
  size_t subtotals_offset_ = 0;
  std::vector<size_t> state_offsets_;
  size_t record_bytes_ = 0;

  // Records live in fixed-size blocks, so a record's address is stable and
  // records are addressed by index.
  static constexpr uint32_t kBlockShift = 8;
  std::vector<std::unique_ptr<uint64_t[]>> blocks_;
  uint32_t num_records_ = 0;
  uint32_t next_serial_ = 0;
  std::vector<uint32_t> free_records_;
  int32_t num_live_ = 0;
  int64_t num_scored_ = 0;
  int64_t num_restored_ = 0;
  bool scoring_ = false;  // Between StartScoring and FinishScoring.
  std::vector<uint32_t> deferred_releases_;  // Asked for while scoring_.
  std::unordered_map<uint64_t, uint32_t> children_;  // (parent, token) -> index

  std::vector<uint32_t> new_nodes_;
  std::vector<std::vector<Pending>> pending_;   // Per scorer.
  std::vector<std::vector<uint32_t>> wave_;     // Pending indices sent.
  std::vector<std::vector<uint64_t>> scratch_;  // Intermediate states.
  std::vector<UnitId> units_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SCORED_PREFIX_TRIE_H_
