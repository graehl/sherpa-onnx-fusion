// sherpa-onnx/csrc/scored-prefix-trie.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/scored-prefix-trie.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace sherpa_onnx {

namespace {

size_t RoundUpTo8(size_t n) { return (n + 7) & ~static_cast<size_t>(7); }

uint64_t ChildKey(uint32_t parent, int32_t token) {
  return (static_cast<uint64_t>(parent) << 32) | static_cast<uint32_t>(token);
}

}  // namespace

void PrepareScorers(const SearchVocabulary &vocabulary,
                    const std::vector<Scorer *> &scorers,
                    UnitId word_boundary_char /*= ' '*/) {
  std::vector<UnitId> tokens;
  std::vector<UnitId> chars = {word_boundary_char};
  for (size_t id = 0; id != vocabulary.tokens.size(); ++id) {
    const SearchToken &token = vocabulary.tokens[id];
    if (!token.emit) continue;
    tokens.push_back(static_cast<UnitId>(id));
    chars.insert(chars.end(), token.chars.begin(), token.chars.end());
  }
  std::sort(chars.begin(), chars.end());
  chars.erase(std::unique(chars.begin(), chars.end()), chars.end());
  for (Scorer *scorer : scorers) {
    const std::vector<UnitId> &units =
        scorer->Info().unit == ScorerUnit::kAsrToken ? tokens : chars;
    scorer->Prepare({units.data(), units.size()});
  }
}

// Scoring of one new node by one scorer: the units still to advance
// through, in units_[begin, begin + length).
struct ScoredPrefixTrie::Pending {
  uint32_t node;
  uint32_t begin;
  uint32_t length;
  uint32_t position;
  float sum;
  bool rejected;
};

ScoredPrefixTrie::ScoredPrefixTrie(
    std::shared_ptr<const SearchVocabulary> vocabulary,
    std::vector<Scorer *> scorers, std::vector<float> weights,
    size_t search_bytes, UnitId word_boundary_char)
    : vocabulary_(std::move(vocabulary)),
      scorers_(std::move(scorers)),
      weights_(std::move(weights)),
      word_boundary_char_(word_boundary_char) {
  size_t num_scorers = scorers_.size();
  search_offset_ = RoundUpTo8(sizeof(Node));
  subtotals_offset_ = search_offset_ + RoundUpTo8(search_bytes);
  size_t offset = subtotals_offset_ + RoundUpTo8(num_scorers * sizeof(float));
  for (Scorer *scorer : scorers_) {
    state_offsets_.push_back(offset);
    offset += RoundUpTo8(scorer->Info().state_bytes);
    utterances_.push_back(scorer->NewUtterance());
    scorers_recombine_ = scorers_recombine_ && scorer->Info().recombines;
  }
  record_bytes_ = offset;
  pending_.resize(num_scorers);
  wave_.resize(num_scorers);
  scratch_.resize(num_scorers);

  uint32_t root = NewNode(kNone, -1);
  Node *node = At(root);
  node->scored = true;
  for (size_t s = 0; s != num_scorers; ++s) {
    scorers_[s]->Start(utterances_[s].get(), ScorerState(node, s));
    Subtotals(node)[s] = 0;
  }
}

ScoredPrefixTrie::~ScoredPrefixTrie() {
  for (uint32_t index = 0; index != num_records_; ++index) {
    Node *node = At(index);
    if (node->live && node->scored) DestroyScorerStates(node);
  }
}

ScoredPrefixTrie::Node *ScoredPrefixTrie::At(uint32_t index) const {
  uint64_t *block = blocks_[index >> kBlockShift].get();
  size_t slot = index & ((1u << kBlockShift) - 1);
  return reinterpret_cast<Node *>(reinterpret_cast<std::byte *>(block) +
                                  slot * record_bytes_);
}

std::byte *ScoredPrefixTrie::SearchState(Node *node) const {
  return reinterpret_cast<std::byte *>(node) + search_offset_;
}

float *ScoredPrefixTrie::Subtotals(Node *node) const {
  return reinterpret_cast<float *>(reinterpret_cast<std::byte *>(node) +
                                   subtotals_offset_);
}

std::byte *ScoredPrefixTrie::ScorerState(Node *node, int32_t scorer) const {
  return reinterpret_cast<std::byte *>(node) + state_offsets_[scorer];
}

void ScoredPrefixTrie::DestroyScorerStates(Node *node) {
  for (size_t s = 0; s != scorers_.size(); ++s) {
    if (auto destroy = scorers_[s]->Info().destroy_state) {
      destroy(ScorerState(node, s));
    }
  }
}

uint32_t ScoredPrefixTrie::NewNode(uint32_t parent, int32_t token) {
  uint32_t index;
  if (!free_records_.empty()) {
    index = free_records_.back();
    free_records_.pop_back();
  } else {
    index = num_records_++;
    if ((index >> kBlockShift) == blocks_.size()) {
      size_t words = (record_bytes_ / 8) << kBlockShift;
      blocks_.emplace_back(new uint64_t[words]);
    }
  }
  ++num_live_;

  Node *node = At(index);
  node->parent = parent;
  node->token = token;
  node->length = 0;
  node->words = 0;
  node->num_children = 0;
  node->serial = next_serial_++;
  node->scorer_total = 0;
  node->word_open = false;
  node->scored = false;
  node->rejected = false;
  node->pinned = false;
  node->live = true;

  if (parent != kNone) {
    Node *up = At(parent);
    const SearchToken &info = vocabulary_->tokens[token];
    bool closes_word = info.starts_word && up->word_open;
    node->length = up->length + 1;
    node->words = up->words + (closes_word ? 1 : 0);
    node->word_open = info.starts_word ? !info.chars.empty()
                                       : (up->word_open || !info.chars.empty());
    up->num_children += 1;
    children_.emplace(ChildKey(parent, token), index);
  }
  return index;
}

void ScoredPrefixTrie::FreeNode(uint32_t index) {
  Node *node = At(index);
  if (node->scored) DestroyScorerStates(node);
  children_.erase(ChildKey(node->parent, node->token));
  At(node->parent)->num_children -= 1;
  node->live = false;
  free_records_.push_back(index);
  --num_live_;
}

void ScoredPrefixTrie::ReleaseIfUnused(uint32_t index) {
  while (index != kRoot) {
    Node *node = At(index);
    if (!node->live || node->pinned || node->num_children != 0) return;
    uint32_t parent = node->parent;
    FreeNode(index);
    index = parent;
  }
}

uint32_t ScoredPrefixTrie::FindOrCreateChild(uint32_t parent, int32_t token,
                                             bool *created) {
  auto it = children_.find(ChildKey(parent, token));
  if (it != children_.end()) {
    *created = false;
    return it->second;
  }
  *created = true;
  uint32_t index = NewNode(parent, token);
  new_nodes_.push_back(index);

  Node *node = At(index);
  const SearchToken &info = vocabulary_->tokens[token];
  bool closes_word = info.starts_word && At(parent)->word_open;
  for (size_t s = 0; s != scorers_.size(); ++s) {
    Pending pending = {index, static_cast<uint32_t>(units_.size()), 0, 0, 0,
                       false};
    if (scorers_[s]->Info().unit == ScorerUnit::kAsrToken) {
      units_.push_back(node->token);
    } else {
      if (closes_word) units_.push_back(word_boundary_char_);
      units_.insert(units_.end(), info.chars.begin(), info.chars.end());
    }
    pending.length = static_cast<uint32_t>(units_.size()) - pending.begin;
    pending_[s].push_back(pending);
  }
  return index;
}

void ScoredPrefixTrie::StartScoring() {
  new_nodes_.clear();
  units_.clear();
  for (auto &pending : pending_) pending.clear();
}

bool ScoredPrefixTrie::GatherScorerWave(
    int32_t scorer, std::vector<ScorerExtension> *extensions) {
  auto &wave = wave_[scorer];
  wave.clear();
  size_t state_words = RoundUpTo8(scorers_[scorer]->Info().state_bytes) / 8;
  auto &pending_list = pending_[scorer];
  // Intermediate states of a several-unit token alternate between two
  // scratch buffers per pending node.
  scratch_[scorer].resize(pending_list.size() * 2 * state_words);
  for (size_t i = 0; i != pending_list.size(); ++i) {
    Pending &pending = pending_list[i];
    if (pending.rejected || pending.position == pending.length) continue;
    Node *node = At(pending.node);
    auto scratch = [&](uint32_t position) {
      return reinterpret_cast<std::byte *>(
          scratch_[scorer].data() + (i * 2 + (position & 1)) * state_words);
    };
    ScorerExtension extension;
    extension.parent.utterance = utterances_[scorer].get();
    extension.parent.bytes = pending.position == 0
                                 ? ScorerState(At(node->parent), scorer)
                                 : scratch(pending.position - 1);
    extension.unit = units_[pending.begin + pending.position];
    extension.child = pending.position + 1 == pending.length
                          ? ScorerState(node, scorer)
                          : scratch(pending.position);
    extensions->push_back(extension);
    wave.push_back(static_cast<uint32_t>(i));
  }
  return !wave.empty();
}

void ScoredPrefixTrie::ApplyScorerWave(int32_t scorer,
                                       const ScorerAdvanceResult *results) {
  size_t state_words = RoundUpTo8(scorers_[scorer]->Info().state_bytes) / 8;
  auto destroy = scorers_[scorer]->Info().destroy_state;
  const auto &wave = wave_[scorer];
  for (size_t j = 0; j != wave.size(); ++j) {
    size_t i = wave[j];
    Pending &pending = pending_[scorer][i];
    if (pending.position > 0 && destroy != nullptr) {
      destroy(reinterpret_cast<std::byte *>(
          scratch_[scorer].data() +
          (i * 2 + ((pending.position - 1) & 1)) * state_words));
    }
    if (results[j].rejected) {
      pending.rejected = true;
    } else {
      pending.sum += results[j].log_score;
      pending.position += 1;
    }
  }
}

void ScoredPrefixTrie::FinishScoring() {
  size_t num_scorers = scorers_.size();
  for (size_t n = 0; n != new_nodes_.size(); ++n) {
    Node *node = At(new_nodes_[n]);
    Node *parent = At(node->parent);
    bool rejected = false;
    for (size_t s = 0; s != num_scorers; ++s) {
      rejected = rejected || pending_[s][n].rejected;
    }
    if (rejected) {
      // Discard the states the other scorers did write.
      for (size_t s = 0; s != num_scorers; ++s) {
        const Pending &pending = pending_[s][n];
        auto destroy = scorers_[s]->Info().destroy_state;
        if (!pending.rejected && pending.length != 0 && destroy != nullptr) {
          destroy(ScorerState(node, s));
        }
      }
      node->rejected = true;
      continue;
    }
    float total = parent->scorer_total;
    for (size_t s = 0; s != num_scorers; ++s) {
      const Pending &pending = pending_[s][n];
      const ScorerInfo &info = scorers_[s]->Info();
      if (pending.length == 0) {
        // No unit for this scorer: the state is the parent's.
        if (info.copy_state != nullptr) {
          info.copy_state(ScorerState(parent, s), ScorerState(node, s));
        } else {
          std::memcpy(ScorerState(node, s), ScorerState(parent, s),
                      info.state_bytes);
        }
      }
      Subtotals(node)[s] = Subtotals(parent)[s] + pending.sum;
      if (weights_[s] != 0) total += weights_[s] * pending.sum;
    }
    node->scorer_total = total;
    node->scored = true;
  }
  num_scored_ += static_cast<int64_t>(new_nodes_.size());
}

void ScoredPrefixTrie::FinalScores(const std::vector<uint32_t> &nodes,
                                   std::vector<float> *final_scores,
                                   std::vector<bool> *accepted) const {
  std::vector<std::vector<float>> scores;
  std::vector<std::vector<bool>> accepts;
  FinalScoresTogether({this}, {nodes}, &scores, &accepts);
  *final_scores = std::move(scores[0]);
  *accepted = std::move(accepts[0]);
}

void ScoredPrefixTrie::FinalScoresTogether(
    const std::vector<const ScoredPrefixTrie *> &tries,
    const std::vector<std::vector<uint32_t>> &nodes,
    std::vector<std::vector<float>> *final_scores,
    std::vector<std::vector<bool>> *accepted) {
  final_scores->assign(tries.size(), {});
  accepted->assign(tries.size(), {});
  if (tries.empty()) return;
  size_t num_scorers = tries[0]->scorers_.size();
  size_t total = 0;
  for (size_t t = 0; t != tries.size(); ++t) {
    (*final_scores)[t].assign(nodes[t].size() * num_scorers, 0);
    (*accepted)[t].assign(nodes[t].size(), true);
    total += nodes[t].size();
  }
  std::vector<ScorerStateRef> states(total);
  std::vector<ScorerFinalScore> results(total);
  for (size_t s = 0; s != num_scorers; ++s) {
    size_t k = 0;
    for (size_t t = 0; t != tries.size(); ++t) {
      for (uint32_t node : nodes[t]) {
        states[k++] = {tries[t]->utterances_[s].get(),
                       tries[t]->ScorerState(tries[t]->At(node), s)};
      }
    }
    tries[0]->scorers_[s]->Finish({states.data(), total}, results.data());
    k = 0;
    for (size_t t = 0; t != tries.size(); ++t) {
      for (size_t i = 0; i != nodes[t].size(); ++i, ++k) {
        (*final_scores)[t][i * num_scorers + s] = results[k].log_score;
        if (!results[k].accepted) (*accepted)[t][i] = false;
      }
    }
  }
}

int32_t ScoredPrefixTrie::LengthIn(const Node &node, LengthBonusUnit unit,
                                   bool at_end) const {
  if (unit == LengthBonusUnit::kToken) return node.length;
  return node.words + (at_end && node.word_open ? 1 : 0);
}

bool ScoredPrefixTrie::SameScorerFuture(const Node &a, const Node &b) const {
  if (a.word_open != b.word_open) return false;
  for (size_t s = 0; s != scorers_.size(); ++s) {
    if (std::memcmp(ScorerState(const_cast<Node *>(&a), s),
                    ScorerState(const_cast<Node *>(&b), s),
                    scorers_[s]->Info().state_bytes) != 0) {
      return false;
    }
  }
  return true;
}

uint64_t ScoredPrefixTrie::HashScorerFuture(const Node &node) const {
  uint64_t hash = node.word_open ? 0x9e3779b97f4a7c15ull : 0;
  auto mix = [&hash](uint64_t word) {
    hash = (hash ^ word) * 0xff51afd7ed558ccdull;
    hash ^= hash >> 32;
  };
  for (size_t s = 0; s != scorers_.size(); ++s) {
    const std::byte *bytes = ScorerState(const_cast<Node *>(&node), s);
    size_t size = scorers_[s]->Info().state_bytes;
    size_t k = 0;
    for (; k + 8 <= size; k += 8) {
      uint64_t word;
      std::memcpy(&word, bytes + k, 8);
      mix(word);
    }
    if (k != size) {
      uint64_t word = 0;
      std::memcpy(&word, bytes + k, size - k);
      mix(word);
    }
  }
  return hash;
}

void ScoredPrefixTrie::TokensOf(uint32_t index,
                                std::vector<int32_t> *tokens) const {
  tokens->clear();
  for (const Node *node = At(index); node->parent != kNone;
       node = At(node->parent)) {
    tokens->push_back(node->token);
  }
  std::reverse(tokens->begin(), tokens->end());
}

}  // namespace sherpa_onnx
