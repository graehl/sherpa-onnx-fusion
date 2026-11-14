// sherpa-onnx/csrc/recurrent-lm-scorer.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/recurrent-lm-scorer.h"

#include <algorithm>
#include <cstring>
#include <deque>
#include <memory>
#include <utility>
#include <vector>

namespace sherpa_onnx {

namespace {

int32_t g_live_nodes = 0;

struct NodePool;

struct Node {
  NodePool *pool = nullptr;
  // Held, with a reference, until this node is computed from its state.
  Node *parent = nullptr;
  int32_t references = 0;
  int32_t unit = 0;  // The unit this node follows.
  bool computed = false;
  bool queued = false;
  // After computing: each state tensor as [layers, hidden], and the
  // log probabilities of every next unit.
  std::vector<std::vector<float>> states;
  std::vector<float> log_probs;
};

// One utterance's nodes. Freed nodes keep their buffers for reuse.
struct NodePool : public ScorerUtterance {
  ~NodePool() override {
    g_live_nodes -= static_cast<int32_t>(nodes.size() - free_nodes.size());
  }

  Node *New() {
    Node *node;
    if (!free_nodes.empty()) {
      node = free_nodes.back();
      free_nodes.pop_back();
    } else {
      nodes.emplace_back();
      node = &nodes.back();
      node->pool = this;
    }
    node->parent = nullptr;
    node->references = 1;
    node->computed = false;
    node->queued = false;
    ++g_live_nodes;
    return node;
  }

  std::deque<Node> nodes;
  std::vector<Node *> free_nodes;
};

Node *NodeOf(const std::byte *state) {
  Node *node;
  std::memcpy(&node, state, sizeof(node));
  return node;
}

void WriteNode(Node *node, std::byte *state) {
  std::memcpy(state, &node, sizeof(node));
}

// Drops one reference; frees the node, and then its parent chain, when
// nothing refers to them.
void Unreference(Node *node) {
  while (node != nullptr && --node->references == 0) {
    Node *parent = node->parent;
    node->parent = nullptr;
    node->pool->free_nodes.push_back(node);
    --g_live_nodes;
    node = parent;
  }
}

void CopyState(const std::byte *from, std::byte *to) {
  Node *node = NodeOf(from);
  node->references += 1;
  WriteNode(node, to);
}

void DestroyState(std::byte *state) { Unreference(NodeOf(state)); }

class RecurrentLmScorer : public Scorer {
 public:
  RecurrentLmScorer(std::unique_ptr<RecurrentLmStep> step,
                    RecurrentLmVocabulary vocabulary)
      : step_(std::move(step)), vocabulary_(std::move(vocabulary)) {
    info_.unit = vocabulary_.token_units ? ScorerUnit::kAsrToken
                                         : ScorerUnit::kCharacter;
    info_.state_bytes = sizeof(Node *);
    info_.copy_state = &CopyState;
    info_.destroy_state = &DestroyState;

    // The node after the sentence boundary is the same for every utterance.
    NodePool pool;
    Node *root = pool.New();
    root->unit = vocabulary_.boundary_id;
    Request(root);
    Compute();
    root_states_ = root->states;
    root_log_probs_ = root->log_probs;
    Unreference(root);
  }

  const ScorerInfo &Info() const override { return info_; }

  std::unique_ptr<ScorerUtterance> NewUtterance() override {
    return std::make_unique<NodePool>();
  }

  void Start(ScorerUtterance *utterance, std::byte *state) override {
    Node *root = static_cast<NodePool *>(utterance)->New();
    root->unit = vocabulary_.boundary_id;
    root->states = root_states_;
    root->log_probs = root_log_probs_;
    root->computed = true;
    WriteNode(root, state);
  }

  void Advance(Span<const ScorerExtension> extensions,
               ScorerAdvanceResult *results) override {
    for (const ScorerExtension &extension : extensions) {
      Request(NodeOf(extension.parent.bytes));
    }
    Compute();
    for (size_t i = 0; i != extensions.size; ++i) {
      Node *parent = NodeOf(extensions[i].parent.bytes);
      int32_t unit = IdOf(extensions[i].unit);
      results[i].log_score = parent->log_probs[unit];
      results[i].rejected = false;

      Node *child = parent->pool->New();
      child->unit = unit;
      child->parent = parent;
      parent->references += 1;
      WriteNode(child, extensions[i].child);
    }
  }

  void Finish(Span<const ScorerStateRef> states,
              ScorerFinalScore *results) override {
    if (!vocabulary_.score_end) {
      for (size_t i = 0; i != states.size; ++i) results[i] = {0.0f, true};
      return;
    }
    for (const ScorerStateRef &state : states) Request(NodeOf(state.bytes));
    Compute();
    for (size_t i = 0; i != states.size; ++i) {
      results[i].log_score =
          NodeOf(states[i].bytes)->log_probs[vocabulary_.boundary_id];
      results[i].accepted = true;
    }
  }

 private:
  int32_t IdOf(UnitId unit) const {
    if (vocabulary_.token_units) {
      int32_t id = unit + vocabulary_.token_id_offset;
      return unit >= 0 && id >= 0 && id < step_->VocabSize()
                 ? id
                 : vocabulary_.unknown_id;
    }
    int32_t id = vocabulary_.id_of_char.Lookup(unit);
    return id < 0 ? vocabulary_.unknown_id : id;
  }

  void Request(Node *node) {
    if (node->computed || node->queued) return;
    node->queued = true;
    batch_.push_back(node);
  }

  // One model step for every requested node.
  void Compute() {
    if (batch_.empty()) return;
    const auto &shapes = step_->StateShapes();
    size_t batch = batch_.size();
    units_.resize(batch);
    states_in_.resize(shapes.size());
    for (size_t k = 0; k != shapes.size(); ++k) {
      states_in_[k].assign(shapes[k].layers * batch * shapes[k].hidden, 0.0f);
    }
    for (size_t i = 0; i != batch; ++i) {
      const Node *node = batch_[i];
      units_[i] = node->unit;
      if (node->parent == nullptr) continue;  // Steps from the zero state.
      for (size_t k = 0; k != shapes.size(); ++k) {
        size_t hidden = shapes[k].hidden;
        for (int32_t layer = 0; layer != shapes[k].layers; ++layer) {
          std::copy_n(node->parent->states[k].begin() + layer * hidden, hidden,
                      states_in_[k].begin() + (layer * batch + i) * hidden);
        }
      }
    }

    const float *log_probs = nullptr;
    step_->Run(units_, states_in_, &log_probs, &states_out_);

    size_t vocab_size = step_->VocabSize();
    for (size_t i = 0; i != batch; ++i) {
      Node *node = batch_[i];
      node->log_probs.assign(log_probs + i * vocab_size,
                             log_probs + (i + 1) * vocab_size);
      node->states.resize(shapes.size());
      for (size_t k = 0; k != shapes.size(); ++k) {
        size_t hidden = shapes[k].hidden;
        node->states[k].resize(shapes[k].layers * hidden);
        for (int32_t layer = 0; layer != shapes[k].layers; ++layer) {
          std::copy_n(states_out_[k] + (layer * batch + i) * hidden, hidden,
                      node->states[k].begin() + layer * hidden);
        }
      }
      node->computed = true;
      node->queued = false;
    }
    // A computed node no longer needs its parent's state.
    for (Node *node : batch_) {
      Node *parent = node->parent;
      node->parent = nullptr;
      Unreference(parent);
    }
    batch_.clear();
  }

  ScorerInfo info_;
  std::unique_ptr<RecurrentLmStep> step_;
  RecurrentLmVocabulary vocabulary_;
  std::vector<std::vector<float>> root_states_;
  std::vector<float> root_log_probs_;

  std::vector<Node *> batch_;
  std::vector<int64_t> units_;
  std::vector<std::vector<float>> states_in_;
  std::vector<const float *> states_out_;
};

}  // namespace

std::unique_ptr<Scorer> MakeRecurrentLmScorer(
    std::unique_ptr<RecurrentLmStep> step, RecurrentLmVocabulary vocabulary) {
  return std::make_unique<RecurrentLmScorer>(std::move(step),
                                             std::move(vocabulary));
}

int32_t RecurrentLmLiveNodes() { return g_live_nodes; }

}  // namespace sherpa_onnx
