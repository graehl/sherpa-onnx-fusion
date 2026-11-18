// sherpa-onnx/csrc/recurrent-lm-scorer.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/recurrent-lm-scorer.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace sherpa_onnx {

namespace {

std::atomic<int32_t> g_live_nodes(0);

struct NodePool;

struct Node {
  NodePool *pool = nullptr;
  // Held, with a reference, until this node is computed from its state.
  Node *parent = nullptr;
  int32_t references = 0;
  int32_t unit = 0;  // The unit this node follows.
  bool computed = false;
  bool queued = false;
  // After computing: this node's item of its step's result, which holds its
  // recurrent state where the model runs and its log probabilities of every
  // next unit. Shared with the other nodes of that step.
  std::shared_ptr<const RecurrentLmStep::Result> result;
  int32_t item = 0;

  const float *LogProbs() const { return result->LogProbs(item); }
};

// One utterance's nodes. Freed nodes are reused.
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
    node->result.reset();  // the step's result is freed with its last node
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
      : step_(std::move(step)),
        vocabulary_(std::move(vocabulary)),
        max_step_nodes_(MaxStepNodes(vocabulary_, step_->ItemBytes())) {
    info_.unit = vocabulary_.token_units ? ScorerUnit::kAsrToken
                                         : ScorerUnit::kCharacter;
    info_.state_bytes = sizeof(Node *);
    info_.copy_state = &CopyState;
    info_.destroy_state = &DestroyState;

    // The node after the sentence boundary is the same for every utterance.
    NodePool pool;
    Node *root = pool.New();
    root->unit = vocabulary_.boundary_id;
    std::vector<Node *> batch;
    Request(root, &batch);
    Compute(batch);
    root_result_ = root->result;
    Unreference(root);
  }

  const ScorerInfo &Info() const override { return info_; }

  std::unique_ptr<ScorerUtterance> NewUtterance() override {
    return std::make_unique<NodePool>();
  }

  void Start(ScorerUtterance *utterance, std::byte *state) override {
    Node *root = static_cast<NodePool *>(utterance)->New();
    root->unit = vocabulary_.boundary_id;
    root->result = root_result_;
    root->item = 0;
    root->computed = true;
    WriteNode(root, state);
  }

  void Advance(Span<const ScorerExtension> extensions,
               ScorerAdvanceResult *results) override {
    std::vector<Node *> batch;
    for (const ScorerExtension &extension : extensions) {
      Request(NodeOf(extension.parent.bytes), &batch);
    }
    Compute(batch);
    for (size_t i = 0; i != extensions.size; ++i) {
      Node *parent = NodeOf(extensions[i].parent.bytes);
      int32_t unit = IdOf(extensions[i].unit);
      results[i].log_score =
          std::max(vocabulary_.score_floor, parent->LogProbs()[unit]);
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
    std::vector<Node *> batch;
    for (const ScorerStateRef &state : states) {
      Request(NodeOf(state.bytes), &batch);
    }
    Compute(batch);
    for (size_t i = 0; i != states.size; ++i) {
      results[i].log_score = std::max(
          vocabulary_.score_floor,
          NodeOf(states[i].bytes)->LogProbs()[vocabulary_.boundary_id]);
      results[i].accepted = true;
    }
  }

 private:
  // The most nodes one step takes under both limits, at least one.
  static size_t MaxStepNodes(const RecurrentLmVocabulary &vocabulary,
                             size_t item_bytes) {
    size_t nodes = std::numeric_limits<size_t>::max();
    if (vocabulary.max_batch > 0) {
      nodes = static_cast<size_t>(vocabulary.max_batch);
    }
    if (vocabulary.max_batch_bytes > 0 && item_bytes > 0) {
      nodes = std::min(nodes, vocabulary.max_batch_bytes / item_bytes);
    }
    return std::max<size_t>(nodes, 1);
  }

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

  // Adds a node that still needs its model step to one call's batch.
  static void Request(Node *node, std::vector<Node *> *nodes) {
    if (node->computed || node->queued) return;
    node->queued = true;
    nodes->push_back(node);
  }

  // One model step for every requested node. Everything it allocates
  // belongs to this call, so calls for different utterances can overlap.
  void Compute(const std::vector<Node *> &nodes) const {
    for (size_t first = 0; first < nodes.size(); first += max_step_nodes_) {
      const size_t batch = std::min(max_step_nodes_, nodes.size() - first);
      std::vector<int64_t> units(batch);
      std::vector<RecurrentLmStep::Parent> parents(batch);
      for (size_t i = 0; i != batch; ++i) {
        const Node *node = nodes[first + i];
        units[i] = node->unit;
        if (node->parent != nullptr) {  // else steps from the zero state
          parents[i] = {node->parent->result.get(), node->parent->item};
        }
      }

      std::shared_ptr<const RecurrentLmStep::Result> result =
          step_->Run(units, parents);
      for (size_t i = 0; i != batch; ++i) {
        Node *node = nodes[first + i];
        node->result = result;
        node->item = static_cast<int32_t>(i);
        node->computed = true;
        node->queued = false;
      }
    }
    // A computed node no longer needs its parent's state.
    for (Node *node : nodes) {
      Node *parent = node->parent;
      node->parent = nullptr;
      Unreference(parent);
    }
  }

  ScorerInfo info_;
  std::unique_ptr<RecurrentLmStep> step_;
  RecurrentLmVocabulary vocabulary_;
  size_t max_step_nodes_;
  // The node after the sentence boundary, shared by every utterance.
  std::shared_ptr<const RecurrentLmStep::Result> root_result_;
};

}  // namespace

std::unique_ptr<Scorer> MakeRecurrentLmScorer(
    std::unique_ptr<RecurrentLmStep> step, RecurrentLmVocabulary vocabulary) {
  return std::make_unique<RecurrentLmScorer>(std::move(step),
                                             std::move(vocabulary));
}

int32_t RecurrentLmLiveNodes() { return g_live_nodes; }

}  // namespace sherpa_onnx
