#pragma once
#include "AncestorMatrix.hpp"
#include "ChainState.hpp"
#include <random>

namespace ffSCITE {
template <uint64_t max_n_nodes, typename RNG> class ChangeProposer {
public:
  using uindex_node_t = typename ParentVector<max_n_nodes>::uindex_node_t;

  ChangeProposer(RNG rng, double prob_beta_change, double prob_prune_n_reattach,
                 double prob_swap_nodes, double beta_jump_sd)
      : rng(rng), prob_beta_change(prob_beta_change),
        prob_prune_n_reattach(prob_prune_n_reattach),
        prob_swap_nodes(prob_swap_nodes), beta_jump_sd(beta_jump_sd) {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(prob_beta_change + prob_prune_n_reattach + prob_swap_nodes < 1.0);
#endif
  }

  ChangeProposer(RNG rng)
      : rng(rng), prob_beta_change(0.55), prob_prune_n_reattach(0.5),
        prob_swap_nodes(0.5), beta_jump_sd(0.1) {}

  enum class MoveType {
    ChangeBeta,
    PruneReattach,
    SwapNodes,
    SwapSubtrees,
  };

  MoveType sample_move() {
    double change_type_draw = std::uniform_real_distribution(0.0, 1.0)(rng);
    if (change_type_draw <= prob_beta_change) {
      return MoveType::ChangeBeta;
    } else if (change_type_draw <= prob_beta_change + prob_prune_n_reattach) {
      return MoveType::PruneReattach;
    } else if (change_type_draw <=
               prob_beta_change + prob_prune_n_reattach + prob_swap_nodes) {
      return MoveType::SwapNodes;
    } else {
      return MoveType::SwapSubtrees;
    }
  }

  std::array<uindex_node_t, 2> sample_nodepair(uindex_node_t n_nodes) {
    std::array<uindex_node_t, 2> sampled_nodes;
    sampled_nodes[0] =
        std::uniform_int_distribution<uint64_t>(0, n_nodes - 1)(rng);
    sampled_nodes[1] =
        std::uniform_int_distribution<uint64_t>(0, n_nodes - 2)(rng);
    if (sampled_nodes[1] >= sampled_nodes[0]) {
      sampled_nodes[1]++;
    }
    return sampled_nodes;
  }

  uindex_node_t sample_descendant_or_nondescendant(
      AncestorMatrix<max_n_nodes> const &ancestor_matrix, uindex_node_t node_i,
      bool sample_descendant) {
    std::array<bool, max_n_nodes> descendant =
        ancestor_matrix.get_descendants(node_i);

    // If we have to sample a nondescendant, we invert the bitvector and
    // continue as if we were to sample a descendant.
    if (!sample_descendant) {
      for (uindex_node_t i = 0; i < max_n_nodes; i++) {
        descendant[i] = !descendant[i];
      }
    }

    // Count the (non)descendants
    uindex_node_t n_descendants = 0;
    for (uindex_node_t i = 0; i < ancestor_matrix.get_n_nodes(); i++) {
      if (descendant[i]) {
        n_descendants++;
      }
    }

    // Sample the occurrence of the (non)descendant to pick. The resulting node
    // will be the `sampled_occurrence_i`th (non)descendant.
    uindex_node_t sampled_occurrence_i =
        std::uniform_int_distribution<uint64_t>(0, n_descendants - 1)(rng);

    // Walk through the (non)descendant bitvector and pick the correct node
    // index.
    uindex_node_t sampled_node_i = 0;
    for (uindex_node_t i = 0; i < ancestor_matrix.get_n_nodes(); i++) {
      if (descendant[i]) {
        if (sampled_occurrence_i == 0) {
          sampled_node_i = i;
          break;
        } else {
          sampled_occurrence_i--;
        }
      }
    }
    return sampled_node_i;
  }

  double change_beta(double old_beta) {
    double new_beta =
        old_beta + std::normal_distribution<double>(0, beta_jump_sd)(rng);
    if (new_beta < 0) {
      new_beta = std::abs(new_beta);
    }
    if (new_beta > 1) {
      new_beta = new_beta - 2 * (new_beta - 1);
    }
    return new_beta;
  }

  uindex_node_t
  prune_and_reattach(ParentVector<max_n_nodes> &parent_vector,
                     AncestorMatrix<max_n_nodes> const &ancestor_matrix) {
    // Pick a node to move.
    uindex_node_t node_to_move_i = std::uniform_int_distribution<uint64_t>(
        0, parent_vector.get_n_nodes() - 1)(rng);

    // Sample one of the node's nondescendants.
    uindex_node_t new_parent_i = sample_descendant_or_nondescendant(
        ancestor_matrix, node_to_move_i, false);

    // Move the node.
    parent_vector.move_subtree(node_to_move_i, new_parent_i);

    return node_to_move_i;
  }

  std::array<uindex_node_t, 2>
  swap_nodes(ParentVector<max_n_nodes> &parent_vector) {
    std::array<uindex_node_t, 2> nodes_to_swap =
        sample_nodepair(parent_vector.get_n_nodes());
    parent_vector.swap_nodes(nodes_to_swap[0], nodes_to_swap[1]);
    return nodes_to_swap;
  }

  std::array<uindex_node_t, 2>
  swap_subtrees(ParentVector<max_n_nodes> &parent_vector,
                AncestorMatrix<max_n_nodes> const &ancestor_matrix,
                double &out_neighborhood_correction) {

    std::array<uindex_node_t, 2> nodes_to_swap =
        sample_nodepair(parent_vector.get_n_nodes());
    uindex_node_t node_a_i = nodes_to_swap[0];
    uindex_node_t node_b_i = nodes_to_swap[1];

    bool distinct_lineages =
        !(ancestor_matrix.is_ancestor(node_a_i, node_b_i) ||
          ancestor_matrix.is_ancestor(node_b_i, node_a_i));
    if (distinct_lineages) {
      // No correction necessary.
      out_neighborhood_correction = 1.0;

      // The nodes are from distinct lineages, we can simply swap the subtrees.
      parent_vector.swap_subtrees(node_a_i, node_b_i);
    } else {
      // The nodes are from a common lineage. We can attach the lower node to
      // the parent of the upper node, but we have to choose something else for
      // the upper node. We therefore sample a descendant of the lower node and
      // attach the upper node to it.

      // Ensure that node a is lower in the tree than node b.
      if (ancestor_matrix.is_ancestor(node_a_i, node_b_i)) {
        std::swap(node_a_i, node_b_i);
      }

      out_neighborhood_correction =
          double(ancestor_matrix.get_n_descendants(node_a_i)) /
          double(ancestor_matrix.get_n_descendants(node_b_i));

      // Sample one of node a's descendants.
      uindex_node_t new_parent_i =
          sample_descendant_or_nondescendant(ancestor_matrix, node_a_i, true);

      // Move node a next to node b.
      parent_vector.move_subtree(node_a_i, parent_vector[node_b_i]);

      // Move node b to its new parent.
      parent_vector.move_subtree(node_b_i, new_parent_i);
    }
    return nodes_to_swap;
  }

  void propose_change(ChainState<max_n_nodes> &state,
                      double &out_neighborhood_correction) {
    AncestorMatrix<max_n_nodes> ancestor_matrix(state.mutation_tree);
    out_neighborhood_correction = 1.0;
    switch (sample_move()) {
    case MoveType::ChangeBeta:
      state.beta = change_beta(state.beta);
      break;
    case MoveType::PruneReattach:
      prune_and_reattach(state.mutation_tree, ancestor_matrix);
      break;
    case MoveType::SwapNodes:
      swap_nodes(state.mutation_tree);
      break;
    case MoveType::SwapSubtrees:
      swap_subtrees(state.mutation_tree, ancestor_matrix,
                    out_neighborhood_correction);
      break;
    default:
      break;
    }
  }

private:
  RNG rng;

  // probabilities for the different move types, prob_swap_subtrees implied.
  double prob_beta_change, prob_prune_n_reattach, prob_swap_nodes;

  double beta_jump_sd;
};
} // namespace ffSCITE