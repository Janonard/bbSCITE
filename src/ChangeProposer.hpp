/*
 * Copyright © 2022 Jan-Oliver Opdenhövel
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <https://www.gnu.org/licenses/>.
 */
#pragma once
#include "AncestorMatrix.hpp"
#include "ChainState.hpp"
#include <oneapi/dpl/random>

namespace ffSCITE {
/**
 * @brief Engine to propose random changes to a chain's state.
 *
 * It uses a URNG to propose a change to a tree presented to it. The individual
 * types of changes are exposed as public methods to ease testing, but users
 * should only use the @ref ChangeProposer::propose_change method.
 *
 * @tparam max_n_genes The maximal number of genes in the dataset.
 * tree.
 * @tparam RNG The URNG to retrieve random numbers for the changes from.
 */
template <uint32_t max_n_genes, typename RNG> class ChangeProposer {
public:
  /**
   * @brief Shorthand for the chain state type.
   */
  using ChainStateImpl = ChainState<max_n_genes>;

  /**
   * @brief Shorthand for the parent vector type.
   */
  using ParentVectorImpl = typename ChainStateImpl::ParentVectorImpl;

  /**
   * @brief Shorthand for the maximal number of nodes in the mutation tree.
   */
  static constexpr uint32_t max_n_nodes = ChainStateImpl::max_n_nodes;

  /**
   * @brief Initialize the change proposer with user-defined parameters.
   *
   * The parameters `prob_beta_change`, `prob_prune_n_reattach`, and
   * `prob_swap_nodes` describe how often which types of changes are proposed.
   * The probability of the fourth move type, the swapping of two subtrees, is
   * implied by the residual probability that the three parameters leave.
   * Therefore, the sum of these three parameters has to be less than 1.0 and
   * this is asserted if the constructor is not compiled for a SYCL device.
   *
   * @param rng The URNG instance that is used to sample changes.
   * @param prob_beta_change The probability that a proposed change changes the
   * beta error rate.
   * @param prob_prune_n_reattach The probability that a
   * proposed change prunes and reattaches a subtree of the mutation tree.
   * @param prob_swap_nodes The probability that a proposed change swap two
   * nodes in the mutation tree.
   * @param beta_jump_sd The standard derivation of
   * the beta error changes.
   */
  ChangeProposer(RNG rng, double prob_beta_change, double prob_prune_n_reattach,
                 double prob_swap_nodes, double beta_jump_sd)
      : rng(rng), prob_beta_change(prob_beta_change),
        prob_prune_n_reattach(prob_prune_n_reattach),
        prob_swap_nodes(prob_swap_nodes), beta_jump_sd(beta_jump_sd) {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(prob_beta_change + prob_prune_n_reattach + prob_swap_nodes < 1.0);
#endif
  }

  /**
   * @brief Initialize the change proposer with default parameters.
   *
   * @param rng The URNG instance that is used to sample changes.
   */
  ChangeProposer(RNG rng)
      : rng(rng), prob_beta_change(0.55), prob_prune_n_reattach(0.5),
        prob_swap_nodes(0.5), beta_jump_sd(0.1) {}

  RNG &get_rng() { return rng; }

  /**
   * @brief The different move types that the proposer may propose.
   */
  enum class MoveType {
    ChangeBeta,
    PruneReattach,
    SwapNodes,
    SwapSubtrees,
  };

  /**
   * @brief Sample one of the possible moves with the defined distribution.
   *
   * @return A random move.
   */
  MoveType sample_move() {
    double change_type_draw =
        oneapi::dpl::uniform_real_distribution(0.0, 1.0)(rng);
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

  /**
   * @brief Sample two distinct non-root nodes.
   *
   * Let V be the set of nodes in the mutation tree and r be the root of the
   * tree. Then, this method can be viewed as a uniformly distributed random
   * variable that samples from {p ⊆ (V \ {r}) | |p| = 2}. However, the order of
   * these two nodes are not guaranteed: The first node may have a lower or
   * higher index than the second node.
   *
   * @return Two random nodes
   */
  std::array<uint32_t, 2> sample_nonroot_nodepair(uint32_t n_nodes) {
    std::array<uint32_t, 2> sampled_nodes;

    // excluding n_nodes - 1, the root.
    sampled_nodes[0] =
        oneapi::dpl::uniform_int_distribution<uint32_t>(0, n_nodes - 2)(rng);
    sampled_nodes[1] =
        oneapi::dpl::uniform_int_distribution<uint32_t>(0, n_nodes - 3)(rng);
    if (sampled_nodes[1] >= sampled_nodes[0]) {
      sampled_nodes[1]++;
    }
    return sampled_nodes;
  }

  /**
   * @brief Sample from one of a node's descendants or nondescendants.
   *
   * Let V be the set of nodes in the mutation tree and i the parameter node.
   * Then, this method can be viewed as a uniformly distributed random variable
   * that samples from {v ∈ V | i \leadsto v} if `sample_descendants` is true,
   * or from {v ∈ V | i \not\leadsto v} if `sample_descendants` is false. If
   * `include_root` is false, the root is removed from the sampled set.
   *
   * @param ancestor_matrix The current ancestor matrix of the mutation tree.
   * @param node_i The node from who's descendants or nondescendants this method
   * samples.
   * @param sample_descandent True iff the method is supposed to sample
   * from the nodes descendants. Otherwise, it will sample from the node's
   * nondescendants.
   * @return One of the nodes descendants or nondescendants.
   */
  uint32_t sample_descendant_or_nondescendant(
      AncestorMatrix<max_n_nodes> const &ancestor_matrix, uint32_t node_i,
      bool sample_descendant, bool include_root) {
    std::array<bool, max_n_nodes> descendant =
        ancestor_matrix.get_descendants(node_i);

    // If we have to sample a nondescendant, we invert the bitvector and
    // continue as if we were to sample a descendant.
    if (!sample_descendant) {
#pragma unroll
      for (uint32_t i = 0; i < max_n_nodes; i++) {
        descendant[i] = !descendant[i];
      }
    }

    // Count the (non)descendants, excluding the root.
    uint32_t n_descendants = 0;
    uint32_t sum_upper_bound =
        ancestor_matrix.get_n_nodes() - (include_root ? 0 : 1);
#pragma unroll
    for (uint32_t i = 0; i < max_n_nodes; i++) {
      if (i < sum_upper_bound && descendant[i]) {
        n_descendants++;
      }
    }

    // Sample the occurrence of the (non)descendant to pick. The resulting node
    // will be the `sampled_occurrence_i`th (non)descendant.
    uint32_t sampled_occurrence_i =
        oneapi::dpl::uniform_int_distribution<uint32_t>(0,
                                                        n_descendants - 1)(rng);

    // Walk through the (non)descendant bitvector and pick the correct node
    // index.
    uint32_t sampled_node_i = 0;
#pragma unroll
    for (uint32_t i = 0; i < max_n_nodes; i++) {
      if (i < ancestor_matrix.get_n_nodes() && descendant[i]) {
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

  /**
   * @brief Propose a new beta error rate.
   *
   * @param old_beta The old beta error rate.
   * @return The newly proposed beta error rate.
   */
  double change_beta(double old_beta) {
    // Not using double as the normal distribution's output here since it introduced a compiler error as of this writing.
    double new_beta = old_beta + oneapi::dpl::normal_distribution<float>(
                                     0, beta_jump_sd)(rng);
    if (new_beta < 0) {
      new_beta = std::abs(new_beta);
    }
    if (new_beta > 1) {
      new_beta = new_beta - 2 * (new_beta - 1);
    }
    return new_beta;
  }

  /**
   * @brief Propose a prune-and-reattach move.
   *
   * This method picks a non-root node in the mutation tree, detaches it from
   * it's current parent and attaches to another node (which may not be a
   * descendant of the moved node). Note that this modifies the referenced
   * parent vector.
   *
   * @param parent_vector The parent vector to modify.
   * @param ancestor_matrix The current ancestor matrix of the mutation tree.
   * @return The index of the the moved node.
   */
  uint32_t
  prune_and_reattach(ParentVector<max_n_nodes> &parent_vector,
                     AncestorMatrix<max_n_nodes> const &ancestor_matrix) {
    // Pick a node to move.
    uint32_t node_to_move_i = oneapi::dpl::uniform_int_distribution<uint32_t>(
        0, parent_vector.get_n_nodes() - 2)(rng);

    // Sample one of the node's nondescendants, including the root.
    uint32_t new_parent_i = sample_descendant_or_nondescendant(
        ancestor_matrix, node_to_move_i, false, true);

    // Move the node.
    parent_vector.move_subtree(node_to_move_i, new_parent_i);

    return node_to_move_i;
  }

  /**
   * @brief Propose a swap-nodes move.
   *
   * This method picks two non-root nodes in the mutation tree and swaps their
   * labels. The structure will remain the same, just two nodes are swapped.
   *
   * @param parent_vector The parent vector to modify.
   * @return The indices of the swapped nodes.
   */
  std::array<uint32_t, 2> swap_nodes(ParentVector<max_n_nodes> &parent_vector) {
    std::array<uint32_t, 2> nodes_to_swap =
        sample_nonroot_nodepair(parent_vector.get_n_nodes());
    parent_vector.swap_nodes(nodes_to_swap[0], nodes_to_swap[1]);
    return nodes_to_swap;
  }

  /**
   * @brief Propose a swap-subtrees move.
   *
   * This method picks two non-root nodes in the mutation tree and swaps their
   * complete subtrees. If these nodes are not ancestors of each other, this
   * involves only swapping the parent vector entries. However, if of the
   * sampled nodes i and j the node i is an ancestor of the node j, then the
   * method samples one of the descendants j and attaches i to instead of the
   * parent of j.
   *
   * If the sampled nodes are related, the neighborhood correction factor is set
   * accordingly, otherwise it is set to 1.0.
   *
   * @param parent_vector The parent vector to modify.
   * @param ancestor_matrix The current ancestor matrix of the mutation tree.
   * @param out_neighborhood_correction Output: Neighborhood correction factor.
   * @return The two swapped/moved nodes.
   */
  std::array<uint32_t, 2>
  swap_subtrees(ParentVector<max_n_nodes> &parent_vector,
                AncestorMatrix<max_n_nodes> const &ancestor_matrix,
                double &out_neighborhood_correction) {

    std::array<uint32_t, 2> nodes_to_swap =
        sample_nonroot_nodepair(parent_vector.get_n_nodes());
    uint32_t node_a_i = nodes_to_swap[0];
    uint32_t node_b_i = nodes_to_swap[1];

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
      uint32_t new_parent_i = sample_descendant_or_nondescendant(
          ancestor_matrix, node_a_i, true, false);

      // Move node a next to node b.
      parent_vector.move_subtree(node_a_i, parent_vector[node_b_i]);

      // Move node b to its new parent.
      parent_vector.move_subtree(node_b_i, new_parent_i);
    }
    return nodes_to_swap;
  }

  /**
   * @brief Propose a random change to the markov chain state.
   *
   * @param state The current state of the chain, which will be modified.
   * @param out_neighborhood_correction Output: The neighborhood correction
   * factor.
   */
  void propose_change(ChainState<max_n_genes> &state,
                      double &out_neighborhood_correction) {
    [[intel::fpga_register]] AncestorMatrix<max_n_nodes> ancestor_matrix(
        state.mutation_tree);
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