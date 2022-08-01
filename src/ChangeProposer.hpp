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
#include "ChainStepParameters.hpp"
#include "MutationTree.hpp"
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
   * @brief Shorthand for the parent vector type.
   */
  using MutationTreeImpl = MutationTree<max_n_genes>;

  using AncestryVector = typename MutationTreeImpl::AncestryVector;

  /**
   * @brief Shorthand for the maximal number of nodes in the mutation tree.
   */
  static constexpr uint32_t max_n_nodes = MutationTreeImpl::max_n_nodes;

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
  ChangeProposer(RNG rng, float prob_beta_change, float prob_prune_n_reattach,
                 float prob_swap_nodes, float beta_jump_sd)
      : rng(rng), prob_beta_change(prob_beta_change),
        prob_prune_n_reattach(prob_prune_n_reattach),
        prob_swap_nodes(prob_swap_nodes), beta_jump_sd(beta_jump_sd) {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(prob_beta_change + prob_prune_n_reattach + prob_swap_nodes <= 1.0);
#endif
  }

  /**
   * @brief Initialize the change proposer with default parameters.
   *
   * @param rng The URNG instance that is used to sample changes.
   */
  ChangeProposer(RNG rng)
      : rng(rng), prob_beta_change(0.0), prob_prune_n_reattach(0.5),
        prob_swap_nodes(0.45), beta_jump_sd(0.1) {}

  RNG &get_rng() { return rng; }

  /**
   * @brief Sample one of the possible moves with the defined distribution.
   *
   * @return A random move.
   */
  MoveType sample_move() {
    float change_type_draw =
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
  std::array<uint32_t, 2>
  sample_nonroot_nodepair(MutationTreeImpl const &tree) {
    std::array<uint32_t, 2> sampled_nodes;

    // excluding n_nodes - 1, the root.
    sampled_nodes[0] = oneapi::dpl::uniform_int_distribution<uint32_t>(
        0, tree.get_n_nodes() - 2)(rng);
    sampled_nodes[1] = oneapi::dpl::uniform_int_distribution<uint32_t>(
        0, tree.get_n_nodes() - 3)(rng);
    if (sampled_nodes[1] >= sampled_nodes[0]) {
      sampled_nodes[1]++;
    }
    if (tree.is_ancestor(sampled_nodes[0], sampled_nodes[1])) {
      std::swap(sampled_nodes[0], sampled_nodes[1]);
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
   * @param tee The current ancestor matrix of the mutation tree.
   * @param node_i The node from who's descendants or nondescendants this method
   * samples.
   * @param sample_descandent True iff the method is supposed to sample
   * from the nodes descendants. Otherwise, it will sample from the node's
   * nondescendants.
   * @return One of the nodes descendants or nondescendants.
   */
  uint32_t sample_descendant_or_nondescendant(MutationTreeImpl const &tree,
                                              uint32_t node_i,
                                              bool sample_descendant,
                                              bool include_root) {
    AncestryVector descendant = tree.get_descendants(node_i);

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
    uint32_t sum_upper_bound = tree.get_n_nodes() - (include_root ? 0 : 1);
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
      if (i < tree.get_n_nodes() && descendant[i]) {
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
  float change_beta(float old_beta) {
    // Not using float as the normal distribution's output here since it
    // introduced a compiler error as of this writing.
    float new_beta = old_beta + oneapi::dpl::normal_distribution<float>(
                                    0, beta_jump_sd)(rng);
    if (new_beta < 0) {
      new_beta = std::abs(new_beta);
    }
    if (new_beta > 1) {
      new_beta = new_beta - 2 * (new_beta - 1);
    }
    return new_beta;
  }

  ChainStepParameters
  sample_step_parameters(MutationTree<max_n_genes> const &current_tree) {
    std::array<uint32_t, 2> v_and_w = sample_nonroot_nodepair(current_tree);
    uint32_t v = v_and_w[0];
    uint32_t w = v_and_w[1];

    float neighborhood_correction;
    if (current_tree.is_ancestor(w, v)) {
      neighborhood_correction = float(current_tree.get_n_descendants(v)) /
                                float(current_tree.get_n_descendants(w));
    } else {
      neighborhood_correction = 1.0;
    }

    uint32_t parent_of_v, parent_of_w;
    for (uint32_t node_i = 0; node_i < max_n_nodes; node_i++) {
      if (node_i >= current_tree.get_n_nodes()) {
        continue;
      }
      if (current_tree.is_parent(node_i, v)) {
        parent_of_v = node_i;
      }
      if (current_tree.is_parent(node_i, w)) {
        parent_of_w = node_i;
      }
    }

    uint32_t descendant_of_v =
        sample_descendant_or_nondescendant(current_tree, v, true, false);
    uint32_t nondescendant_of_v =
        sample_descendant_or_nondescendant(current_tree, v, false, true);
    MoveType move_type = sample_move();
    float new_beta = change_beta(current_tree.get_beta());
    float acceptance_level =
        oneapi::dpl::uniform_real_distribution(0.0, 1.0)(rng);

    return ChainStepParameters{.v = v,
                               .w = w,
                               .parent_of_v = parent_of_v,
                               .parent_of_w = parent_of_w,
                               .descendant_of_v = descendant_of_v,
                               .nondescendant_of_v = nondescendant_of_v,
                               .move_type = move_type,
                               .new_beta = new_beta,
                               .tree_swap_neighborhood_correction =
                                   neighborhood_correction,
                               .acceptance_level = acceptance_level};
  }

private:
  RNG rng;

  // probabilities for the different move types, prob_swap_subtrees implied.
  float prob_beta_change, prob_prune_n_reattach, prob_swap_nodes;

  float beta_jump_sd;
};
} // namespace ffSCITE