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
#include "MoveType.hpp"
#include "Parameters.hpp"

#include <array>
#include <random>

namespace ffSCITE {
struct RawMoveSample {
  MoveType move_type;
  uint32_t raw_v, raw_w;
  float raw_descendant_of_v, raw_nondescendant_of_v;
  float beta_jump;
  float acceptance_level;
} __attribute__((aligned(32)));

class RawMoveDistribution {
public:
  RawMoveDistribution(uint32_t n_nodes, Parameters const &parameters)
      : prob_beta_change(parameters.get_prob_beta_change()),
        prob_prune_n_reattach(parameters.get_prob_prune_n_reattach()),
        prob_swap_nodes(parameters.get_prob_swap_nodes()),
        v_distribution(0, n_nodes - 2), w_distribution(0, n_nodes - 3),
        unit_distribution(0, 1),
        normal_distribution(0, parameters.get_beta_jump_sd()) {}

  template <typename RNG> RawMoveSample operator()(RNG &rng) {
    MoveType move_type = sample_move(rng);
    std::array<uint32_t, 2> v_and_w = sample_nonroot_nodepair(rng);

    return RawMoveSample{
        .move_type = move_type,
        .raw_v = v_and_w[0],
        .raw_w = v_and_w[1],
        .raw_descendant_of_v = unit_distribution(rng),
        .raw_nondescendant_of_v = unit_distribution(rng),
        .beta_jump = normal_distribution(rng),
        .acceptance_level = unit_distribution(rng),
    };
  }

  /**
   * @brief Sample one of the random move types.
   *
   * @tparam The type of URNG to use.
   * @param rng The URNG instance to use.
   * @return MoveType The sampled move type.
   */
  template <typename RNG> MoveType sample_move(RNG &rng) {
    float change_type_draw = unit_distribution(rng);
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
   * @brief Sample two distinct nodes uniformly from the tree.
   *
   * The first node v is sampled uniformly from the entire tree and the second
   * node w is sampled from all nodes except v. v and w are either unrelated or
   * w is an ancestor of v.
   *
   * This method works equally well on CPUs and on FPGAs.
   *
   * @tparam The type of URNG to use.
   * @param rng The URNG instance to use.
   * @return std::array<uint32_t, 2> Two distinct nodes from the tree.
   */
  template <typename RNG>
  std::array<uint32_t, 2> sample_nonroot_nodepair(RNG &rng) {
    // excluding n_nodes - 1, the root.
    uint32_t v = v_distribution(rng);
    uint32_t w = w_distribution(rng);
    if (w >= v) {
      w++;
    }
    return {v, w};
  }

private:
  float prob_beta_change, prob_prune_n_reattach, prob_swap_nodes;

  std::uniform_int_distribution<uint32_t> v_distribution, w_distribution;
  std::uniform_real_distribution<float> unit_distribution;
  std::normal_distribution<float> normal_distribution;
};
} // namespace ffSCITE