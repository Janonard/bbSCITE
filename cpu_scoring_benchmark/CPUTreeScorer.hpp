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
#include <array>
#include <bit>
#include <bitset>
#include <cmath>
#include <cstdint>

namespace ffSCITE {
/**
 * @brief Class that calculates the likelihood of a tree, relative to given
 * mutation data.
 *
 * An object of this class takes mutation data from single cell sequencing and
 * computes how likely it is that a given mutation tree is correct. Logically,
 * this is done by finding the most likely attachment node for every cell and
 * multiplying the likelihoods for every cell. However, in order to eliminate
 * expensive power operations, the algorithms internally work on a logarithmic
 * level where values can be added instead of multiplied and multiplied instead
 * of powered.
 *
 * @tparam max_n_cells The maximum number of cells processable.
 * @tparam max_n_genes The maximum number of genes processable.
 * @tparam access_target The access target where the input data resides.
 * Defaults to the device buffer but may be changed, e.g. for testing.
 */
class CPUTreeScorer {
public:
  static constexpr uint32_t n_words = 2;
  static constexpr uint32_t n_cells = n_words * 64; // 128
  static constexpr uint32_t n_genes = n_cells - 1; // 127
  static constexpr uint32_t n_nodes = n_cells; // 128

  using AncestorMatrix = std::array<std::array<uint64_t, n_nodes>, n_words>;
  using MutationDataMatrix = std::array<std::array<uint64_t, n_cells>, n_words>;

  /**
   * @brief Initialize a new tree scorer
   *
   * This new tree scorer uses the initial assumptions over the alpha and beta
   * error rates and also loads the mutation data.
   *
   * @param alpha_mean The initial assumption of the alpha error rate (false
   * positive)
   * @param beta_mean The initial assumption of the beta error rate
   * (false negative)
   * @param prior_sd The assumed standard derivation of the beta error rate
   * @param is_mutated_ac An accessor to the mutation status bit vectors.
   * @param is_known_ac An accessor to the mutation knowledge bit vectors.
   * @param is_mutated A reference to the local data memory block to use for the
   * mutation status bit vectors.
   * @param is_known A reference to the local data memory block to use for the
   * mutation knowledge bit vectors.
   */
  CPUTreeScorer(float alpha_mean, float beta_mean, float beta_sd,
                MutationDataMatrix const &is_mutated,
                MutationDataMatrix const &is_known)
      : log_error_probabilities(), is_mutated(is_mutated), is_known(is_known) {
    // mutation not observed, not present
    log_error_probabilities[0][0] = std::log(1.0 - alpha_mean);
    // mutation observed, not present
    log_error_probabilities[1][0] = std::log(alpha_mean);
    // mutation not observed, but present
    log_error_probabilities[0][1] = std::log(beta_mean);
    // mutation observed and present
    log_error_probabilities[1][1] = std::log(1.0 - beta_mean);
  }

  /**
   * @brief Compute the log-likelihood of the given mutation tree.
   */
  float logscore_tree(AncestorMatrix const &descendant_matrix) const {
    float tree_score = 0.0;

    for (uint32_t node_i = 0; node_i < n_nodes; node_i++) {
      float individual_scores[n_cells];
      for (uint32_t cell_i = 0; cell_i < n_cells; cell_i++) {
        individual_scores[cell_i] = 0;
      }

      for (uint32_t word_i = 0; word_i < n_words; word_i++) {
        uint64_t is_ancestor = descendant_matrix[word_i][node_i];

        for (uint32_t cell_i = 0; cell_i < n_cells; cell_i++) {
          uint64_t is_mutated = this->is_mutated[word_i][cell_i];
          uint64_t is_known = this->is_known[word_i][cell_i];

          for (uint32_t i_posterior = 0; i_posterior < 2; i_posterior++) {
            for (uint32_t i_prior = 0; i_prior < 2; i_prior++) {
              uint64_t occurrence_vector =
                  is_known & (i_posterior == 1 ? is_mutated : ~is_mutated) &
                  (i_prior == 1 ? is_ancestor : ~is_ancestor);
                
              individual_scores[cell_i] += std::popcount(occurrence_vector) *
                                  log_error_probabilities[i_posterior][i_prior];
            }
          }
        }
      }

      float node_score = -std::numeric_limits<float>::infinity();
      for (uint32_t cell_i = 0; cell_i < n_cells; cell_i++) {
        node_score = std::max(node_score, individual_scores[cell_i]);
      }
      tree_score += node_score;
    }

    return tree_score;
  }

private:
  float log_error_probabilities[2][2];
  MutationDataMatrix is_mutated, is_known;
};
} // namespace ffSCITE