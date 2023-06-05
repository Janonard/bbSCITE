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
#include <CL/sycl.hpp>
#include <array>
#include <bit>
#include <bitset>
#include <cmath>
#include <cstdint>

namespace bbSCITE {
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
  static constexpr uint32_t n_quadwords = 2;
  static constexpr uint32_t n_cells = n_quadwords * 64; // 128
  static constexpr uint32_t n_genes = n_cells - 1;      // 127
  static constexpr uint32_t n_nodes = n_cells;          // 128
#if PC2_SYSTEM == 1
  static constexpr uint32_t n_vec_elems = 8;
#elif PC2_SYSTEM == 2
  static constexpr uint32_t n_vec_elems = 1;
#else
  static constexpr uint32_t n_vec_elems = 1;
#endif

  static_assert(n_cells % n_vec_elems == 0);

  using float_vec_t = cl::sycl::vec<float, n_vec_elems>;
  using uint64_vec_t = cl::sycl::vec<uint64_t, n_vec_elems>;
  using AncestorMatrix = std::array<std::array<uint64_t, n_nodes>, n_quadwords>;
  using MutationDataAccessor =
      cl::sycl::accessor<uint64_t, 2, cl::sycl::access::mode::read,
                         cl::sycl::access::target::device>;

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
                MutationDataAccessor is_mutated, MutationDataAccessor is_known)
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

  float logscore_tree(AncestorMatrix const &descendant_matrix) const {
#if PC2_SYSTEM == 1
    return logscore_tree_avx512(descendant_matrix);
#else
    return logscore_tree_avx2(descendant_matrix);
#endif
  }

private:
  /**
   * @brief Compute the log-likelihood of the given mutation tree.
   */
  float logscore_tree_avx512(AncestorMatrix const &descendant_matrix) const {
    float tree_score = 0.0;

    for (uint32_t node_i = 0; node_i < n_nodes; node_i++) {
      float_vec_t individual_scores[n_cells / n_vec_elems];
      for (uint32_t cell_vec_i = 0; cell_vec_i < n_cells / n_vec_elems;
           cell_vec_i++) {
        individual_scores[cell_vec_i] = 0;
      }

      for (uint32_t quadword_i = 0; quadword_i < n_quadwords; quadword_i++) {
        uint64_vec_t is_ancestor_vec(descendant_matrix[quadword_i][node_i]);

        for (uint32_t cell_vec_i = 0; cell_vec_i < n_cells / n_vec_elems;
             cell_vec_i++) {
          uint64_vec_t is_mutated_vec;
          uint64_vec_t is_known_vec;

          for (uint32_t element = 0; element < n_vec_elems; element++) {
            is_mutated_vec[element] =
                is_mutated[quadword_i][cell_vec_i * n_vec_elems + element];
            is_known_vec[element] =
                is_known[quadword_i][cell_vec_i * n_vec_elems + element];
          }

          for (uint32_t i_posterior = 0; i_posterior < 2; i_posterior++) {
            for (uint32_t i_prior = 0; i_prior < 2; i_prior++) {
              uint64_vec_t occurrences_vec =
                  is_known_vec &
                  (i_posterior == 1 ? is_mutated_vec : ~is_mutated_vec) &
                  (i_prior == 1 ? is_ancestor_vec : ~is_ancestor_vec);

              float_vec_t popcount_vec;
              for (uint32_t element = 0; element < n_vec_elems; element++) {
                popcount_vec[element] = std::popcount(occurrences_vec[element]);
              }

              individual_scores[cell_vec_i] +=
                  popcount_vec * log_error_probabilities[i_posterior][i_prior];
            }
          }
        }
      }

      float node_score = -std::numeric_limits<float>::infinity();
      for (uint32_t cell_i = 0; cell_i < n_cells; cell_i++) {
        node_score = std::max(
            node_score,
            individual_scores[cell_i / n_vec_elems][cell_i % n_vec_elems]);
      }
      tree_score += node_score;
    }

    return tree_score;
  }

  float logscore_tree_avx2(AncestorMatrix const &descendant_matrix) const {
    float tree_score = 0.0;

    for (uint32_t node_i = 0; node_i < n_genes + 1; node_i++) {
      float node_score = -std::numeric_limits<float>::infinity();

#pragma unroll 4
      for (uint32_t cell_i = 0; cell_i < n_cells; cell_i++) {
        float individual_score = 0.0;

#pragma unroll
        for (uint32_t i_posterior = 0; i_posterior < 2; i_posterior++) {
#pragma unroll
          for (uint32_t i_prior = 0; i_prior < 2; i_prior++) {
            uint32_t n_occurrences = 0;

#pragma unroll
            for (uint32_t quadword_i = 0; quadword_i < n_quadwords;
                 quadword_i++) {
              uint64_t occurrence_vector = is_known[quadword_i][cell_i];

              if (i_posterior == 1) {
                occurrence_vector &= is_mutated[quadword_i][cell_i];
              } else {
                occurrence_vector &= ~is_mutated[quadword_i][cell_i];
              }

              if (i_prior == 1) {
                occurrence_vector &= descendant_matrix[quadword_i][node_i];
              } else {
                occurrence_vector &= ~descendant_matrix[quadword_i][node_i];
              }

              n_occurrences += std::popcount(occurrence_vector);
            }

            individual_score += float(n_occurrences) *
                                log_error_probabilities[i_posterior][i_prior];
          }
        }

        node_score = std::max(node_score, individual_score);
      }

      tree_score += node_score;
    }

    return tree_score;
  }

private:
  float log_error_probabilities[2][2];
  MutationDataAccessor is_mutated, is_known;
};
} // namespace bbSCITE