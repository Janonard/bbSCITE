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
#include "MutationTree.hpp"

#include <CL/sycl.hpp>
#include <bit>
#include <sycl/ext/intel/ac_types/ac_int.hpp>

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
template <uint32_t max_n_cells, uint32_t max_n_genes,
          cl::sycl::access::target access_target =
              cl::sycl::access::target::device>
class TreeScorer {
public:
  /**
   * @brief Shorthand for the parent vector class in use.
   */
  using MutationTreeImpl = MutationTree<max_n_genes>;

  /**
   * @brief Type of the ancestor matrix rows.
   */
  using AncestryVector = typename MutationTreeImpl::AncestryVector;

  /**
   * @brief Internal representation of the input mutation data.
   */
  using MutationDataMatrix = std::array<AncestryVector, max_n_cells>;

  /**
   * @brief Shorthand for the mutation data input accessor.
   */
  using MutationDataAccessor =
      cl::sycl::accessor<AncestryVector, 1, cl::sycl::access::mode::read,
                         access_target>;

  using popcount_t = ac_int<std::bit_width(max_n_genes), false>;

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
  TreeScorer(float alpha_mean, float beta_mean, float beta_sd, uint32_t n_cells,
             uint32_t n_genes, MutationDataAccessor is_mutated_ac,
             MutationDataAccessor is_known_ac, MutationDataMatrix &is_mutated,
             MutationDataMatrix &is_known)
      : log_error_probabilities(), bpriora(0.0), bpriorb(0.0),
        is_mutated(is_mutated), is_known(is_known), n_cells(n_cells),
        n_genes(n_genes) {
    // mutation not observed, not present
    log_error_probabilities[0][0] = std::log(1.0 - alpha_mean);
    // mutation observed, not present
    log_error_probabilities[1][0] = std::log(alpha_mean);
    // mutation not observed, but present
    log_error_probabilities[0][1] = std::log(beta_mean);
    // mutation observed and present
    log_error_probabilities[1][1] = std::log(1.0 - beta_mean);

    bpriora =
        ((1 - beta_mean) * std::pow(beta_mean, 2) / std::pow(beta_sd, 2)) -
        beta_mean;
    bpriorb = bpriora * ((1 / beta_mean) - 1);

    for (uint32_t cell_i = 0; cell_i < max_n_cells; cell_i++) {
      is_mutated[cell_i] = is_mutated_ac[cell_i];
      is_known[cell_i] = is_known_ac[cell_i];
    }
  }

  /**
   * @brief Compute the log-likelihood of the given beta probability.
   */
  float logscore_beta(float beta) const {
    return std::log(std::tgamma(bpriora + bpriorb)) +
           (bpriora - 1) * std::log(beta) + (bpriorb - 1) * std::log(1 - beta) -
           std::log(std::tgamma(bpriora)) - std::log(std::tgamma(bpriorb));
  }

  /**
   * @brief Compute the log-likelihood of the given mutation tree.
   */
  float logscore_tree(MutationTreeImpl const &tree) {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(tree.get_n_nodes() == n_genes + 1);
#endif

    log_error_probabilities[0][1] = std::log(tree.get_beta());
    log_error_probabilities[1][1] = std::log(1.0 - tree.get_beta());

    /*
     * First, we compute the likelihood for every cell-node combination, in
     * other words the likelihood that cell i is attached to node j, and store
     * those likelihoods in `individual_scores`. Then, we execute a maximum
     * reduction to find the most-likelihood attachment node for every cell and
     * store this cell maximum in `cell_scores.` Lastly, we sum all those local
     * likelihoods up to obtain the likelihood of the entire tree.
     *
     * This may look unnecessarily complicated compared to the CPU version, but
     * this structure is easier to implement on FPGAs and achieves higher
     * throughput.
     */

    float individual_scores[max_n_cells];

    for (uint32_t node_i = 0; node_i < n_genes + 1; node_i++) {
      AncestryVector is_ancestor = tree.get_ancestors(node_i);

#pragma unroll
      for (uint32_t cell_i = 0; cell_i < max_n_cells; cell_i++) {
        AncestryVector is_mutated = this->is_mutated[cell_i];
        AncestryVector is_known = this->is_known[cell_i];
        float individual_score = 0.0;

#pragma unroll
        for (uint32_t i_posterior = 0; i_posterior < 2; i_posterior++) {
#pragma unroll
          for (uint32_t i_prior = 0; i_prior < 2; i_prior++) {
            AncestryVector occurrence_vector =
                is_known &
                (i_posterior == 1 ? is_mutated : is_mutated.bit_complement()) &
                (i_prior == 1 ? is_ancestor : is_ancestor.bit_complement());
            popcount_t n_occurrences = 0;

            #pragma unroll
            for (uint32_t bit_offset = 0; bit_offset < max_n_genes+1; bit_offset += 64) {
              n_occurrences += std::popcount(occurrence_vector.template slc<64>(bit_offset).to_uint64());
            }

            individual_score += float(n_occurrences) *
                                log_error_probabilities[i_posterior][i_prior];
          }
        }

        float old_score = individual_scores[cell_i];
        float new_score;
        if (node_i == 0) {
          new_score = individual_score;
        } else if (node_i < n_genes + 1) {
          new_score = std::max(old_score, individual_score);
        } else {
          new_score = old_score;
        }
        individual_scores[cell_i] = new_score;
      }
    }

    float tree_score = 0.0;

#pragma unroll
    for (uint32_t cell_i = 0; cell_i < max_n_cells; cell_i++) {
      if (cell_i < n_cells) {
        tree_score += individual_scores[cell_i];
      }
    }

    float beta_score = logscore_beta(tree.get_beta());

    return tree_score + beta_score;
  }

private:
  float log_error_probabilities[2][2];
  float bpriora, bpriorb;
  MutationDataMatrix &is_mutated;
  MutationDataMatrix &is_known;
  uint32_t n_cells, n_genes;
};
} // namespace ffSCITE