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
#include "StaticMatrix.hpp"
#include <CL/sycl.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>

namespace ffSCITE {
/**
 * @brief Class that calculates the likelihood of a tree, relative to given
 * mutation data.
 *
 * An object of this class takes mutation data from single cell sequencing and
 * computes how likely it is that a given chain tree (combination of error rate
 * and mutation tree) is correct. Logically, this is done by finding the most
 * likely attachment node for every cell and multiplying the likelihoods for
 * every cell. However, in order to eliminate expensive power operations, the
 * algorithms internally work on a logarithmic level where values can be added
 * instead of multiplied and multiplied instead of powered.
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
   * @brief Re
   */
  using AncestryVector = typename MutationTreeImpl::AncestryVector;

  /**
   * @brief Data type of the entries in the mutation data matrix.
   *
   * 0b00 stands for "no mutation found", 0b01 stands for "mutation found", 0b10
   * stands for "no data", and 0b11 is currently unused.
   */
  using MutationDataWord = ac_int<2 * max_n_genes, false>;

  using MutationDataMatrix = std::array<MutationDataWord, max_n_cells>;

  /**
   * @brief Shorthand for the occurrence matrix type.
   */
  using OccurrenceMatrix = StaticMatrix<uint32_t, 3, 2>;

  /**
   * @brief Shorthand for the mutation data input accessor.
   *
   */
  using MutationDataAccessor =
      cl::sycl::accessor<MutationDataWord, 1, cl::sycl::access::mode::read,
                         access_target>;

  /**
   * @brief Initialize a new tree scorer
   *
   * This new tree scorer uses the initial assumptions over the alpha and beta
   * error rates and also load the mutation data.
   *
   * @param alpha_mean The initial assumption of the alpha error rate (false
   * positive)
   * @param beta_mean The initial assumption of the beta error rate
   * (false negative)
   * @param prior_sd The assumed standard derivation of the beta error rate
   * @param data An accessor to the mutation input data. The number of cells and
   * genes is inferred from the accessor range.
   */
  TreeScorer(float alpha_mean, float beta_mean, float beta_sd, uint32_t n_cells,
             uint32_t n_genes, MutationDataAccessor data_ac,
             MutationDataMatrix &data)
      : log_error_probabilities(), bpriora(0.0), bpriorb(0.0), data(data),
        n_cells(n_cells), n_genes(n_genes) {
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
      data[cell_i] = data_ac[cell_i];
    }
  }

  float logscore_beta(float beta) const {
    return std::log(std::tgamma(bpriora + bpriorb)) +
           (bpriora - 1) * std::log(beta) + (bpriorb - 1) * std::log(1 - beta) -
           std::log(std::tgamma(bpriora)) - std::log(std::tgamma(bpriorb));
  }

  float logscore_tree(MutationTreeImpl const &tree) {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(tree.get_n_nodes() == n_genes + 1);
#endif

    log_error_probabilities[0][1] = std::log(tree.get_beta());
    log_error_probabilities[1][1] = std::log(1.0 - tree.get_beta());

    float individual_scores[max_n_cells][max_n_genes + 1];

    [[intel::loop_coalesce(2)]] for (uint32_t cell_i = 0; cell_i < max_n_cells;
                                     cell_i++) {
      MutationDataWord observed_mutations = data[cell_i];

#pragma unroll 8
      for (uint32_t node_i = 0; node_i < max_n_genes + 1; node_i++) {
        AncestryVector true_mutations = tree.get_ancestors(node_i);

        OccurrenceMatrix occurrences(0);
#pragma unroll
        for (uint32_t gene_i = 0; gene_i < max_n_genes; gene_i++) {
          if (gene_i < n_genes) {
            ac_int<2, false> posterior =
                observed_mutations.template slc<2>(gene_i << 1);
            ac_int<1, false> prior = true_mutations[gene_i];
            occurrences[{posterior, prior}]++;
          }
        }

        individual_scores[cell_i][node_i] =
            get_logscore_of_occurrences(occurrences);
      }
    }

    float cell_scores[max_n_cells];

    for (uint32_t cell_i = 0; cell_i < max_n_cells; cell_i++) {
      float best_cell_score = individual_scores[cell_i][0];
#pragma unroll
      for (uint32_t node_i = 0; node_i < max_n_genes + 1; node_i++) {
        if (node_i < n_genes + 1 &&
            individual_scores[cell_i][node_i] > best_cell_score) {
          best_cell_score = individual_scores[cell_i][node_i];
        }
      }
      cell_scores[cell_i] = best_cell_score;
    }

    float tree_score = 0.0;

#pragma unroll
    for (uint32_t cell_i = 0; cell_i < max_n_cells; cell_i++) {
      if (cell_i < n_cells) {
        tree_score += cell_scores[cell_i];
      }
    }

    float beta_score = logscore_beta(tree.get_beta());

    return tree_score + beta_score;
  }

  /**
   * @brief Compute the logarithm of the likelihood score for the given
   * occurrences.
   *
   * @param occurrences The occurrences to compute the likelihood of.
   * @return The log-likelihood for the given occurrences.
   */
  float get_logscore_of_occurrences(OccurrenceMatrix occurrences) {
    float logscore = 0.0;
#pragma unroll
    for (uint32_t i_posterior = 0; i_posterior < 2; i_posterior++) {
#pragma unroll
      for (uint32_t i_prior = 0; i_prior < 2; i_prior++) {
        logscore += occurrences[{i_posterior, i_prior}] *
                    log_error_probabilities[i_posterior][i_prior];
      }
    }
    return logscore;
  }

private:
  float log_error_probabilities[2][2];
  float bpriora, bpriorb;
  MutationDataMatrix &data;
  uint32_t n_cells, n_genes;
};
} // namespace ffSCITE