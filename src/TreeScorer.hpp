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
   * @brief Data type of the mutation data matrix rows.
   *
   * It is implemented as a bitvector to fully utilize the space. There are two
   * bits per gene. 0b00 stands for "no mutation found", 0b01 stands for
   * "mutation found", 0b10 stands for "no data", and 0b11 is currently unused.
   * These bits need to be extracted from the vector, so the (2*i)th bit and
   * (2*i+1)th bit together denote whether the cell has a mutation at the i-th
   * gene or not.
   */
  using MutationDataWord = ac_int<2 * max_n_genes, false>;

  /**
   * @brief Internal representation of the input mutation data.
   *
   * It is implemented as an array of bitvectors to fully utilize the space. In
   * order to find out whether the ith cell has a mutation at the jth gene, one
   * needs to load the ith word from the array and extract the (2*j)th and
   * (2*j+1)th bit from the word.
   */
  using MutationDataMatrix = std::array<MutationDataWord, max_n_cells>;

  /**
   * @brief Shorthand for the occurrence matrix type.
   */
  using OccurrenceMatrix = StaticMatrix<uint32_t, 3, 2>;

  /**
   * @brief Shorthand for the mutation data input accessor.
   */
  using MutationDataAccessor =
      cl::sycl::accessor<MutationDataWord, 1, cl::sycl::access::mode::read,
                         access_target>;

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
   * @param data_ac An accessor to the mutation input data. The number of cells
   * and genes is inferred from the accessor range.
   * @param data A reference to the local mutation data memory block to use.
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

    float individual_scores[max_n_cells][max_n_genes + 1];

    for (uint32_t cell_i = 0; cell_i < max_n_cells; cell_i++) {
      MutationDataWord observed_mutations = data[cell_i];

#pragma unroll
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
   * @brief Compute the log-likelihood for the given occurrences.
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