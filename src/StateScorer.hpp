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
#include "StaticMatrix.hpp"
#include <sycl/ext/intel/ac_types/ac_int.hpp>

namespace ffSCITE {
/**
 * @brief Class that calculates the likelihood of a state, relative to given
 * mutation data.
 *
 * An object of this class takes mutation data from single cell sequencing and
 * computes how likely it is that a given chain state (combination of error rate
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
class StateScorer {
public:
  /**
   * @brief Shorthand for the chain state class in use.
   */
  using ChainStateImpl = ChainState<max_n_genes>;
  /**
   * @brief Shorthand for the parent vector class in use.
   */
  using ParentVectorImpl = typename ChainStateImpl::ParentVectorImpl;
  /**
   * @brief Shorthand for the ancestor matrix class in use.
   */
  using AncestorMatrixImpl = AncestorMatrix<max_n_genes + 1>;

  /**
   * @brief Data type of the entries in the mutation data matrix.
   *
   * 0b00 stands for "no mutation found", 0b01 stands for "mutation found", 0b10
   * stands for "no data", and 0b11 is currently unused.
   */
  using DataEntry = ac_int<2, false>;

  using DataMatrix =
      std::array<std::array<DataEntry, max_n_genes>, max_n_cells>;

  /**
   * @brief Shorthand for the occurrence matrix type.
   */
  using OccurrenceMatrix = StaticMatrix<uint32_t, 3, 2>;

  /**
   * @brief Shorthand for the mutation data input accessor.
   *
   */
  using MutationDataAccessor =
      cl::sycl::accessor<DataEntry, 2, cl::sycl::access::mode::read,
                         access_target>;

  /**
   * @brief Initialize a new state scorer
   *
   * This new state scorer uses the initial assumptions over the alpha and beta
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
  StateScorer(double alpha_mean, double beta_mean, double beta_sd,
              MutationDataAccessor data_ac, DataMatrix &data)
      : log_error_probabilities(), bpriora(0.0), bpriorb(0.0), data(data),
        n_cells(data_ac.get_range()[0]), n_genes(data_ac.get_range()[1]) {
    // mutation not observed, not present
    log_error_probabilities[0][0] = std::log(1.0 - alpha_mean);
    // mutation observed, not present
    log_error_probabilities[1][0] = std::log(alpha_mean);
    // missing data, mutation not present
    log_error_probabilities[2][0] = std::log(1.0);
    // mutation not observed, but present
    log_error_probabilities[0][1] = std::log(beta_mean);
    // mutation observed and present
    log_error_probabilities[1][1] = std::log(1.0 - beta_mean);
    // missing data, mutation present
    log_error_probabilities[2][1] = std::log(1.0);

    bpriora =
        ((1 - beta_mean) * std::pow(beta_mean, 2) / std::pow(beta_sd, 2)) -
        beta_mean;
    bpriorb = bpriora * ((1 / beta_mean) - 1);

    [[intel::loop_coalesce]] for (uint32_t cell_i = 0; cell_i < max_n_cells;
                                  cell_i++) {
      for (uint32_t gene_i = 0; gene_i < max_n_genes; gene_i++) {
        if (cell_i < n_cells && gene_i < n_genes) {
          this->data[cell_i][gene_i] = data_ac[cell_i][gene_i];
        }
      }
    }
  }

  /**
   * @brief Compute the likelihood score of the given state.
   *
   * This is done by finding the most likely attachment point for every cell and
   * then multiplying the likelihoods of every cell-gene-combination.
   *
   * @param state The state to score.
   * @return The likelihood that the state represents the true mutation history
   * of the sequenced cells.
   */
  double logscore_state(ChainStateImpl const &state) {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(state.mutation_tree.get_n_nodes() == n_genes + 1);
#endif

    log_error_probabilities[0][1] = std::log(state.beta);
    log_error_probabilities[1][1] = std::log(1.0 - state.beta);

    double tree_score = logscore_tree(state.mutation_tree);
    double beta_score = logscore_beta(state.beta);
    return tree_score + beta_score;
  }

  /**
   * @brief Compute the likelihood of the beta error rate being correct.
   *
   * This is based on the a-priori assumption of the beta error rate and it's
   * assumed standard derivation.
   *
   * @param beta The beta error rate to score.
   */
  double logscore_beta(double beta) {
    return std::log(std::tgamma(bpriora + bpriorb)) +
           (bpriora - 1) * std::log(beta) + (bpriorb - 1) * std::log(1 - beta) -
           std::log(std::tgamma(bpriora)) - std::log(std::tgamma(bpriorb));
  }

  double logscore_tree(ParentVectorImpl const &tree) {
    [[intel::fpga_register]] AncestorMatrixImpl am(tree);

    double best_scores[max_n_cells];

    for (uint32_t node_i = 0; node_i < max_n_genes + 1; node_i++) {
      if (node_i >= n_genes + 1) {
        continue;
      }

      std::array<bool, max_n_genes + 1> true_mutations =
          am.get_ancestors(node_i);

      for (uint32_t cell_i = 0; cell_i < max_n_cells; cell_i++) {
        if (cell_i >= n_cells) {
          continue;
        }

        std::array<DataEntry, max_n_genes> observed_mutations = data[cell_i];
        OccurrenceMatrix occurrences(0);

#pragma unroll
        for (uint32_t gene_i = 0; gene_i < max_n_genes; gene_i++) {
          occurrences[{observed_mutations[gene_i], true_mutations[gene_i]}]++;
        }

        double score = get_logscore_of_occurrences(occurrences);

        if (node_i == 0 || score > best_scores[cell_i]) {
          best_scores[cell_i] = score;
        }
      }
    }

    double tree_score = 0.0;

#pragma unroll
    for (uint32_t cell_i = 0; cell_i < max_n_cells; cell_i++) {
      if (cell_i < n_cells) {
        tree_score += best_scores[cell_i];
      }
    }

    return tree_score;
  }

  /**
   * @brief Compute the logarithm of the likelihood score for the given
   * occurrences.
   *
   * @param occurrences The occurrences to compute the likelihood of.
   * @return The log-likelihood for the given occurrences.
   */
  double get_logscore_of_occurrences(OccurrenceMatrix occurrences) {
    double logscore = 0.0;
#pragma unroll
    for (uint32_t i_posterior = 0; i_posterior < 3; i_posterior++) {
#pragma unroll
      for (uint32_t i_prior = 0; i_prior < 2; i_prior++) {
        logscore += occurrences[{i_posterior, i_prior}] *
                    log_error_probabilities[i_posterior][i_prior];
      }
    }
    return logscore;
  }

private:
  double log_error_probabilities[3][2];
  double bpriora, bpriorb;
  DataMatrix &data;
  uint32_t n_cells, n_genes;
};
} // namespace ffSCITE