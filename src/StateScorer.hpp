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
 * \brief Class that calculates the likelihood of a state, relative to given
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
 * \tparam max_n_cells The maximum number of cells processable.
 * \tparam max_n_genes The maximum number of genes processable.
 */
template <uint64_t max_n_cells, uint64_t max_n_genes,
          cl::sycl::access::target access_target =
              cl::sycl::access::target::device>
class StateScorer {
public:
  /**
   * \brief Shorthand for the chain state class in use.
   */
  using ChainStateImpl = ChainState<max_n_genes>;
  /**
   * \brief Shorthand for the parent vector class in use.
   */
  using ParentVectorImpl = typename ChainStateImpl::ParentVectorImpl;
  /**
   * \brief Shorthand for the ancestor matrix class in use.
   */
  using AncestorMatrixImpl = AncestorMatrix<max_n_genes + 1>;

  /**
   * \brief Data type of the entries in the mutation data matrix.
   *
   * 0b00 stands for "no mutation found", 0b01 stands for "mutation found", 0b10
   * stands for "no data", and 0b11 is currently unused.
   */
  using DataEntry = ac_int<2, false>;

  /**
   * \brief Shorthand for the occurrence matrix type.
   */
  using OccurrenceMatrix = StaticMatrix<uint64_t, 3, 2>;

  using MutationDataAccessor =
      cl::sycl::accessor<DataEntry, 2, cl::sycl::access::mode::read,
                         access_target>;

  /**
   * \brief Initialize a new state scorer
   *
   * This new state scorer uses the initial assumptions over the alpha and beta
   * error rates and also load the mutation data.
   *
   * \param alpha_mean The initial assumption of the alpha error rate (false
   * positive)
   * \param beta_mean The initial assumption of the beta error rate
   * (false negative)
   * \param prior_sd The assumed standard derivation of the beta error rate
   * \param n_cells The number of cells covered by the mutation
   * data matrix.
   * \param n_genes The number of genes covered by the mutation
   * data matrix.
   */
  StateScorer(double alpha_mean, double beta_mean, double beta_sd,
              MutationDataAccessor data)
      : log_error_probabilities(), bpriora(0.0), bpriorb(0.0), data(data) {
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
  }

  /**
   * \brief Compute the likelihood score of the given state.
   *
   * This is done by finding the most likely attachment point for every cell and
   * then multiplying the likelihoods of every cell-gene-combination.
   *
   * \param state The state to score.
   * \return The likelihood that the state represents the true mutation history
   * of the sequenced cells.
   */
  double logscore_state(ChainStateImpl const &state) {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(state.mutation_tree.get_n_nodes() == data.get_range()[1] + 1);
#endif

    log_error_probabilities[0][1] = std::log(state.beta);
    log_error_probabilities[1][1] = std::log(1.0 - state.beta);

    AncestorMatrixImpl ancestor_matrix(state.mutation_tree);
    OccurrenceMatrix occurrences(0);

    for (uint64_t cell_i = 0; cell_i < data.get_range()[0]; cell_i++) {
      auto best_attachment = get_best_attachment(cell_i, ancestor_matrix);
      occurrences += best_attachment.occurrences;
    }

    double tree_score = get_logscore_of_occurrences(occurrences);
    double beta_score = logscore_beta(state.beta);
    return tree_score + beta_score;
  }

  /**
   * \brief Compute the likelihood of the beta error rate being correct.
   *
   * This is based on the a-priori assumption of the beta error rate and it's
   * assumed standard derivation.
   *
   * \param beta The beta error rate to score.
   */
  double logscore_beta(double beta) {
    return std::log(std::tgamma(bpriora + bpriorb)) +
           (bpriora - 1) * std::log(beta) + (bpriorb - 1) * std::log(1 - beta) -
           std::log(std::tgamma(bpriora)) - std::log(std::tgamma(bpriorb));
  }

  /**
   * \brief Information about a found mutation tree attachment for a cell.
   *
   * It is exclusively used as a return value of \ref get_best_attachment.
   */
  struct Attachment {
    /**
     * \brief The index of the best node to attach the cell to.
     */
    uint64_t node_i;
    /**
     * \brief The occurrences of correct and incorrect data values, assuming the
     * cell is attached to the node with index \ref node_i.
     */
    OccurrenceMatrix occurrences;
    /**
     * \brief The log-likelihood that the found attachment is correct.
     */
    double logscore;
  };

  /**
   * \brief Find the most likely attachment point for the given cell.
   *
   * \param cell_i The index of the cell to attach.
   * \param mutation_tree The ancestor matrix of the mutation tree to attach the
   * cell to.
   * \return A struct with information about the found attachment.
   */
  Attachment get_best_attachment(uint64_t cell_i,
                                 AncestorMatrixImpl mutation_tree) {
    uint64_t best_attachment = mutation_tree.get_root();
    OccurrenceMatrix best_attachment_occurrences(0);
    double best_attachment_logscore = -std::numeric_limits<double>::infinity();

    for (uint64_t attachment_node_i = 0;
         attachment_node_i < mutation_tree.get_n_nodes(); attachment_node_i++) {
      OccurrenceMatrix occurrences(0);

      for (uint64_t gene_i = 0; gene_i < data.get_range()[1]; gene_i++) {
        uint64_t posterior = data[cell_i][gene_i];
        uint64_t prior =
            mutation_tree.is_ancestor(gene_i, attachment_node_i) ? 1 : 0;
        occurrences[{posterior, prior}]++;
      }

      double attachment_logscore = get_logscore_of_occurrences(occurrences);
      if (attachment_logscore > best_attachment_logscore) {
        best_attachment = attachment_node_i;
        best_attachment_occurrences = occurrences;
        best_attachment_logscore = attachment_logscore;
      }
    }

    return {best_attachment, best_attachment_occurrences,
            best_attachment_logscore};
  }

  /**
   * \brief Compute the logarithm of the likelihood score for the given
   * occurrences.
   *
   * \param occurrences The occurrences to compute the likelihood of.
   * \return The log-likelihood for the given occurrences.
   */
  double get_logscore_of_occurrences(OccurrenceMatrix occurrences) {
    double logscore = 0.0;
    for (uint64_t i_posterior = 0; i_posterior < 3; i_posterior++) {
      for (uint64_t i_prior = 0; i_prior < 2; i_prior++) {
        logscore += occurrences[{i_posterior, i_prior}] *
                    log_error_probabilities[i_posterior][i_prior];
      }
    }
    return logscore;
  }

private:
  double log_error_probabilities[3][2];
  double bpriora, bpriorb;
  MutationDataAccessor data;
};
} // namespace ffSCITE