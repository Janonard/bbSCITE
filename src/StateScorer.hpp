#pragma once
#include "AncestorMatrix.hpp"
#include "ChainState.hpp"
#include "StaticMatrix.hpp"

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
template <uint64_t max_n_cells, uint64_t max_n_genes> class StateScorer {
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
   * \brief Shorthand for the node index type in use.
   */
  using uindex_node_t = typename ChainStateImpl::uindex_node_t;

  /**
   * \brief Width of the indices to address cells in the mutation data matrix.
   */
  static constexpr uint64_t n_cells_bits = std::bit_width(max_n_cells);
  /**
   * \brief Index type to address cells in the mutation data matrix.
   */
  using uindex_cell_t = ac_int<n_cells_bits, false>;

  /**
   * \brief Width of the indices to address combined cells and genes.
   */
  static constexpr uint64_t n_data_entry_bits =
      std::bit_width(max_n_cells * max_n_genes);

  /**
   * \brief Index type to address combined cells and genes.
   */
  using uindex_data_entry_t = ac_int<n_data_entry_bits, false>;

  /**
   * \brief Data type of the entries in the mutation data matrix.
   *
   * 0b00 stands for "no mutation found", 0b01 stands for "mutation found", 0b10
   * stands for "no data", and 0b11 is currently unused.
   */
  using DataEntry = ac_int<2, false>;

  /**
   * \brief Shorthand for the mutation data matrix type.
   */
  using MutationDataMatrix = StaticMatrix<DataEntry, max_n_cells, max_n_genes>;

  /**
   * \brief Shorthand for the occurrence matrix type.
   */
  using OccurrenceMatrix = StaticMatrix<uindex_data_entry_t, 3, 2>;

  /**
   * \brief Initialize a new state scorer
   *
   * This new state scorer uses the initial assumptions over the alpha and beta
   * error rates and also load the mutation data.
   *
   * \param prior_alpha The initial assumption of the alpha error rate (false
   * positive)
   * \param prior_beta The initial assumption of the beta error rate
   * (false negative)
   * \param n_cells The number of cells covered by the mutation
   * data matrix.
   * \param n_genes The number of genes covered by the mutation
   * data matrix.
   */
  StateScorer(double prior_alpha, double prior_beta, uindex_cell_t n_cells,
              uindex_node_t n_genes, MutationDataMatrix data)
      : log_error_probabilities(), n_cells(n_cells), n_genes(n_genes),
        data(data) {
    // mutation not observed, not present
    log_error_probabilities[0][0] = std::log(1.0 - prior_alpha);
    // mutation observed, not present
    log_error_probabilities[1][0] = std::log(prior_alpha);
    // missing data, mutation not present
    log_error_probabilities[2][0] = std::log(1.0);
    // mutation not observed, but present
    log_error_probabilities[0][1] = std::log(prior_beta);
    // mutation observed and present
    log_error_probabilities[1][1] = std::log(1.0 - prior_beta);
    // missing data, mutation present
    log_error_probabilities[2][1] = std::log(1.0);
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
  double score_state(ChainStateImpl const &state) {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(state.mutation_tree.get_n_nodes() == n_genes + 1);
#endif

    log_error_probabilities[0][1] = std::log(state.beta);
    log_error_probabilities[1][1] = std::log(1.0 - state.beta);

    AncestorMatrixImpl ancestor_matrix(state.mutation_tree);
    OccurrenceMatrix occurrences(0);

    for (uindex_cell_t cell_i = 0; cell_i < n_cells; cell_i++) {
      auto best_attachment = get_best_attachment(cell_i, ancestor_matrix);
      occurrences += std::get<1>(best_attachment);
    }

    return std::exp(get_logscore_of_occurrences(occurrences));
  }

  /**
   * \brief Find the most likely attachment point for the given cell.
   *
   * \param cell_i The index of the cell to attach.
   * \param mutation_tree The ancestor matrix of the mutation tree to attach the
   * cell to.
   * \return A tuple of the most likely node to attach the node to, the
   * occurrences of the different likelihood types, and the log-likelihood that
   * the return attachment point is correct.
   */
  std::tuple<uindex_node_t, OccurrenceMatrix, double>
  get_best_attachment(uindex_cell_t cell_i, AncestorMatrixImpl mutation_tree) {
    uindex_node_t best_attachment = mutation_tree.get_root();
    OccurrenceMatrix best_attachment_occurrences(0);
    double best_attachment_logscore = -std::numeric_limits<double>::infinity();

    for (uindex_node_t attachment_node_i = 0;
         attachment_node_i < mutation_tree.get_n_nodes(); attachment_node_i++) {
      OccurrenceMatrix occurrences(0);

      for (uindex_node_t gene_i = 0; gene_i < n_genes; gene_i++) {
        uint64_t posterior = data[{cell_i, gene_i}];
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
  uindex_node_t n_cells, n_genes;
  StaticMatrix<ac_int<2, false>, max_n_cells, max_n_genes> data;
};
} // namespace ffSCITE