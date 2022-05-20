#pragma once
#include "ParentVector.hpp"

namespace ffSCITE {
/**
 * \brief Compact structure containing the state of a SCITE markov chain.
 *
 * This is nothing more than a parent vector for the mutation tree, the beta
 * error rate, and some default operations.
 *
 * \tparam The maximal number of genes in the dataset.
 */
template <uint64_t max_n_genes> struct ChainState {
public:
  /**
   * \brief The maximal number of nodes in a mutation tree.
   *
   * The mutation tree has a node for every gene that may mutate, but also an
   * additional node that represents the totally unmutated state.
   */
  static constexpr uint64_t max_n_nodes = max_n_genes + 1;

  /**
   * \brief Shorthand for the parent vector type.
   */
  using ParentVectorImpl = ParentVector<max_n_nodes>;
  /**
   * \brief Shorthand for the parent vector's node index type.
   */
  using uindex_node_t = typename ParentVectorImpl::uindex_node_t;

  ChainState() : mutation_tree(), beta(0.0) {}
  ChainState(ChainState<max_n_genes> const &other) = default;
  ChainState<max_n_genes> &
  operator=(ChainState<max_n_genes> const &other) = default;

  /**
   * \brief Initialize a chain state with the given mutation tree and beta error
   * rate.
   *
   * \param mutation_tree The mutation tree to use.
   * \param beta The beta error rate to use.
   */
  ChainState(ParentVector<max_n_nodes> mutation_tree, double beta)
      : mutation_tree(mutation_tree), beta(beta) {}

  /**
   * \brief Initialize a new, random chain state.
   *
   * The mutation tree will be sampled from a uniform distribution and the beta
   * error rate will be set to a prior value.
   *
   * \tparam RNG The URNG to use for mutation tree sampling.
   * \param rng The URNG instance to use.
   * \param n_genes The number of genes in the dataset.
   * \param beta_prior A prior estimate of the beta error rate.
   */
  template <typename RNG>
  static ChainState<max_n_genes> sample_random_state(RNG &rng, uindex_node_t n_genes,
                                              double beta_prior) {
    ParentVectorImpl mutation_tree =
        ParentVectorImpl::sample_random_tree(rng, n_genes + 1);
    return ChainState<max_n_genes>(mutation_tree, beta_prior);
  }

  /**
   * \brief The (proposed) mutation tree.
   */
  ParentVector<max_n_nodes> mutation_tree;

  /**
   * \brief The (proposed) beta error rate.
   */
  double beta;
};
} // namespace ffSCITE