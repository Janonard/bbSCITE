#pragma once
#include "ParentVector.hpp"

namespace ffSCITE {
/**
 * \brief Compact structure containing the state of a SCITE markov chain.
 *
 * This is nothing more than a parent vector for the mutation tree, the beta
 * error rate, and some default operations.
 *
 * \tparam The maximal number of free, movable nodes in the mutation tree, i.e.
 * genome positions.
 */
template <uint64_t max_n_nodes> struct ChainState {
public:
  /**
   * \brief Shorthand for the parent vector type.
   */
  using ParentVectorImpl = ParentVector<max_n_nodes>;
  /**
   * \brief Shorthand for the parent vector's node index type.
   */
  using uindex_node_t = typename ParentVectorImpl::uindex_node_t;

  ChainState() : mutation_tree(), beta(0.0) {}
  ChainState(ChainState<max_n_nodes> const &other) = default;
  ChainState<max_n_nodes> &
  operator=(ChainState<max_n_nodes> const &other) = default;

  /**
   * \brief Initialize a new, random chain state.
   *
   * The mutation tree will be sampled from a uniform distribution and the beta
   * error rate will be set to a prior value.
   *
   * \tparam RNG The URNG to use for mutation tree sampling.
   * \param rng The URNG instance to use.
   * \param n_nodes The number of free, movable nodes in the tree, i.e. genome
   * positions. \param beta_prior A prior estimate of the beta error rate.
   */
  template <typename RNG>
  ChainState(RNG &rng, uindex_node_t n_nodes, double beta_prior)
      : mutation_tree(ParentVectorImpl::sample_random_tree(rng, n_nodes)),
        beta(beta_prior) {}

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