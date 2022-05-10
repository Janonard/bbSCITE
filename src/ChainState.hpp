#include "ParentVector.hpp"

namespace ffSCITE {
template <uint64_t max_n_nodes> struct ChainState {
public:
  using ParentVectorImpl = ParentVector<max_n_nodes>;
  using uindex_node_t = typename ParentVectorImpl::uindex_node_t;

  ChainState() : mutation_tree(), beta(0.0) {}
  ChainState(ChainState<max_n_nodes> const &other) = default;
  ChainState<max_n_nodes> &
  operator=(ChainState<max_n_nodes> const &other) = default;

  template <typename RNG>
  ChainState(RNG &rng, uindex_node_t n_nodes, double beta_prior)
      : mutation_tree(n_nodes), beta(beta_prior) {
    mutation_tree.randomize(rng);
  }

  ParentVector<max_n_nodes> mutation_tree;
  double beta;
};
} // namespace ffSCITE