#pragma once
#include "ParentVector.hpp"

namespace ffSCITE {
template <uint64_t max_n_nodes> class AncestorMatrix {
public:
  using ParentVectorImpl = ParentVector<max_n_nodes>;
  using uindex_node_t = typename ParentVectorImpl::uindex_node_t;

  AncestorMatrix() : ancestor(), n_nodes(max_n_nodes) {
    for (uindex_node_t i = 0; i < max_n_nodes + 1; i++) {
      for (uindex_node_t j = 0; j < max_n_nodes + 1; j++) {
        ancestor[i][j] = i == j;
      }
    }
  }
  AncestorMatrix(AncestorMatrix<max_n_nodes> const &other) = default;
  AncestorMatrix<max_n_nodes> &
  operator=(AncestorMatrix<max_n_nodes> const &other) = default;

  AncestorMatrix(ParentVectorImpl const &parent_vector)
      : ancestor(), n_nodes(parent_vector.get_n_nodes()) {
    for (uindex_node_t i = 0; i < max_n_nodes; i++) {
      // First, we assume that node i has no ancestors.
      for (uindex_node_t j = 0; j < max_n_nodes; j++) {
        ancestor[j][i] = false;
      }

      // Then we start from the node i and walk up to the root, marking all
      // nodes on the way as ancestors.
      uindex_node_t anc = i;
      while (anc < n_nodes) {
        ancestor[anc][i] = true;
        anc = parent_vector[anc];
      }
    }
  }

  bool is_ancestor(uindex_node_t node_a_i, uindex_node_t node_b_i) const {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(node_a_i < n_nodes + 1 && node_b_i < n_nodes + 1);
#endif
    if (node_a_i >= n_nodes) {
      return true;
    } else if (node_b_i >= n_nodes) {
      return false;
    } else {
      return ancestor[node_a_i][node_b_i];
    }
  }

  std::array<bool, max_n_nodes> get_descendants(uindex_node_t node_i) const {
    std::array<bool, max_n_nodes> descendants;
    for (uindex_node_t i = 0; i < n_nodes; i++) {
      descendants[i] = (node_i >= n_nodes) || ancestor[node_i][i];
    }
    return descendants;
  }

  uindex_node_t get_n_descendants(uindex_node_t node_i) const {
    if (node_i >= n_nodes) {
      return n_nodes + 1;
    } else {
      uindex_node_t n_descendants = 0;
      for (uindex_node_t i = 0; i < n_nodes; i++) {
        if (ancestor[node_i][i]) {
          n_descendants++;
        }
      }
      return n_descendants;
    }
  }

  uindex_node_t get_n_nodes() const { return n_nodes; }

private:
  std::array<std::array<bool, max_n_nodes>, max_n_nodes> ancestor;
  uindex_node_t n_nodes;
};
} // namespace ffSCITE