#include "StackVector.hpp"

namespace ffSCITE {
template <uint64_t max_n_nodes> class ParentVector {
public:
  static constexpr uint64_t n_node_bits = std::bit_width(max_n_nodes);
  using uindex_node_t = ac_int<n_node_bits, false>;
  using StackVectorImpl = StackVector<uindex_node_t, max_n_nodes>;

  ParentVector(uindex_node_t n_nodes) : parent() {
    for (uindex_node_t i = 0; i < n_nodes; i++) {
      parent.push_back(n_nodes);
    }
  }

  uindex_node_t const &operator[](uindex_node_t node_i) const {
    return parent[node_i];
  }

  uindex_node_t &operator[](uindex_node_t node_i) { return parent[node_i]; }

  uindex_node_t get_n_nodes() const { return parent.get_n_elements(); }

  bool is_root(uindex_node_t node_i) const {
    return parent[node_i] >= get_n_nodes();
  }

  bool is_descendant(uindex_node_t node_a_i, uindex_node_t node_b_i) const {
    if (node_a_i == node_b_i) {
      return true;
    }

    while (!(node_a_i == node_b_i || is_root(node_a_i))) {
      node_a_i = parent[node_a_i];
    }

    return node_a_i == node_b_i;
  }

  void swap_nodes(uindex_node_t node_a_i, uindex_node_t node_b_i) {
    StackVectorImpl new_a_children, new_b_children;

    for (uindex_node_t i = 0; i < max_n_nodes; i++) {
      if (i < get_n_nodes()) {
        if (parent[i] == node_a_i) {
          new_b_children.push_back(i);
        }
        if (parent[i] == node_b_i) {
          new_a_children.push_back(i);
        }
      }
    }

    for (uindex_node_t i = 0; i < max_n_nodes; i++) {
      if (i < new_a_children.get_n_elements()) {
        parent[new_a_children[i]] = node_a_i;
      }
      if (i < new_b_children.get_n_elements()) {
        parent[new_b_children[i]] = node_b_i;
      }
    }

    std::swap(parent[node_a_i], parent[node_b_i]);
  }

private:
  StackVectorImpl parent;
};
} // namespace ffSCITE