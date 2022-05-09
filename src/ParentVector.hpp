#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <array>
#include <bit>
#include <cstdint>

namespace ffSCITE {
template <uint64_t max_n_nodes> class ParentVector {
public:
  static constexpr uint64_t n_node_bits = std::bit_width(max_n_nodes);
  using uindex_node_t = ac_int<n_node_bits, false>;

  ParentVector(uindex_node_t n_nodes) : parent(), n_nodes(n_nodes) {
    for (uindex_node_t i = 0; i < max_n_nodes; i++) {
      parent[i] = max_n_nodes;
    }
  }

  uindex_node_t const &operator[](uindex_node_t node_i) const {
    return parent[node_i];
  }

  uindex_node_t &operator[](uindex_node_t node_i) { return parent[node_i]; }

  uindex_node_t get_n_nodes() const { return n_nodes; }

  bool is_root(uindex_node_t node_i) const {
    return parent[node_i] >= n_nodes;
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
    std::array<uindex_node_t, max_n_nodes> new_a_children, new_b_children;
    uindex_node_t n_new_a_children = 0, n_new_b_children = 0;

    for (uindex_node_t i = 0; i < max_n_nodes; i++) {
      if (i < n_nodes) {
        if (parent[i] == node_a_i) {
          new_b_children[n_new_b_children] = i;
          n_new_b_children++;
        }
        if (parent[i] == node_b_i) {
          new_a_children[n_new_a_children] = i;
          n_new_a_children++;
        }
      }
    }

    for (uindex_node_t i = 0; i < max_n_nodes; i++) {
      if (i < n_new_a_children) {
        parent[new_a_children[i]] = node_a_i;
      }
      if (i < n_new_b_children) {
        parent[new_b_children[i]] = node_b_i; 
      }
    }

    std::swap(parent[node_a_i], parent[node_b_i]);
  }

private:
  std::array<uindex_node_t, max_n_nodes> parent;
  uindex_node_t n_nodes;
};
} // namespace ffSCITE