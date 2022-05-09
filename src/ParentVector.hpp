#include <array>
#include <bit>
#include <cstdint>
#include <sycl/ext/intel/ac_types/ac_int.hpp>

namespace ffSCITE {
template <uint64_t max_n_nodes> class ParentVector {
public:
  static constexpr uint64_t n_node_bits = std::bit_width(max_n_nodes);
  using uindex_node_t = ac_int<n_node_bits, false>;

  static constexpr uindex_node_t root = max_n_nodes;

  ParentVector(uindex_node_t n_nodes) : parent(), n_nodes(n_nodes) {
    for (uindex_node_t i = 0; i < max_n_nodes; i++) {
      parent[i] = max_n_nodes;
    }
  }

  uindex_node_t operator[](uindex_node_t node_i) const {
    return parent[node_i];
  }

  uindex_node_t get_n_nodes() const { return parent.get_n_elements(); }

  bool is_descendant(uindex_node_t node_a_i, uindex_node_t node_b_i) const {
    if (node_a_i == node_b_i) {
      return true;
    }

    while (!(node_a_i == node_b_i || node_a_i == root)) {
      node_a_i = parent[node_a_i];
    }

    return node_a_i == node_b_i;
  }

  void swap_nodes(uindex_node_t node_a_i, uindex_node_t node_b_i) {
    if (node_a_i == node_b_i) {
      return;
    }

    for (uindex_node_t i = 0; i < max_n_nodes; i++) {
      if (i != node_a_i && i != node_b_i) {
        if (parent[i] == node_a_i) {
          parent[i] = node_b_i;
        } else if (parent[i] == node_b_i) {
          parent[i] = node_a_i;
        }
      }
    }

    if (node_a_i == parent[node_b_i]) {
      parent[node_b_i] = parent[node_a_i];
      parent[node_a_i] = node_b_i;
    } else if (node_b_i == parent[node_a_i]) {
      parent[node_a_i] = parent[node_b_i];
      parent[node_b_i] = node_a_i;
    } else {
      std::swap(parent[node_a_i], parent[node_b_i]);
    }
  }

  void move_subtree(uindex_node_t node_i, uindex_node_t new_parent_i) {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(!is_descendant(new_parent_i, node_i));
#endif

    parent[node_i] = new_parent_i;
  }

private:
  std::array<uindex_node_t, max_n_nodes> parent;
  uindex_node_t n_nodes;
};
} // namespace ffSCITE