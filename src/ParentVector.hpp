#include <array>
#include <bit>
#include <cstdint>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <unordered_map>
#include <vector>

namespace ffSCITE {
template <uint64_t max_n_nodes> class ParentVector {
public:
  static constexpr uint64_t n_node_bits = std::bit_width(max_n_nodes);
  using uindex_node_t = ac_int<n_node_bits, false>;

  ParentVector(uindex_node_t n_nodes) : parent(), n_nodes(n_nodes) {
    assert(n_nodes <= max_n_nodes);

    for (uindex_node_t i = 0; i < max_n_nodes; i++) {
      parent[i] = max_n_nodes;
    }
  }

  void from_pruefer_code(std::vector<uindex_node_t> const &pruefer_code) {
    // Algorithm adapted from
    // https://en.wikipedia.org/wiki/Pr%C3%BCfer_sequence, 09th of May 2022,
    // 16:07 Original reference implementation is sketchy. Here, we work with an
    // internal parent vector that also includes the root since a pruefer code
    // may describe a tree where the last node is not the root of the tree. We
    // repair this afterwards.
    assert(pruefer_code.size() == n_nodes - 1);

    // Compute the (resulting) degrees of every node.
    std::vector<uindex_node_t> degree;
    for (uindex_node_t i = 0; i <= n_nodes; i++) {
      degree.push_back(1);
    }
    for (uindex_node_t i = 0; i < pruefer_code.size(); i++) {
      degree[pruefer_code[i]]++;
    }

    // Build the tree.
    for (uindex_node_t i = 0; i < pruefer_code.size(); i++) {
      for (uindex_node_t j = 0; j < n_nodes; j++) {
        if (degree[j] == 1) {
          parent[j] = pruefer_code[i];
          degree[pruefer_code[i]]--;
          degree[j]--;
          break;
        }
      }
    }

    // Construct the last edge. v is the root of tree as it's new parent has
    // never been assigned.
    uindex_node_t u = 0, v = 0;
    bool u_found = false;
    for (uindex_node_t i = 0; i <= n_nodes; i++) {
      if (degree[i] == 1) {
        if (!u_found) {
          u = i;
          u_found = true;
        } else {
          v = i;
          parent[u] = v;
          break;
        }
      }
    }
  }

  template <typename RNG> void randomize(RNG &rng) {
    // Generate a pruefer code for the tree.
    std::vector<uindex_node_t> pruefer_code;
    for (uindex_node_t i = 0; i < n_nodes - 1; i++) {
      pruefer_code.push_back(rng() % (n_nodes + 1));
    }

    from_pruefer_code(pruefer_code);
  }

  uindex_node_t operator[](uindex_node_t node_i) const {
    return parent[node_i];
  }

  uindex_node_t get_n_nodes() const { return n_nodes; }

  bool is_root(uindex_node_t node_i) const { return node_i >= n_nodes; }

  bool is_descendant(uindex_node_t node_a_i, uindex_node_t node_b_i) const {
    if (node_a_i == node_b_i) {
      return true;
    }

    while (!(node_a_i == node_b_i || node_a_i >= n_nodes)) {
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