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
#include <array>
#include <cassert>
#include <cstdint>
#include <oneapi/dpl/random>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <unordered_map>
#include <vector>

namespace ffSCITE {
template <uint32_t max_n_nodes> class MutationTree {
public:
  using AncestryVector = ac_int<max_n_nodes, false>;

  MutationTree() : parent(), ancestor(), n_nodes(max_n_nodes) {}

  MutationTree(uint32_t n_nodes) : parent(), ancestor(), n_nodes(n_nodes) {
    for (uint32_t i = 0; i < max_n_nodes; i++) {
      parent[i] = get_root();
      for (uint32_t j = 0; j < max_n_nodes; j++) {
        ancestor[j][i] = (i == j || j == get_root()) ? true : false;
      }
    }
  }

  MutationTree(std::vector<uint32_t> parent_vector)
      : parent(), ancestor(), n_nodes(parent_vector.size()) {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(n_nodes <= max_n_nodes);
#endif
    for (uint32_t i = 0; i < max_n_nodes; i++) {

      // First, we assume that node i has no ancestors.
      for (uint32_t j = 0; j < max_n_nodes; j++) {
        ancestor[j][i] = false;
      }

      if (i < n_nodes) {
        // Copy the parent vector value
        parent[i] = parent_vector[i];

        // Then we start from the node i and walk up to the root, marking all
        // nodes on the way as ancestors.
        uint32_t anc = i;
        while (anc != get_root()) {
          ancestor[anc][i] = true;
          anc = parent_vector[anc];
          // Otherwise, there is a circle in the graph!
          assert(anc != i);
        }
      } else {
        parent[i] = get_root();
      }

      // Lastly, also mark the root as our ancestor.
      ancestor[get_root()][i] = true;
    }
  }

  MutationTree(MutationTree const &other) = default;
  MutationTree &operator=(MutationTree const &other) = default;

  static MutationTree<max_n_nodes>
  from_pruefer_code(std::vector<uint32_t> const &pruefer_code) {
    // Algorithm adapted from
    // https://en.wikipedia.org/wiki/Pr%C3%BCfer_sequence, 09th of May 2022,
    // 16:07, since the original reference implementation is sketchy.
#if __SYCL_DEVICE_ONLY__ == 0
    assert(pruefer_code.size() <= max_n_nodes - 2);
#endif
    uint32_t n_nodes = pruefer_code.size() + 2;

    // Compute the (resulting) degrees of every node.
    std::vector<uint32_t> degree;
    for (uint32_t i = 0; i < n_nodes; i++) {
      degree.push_back(1);
    }
    for (uint32_t i = 0; i < pruefer_code.size(); i++) {
      degree[pruefer_code[i]]++;
    }

    // Initialize the parent vector
    std::vector<uint32_t> parent_vector;
    parent_vector.reserve(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
      parent_vector.push_back(n_nodes - 1);
    }

    // Build the tree.
    for (uint32_t i = 0; i < pruefer_code.size(); i++) {
      for (uint32_t j = 0; j < n_nodes; j++) {
        if (degree[j] == 1) {
          parent_vector[j] = pruefer_code[i];
          degree[pruefer_code[i]]--;
          degree[j]--;
          break;
        }
      }
    }

    // Construct the last edge. v is the root of tree as it's new parent has
    // never been assigned.
    uint32_t u = 0;
    for (uint32_t i = 0; i < n_nodes; i++) {
      if (degree[i] == 1) {
        parent_vector[i] = n_nodes - 1; // = root
        break;
      }
    }

    return MutationTree<max_n_nodes>(parent_vector);
  }

  /**
   * @brief Generate a random, uniformly distributed tree.
   *
   * This is done by generating a random Prüfer Code and using
   * `from_pruefer_code` to construct the tree.
   *
   * @tparam The type of URNG to use.
   * @param rng The URNG instance to use.
   * @param n_nodes The number of nodes in the resulting tree, must be lower
   * than or equal to `max_n_nodes`.
   * @return A random, uniformly distributed tree.
   */
  template <typename RNG>
  static MutationTree<max_n_nodes> sample_random_tree(RNG &rng,
                                                      uint32_t n_nodes) {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(n_nodes <= max_n_nodes);
#endif

    oneapi::dpl::uniform_int_distribution<unsigned long> int_distribution(
        0, n_nodes - 1);

    // Generate a pruefer code for the tree.
    std::vector<uint32_t> pruefer_code;
    for (uint32_t i = 0; i < n_nodes - 2; i++) {
      pruefer_code.push_back(int_distribution(rng));
    }

    return from_pruefer_code(pruefer_code);
  }

  uint32_t get_root() const { return n_nodes - 1; }

  uint32_t get_parent(uint32_t node_i) const {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(node_i < n_nodes);
#endif
    return parent[node_i];
  }

  uint32_t get_n_nodes() const { return n_nodes; }

  /**
   * @brief Query whether node a is an ancestor of node b.
   *
   * @param node_a_i The index of the potential ancestor.
   * @param node_b_i The index of the potential descendant.
   * @return true iff node a is an ancestor of node b.
   */
  bool is_ancestor(uint32_t node_a_i, uint32_t node_b_i) const {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(node_a_i < n_nodes && node_b_i < n_nodes);
#endif
    return ancestor[node_a_i][node_b_i];
  }

  /**
   * @brief Query whether node a is a descendant of node b.
   *
   * @param node_a_i The index of the potential descendant.
   * @param node_b_i The index of the potential ancestor.
   * @return true iff node a is an ancestor of node b.
   */
  bool is_descendant(uint32_t node_a_i, uint32_t node_b_i) const {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(node_a_i < n_nodes && node_b_i < n_nodes);
#endif
    return ancestor[node_b_i][node_a_i];
  }

  /**
   * @brief Return a boolean array describing a node's descendants.
   *
   * For example, if this method was invoked for node a, one can query whether
   * node b is a descendant of node a by checking whether the bit with index b
   * in the array is true. This form of an array of boolean values can be used
   * to iterate over the descendants of a node.
   *
   * @param node_i The index of the node who's descendants are queried.
   * @return The descendants bit array.
   */
  std::array<bool, max_n_nodes> get_descendants(uint32_t node_i) const {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(node_i < n_nodes);
#endif
    std::array<bool, max_n_nodes> descendants;
#pragma unroll
    for (uint32_t i = 0; i < max_n_nodes; i++) {
      if (i < n_nodes) {
        descendants[i] = is_ancestor(node_i, i);
      }
    }
    return descendants;
  }

  /**
   * @brief Get the total number of a node's descendants.
   *
   * @param node_i The index of the node who's number of descendants is queried.
   * @return The number of descendants.
   */
  uint32_t get_n_descendants(uint32_t node_i) const {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(node_i < n_nodes);
#endif
    uint32_t n_descendants = 0;
#pragma unroll
    for (uint32_t i = 0; i < max_n_nodes; i++) {
      if (i < n_nodes && is_ancestor(node_i, i)) {
        n_descendants++;
      }
    }
    return n_descendants;
  }

  /**
   * @brief Return a boolean array describing a node's ancestors.
   *
   * For example, if this method was invoked for node a, one can query whether
   * node b is an ancestor of node a by checking whether the bit with index b
   * in the array is true. This form of an array of boolean values can be used
   * to iterate over the ancestors of a node.
   *
   * @param node_i The index of the node who's ancestors are queried.
   * @return The ancestors bit array.
   */
  std::array<bool, max_n_nodes> get_ancestors(uint32_t node_i) const {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(node_i < n_nodes);
#endif
    std::array<bool, max_n_nodes> ancestors;
#pragma unroll
    for (uint32_t i = 0; i < max_n_nodes; i++) {
      if (i < n_nodes) {
        ancestors[i] = is_ancestor(i, node_i);
      }
    }
    return ancestors;
  }

  /**
   * @brief Get the total number of a node's ancestors.
   *
   * @param node_i The index of the node who's number of ancestors is queried.
   * @return The number of ancestors.
   */
  uint32_t get_n_ancestors(uint32_t node_i) const {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(node_i < n_nodes);
#endif
    uint32_t n_ancestors = 0;
#pragma unroll
    for (uint32_t i = 0; i < max_n_nodes; i++) {
      if (i < n_nodes && is_ancestor(i, node_i)) {
        n_ancestors++;
      }
    }
    return n_ancestors;
  }

  /**
   * @brief Compare two parent vectors for equality.
   *
   * Two trees are equal iff their number of nodes is equal and every node has
   * the same parent.
   *
   * @param other The other tree to compare too.
   * @return true The two trees are equal.
   * @return false The two trees are not equal.
   */
  bool operator==(MutationTree<max_n_nodes> const &other) const {
    if (n_nodes != other.n_nodes) {
      return false;
    }
#pragma unroll
    for (uint32_t node_i = 0; node_i < max_n_nodes; node_i++) {
      if (node_i < n_nodes && parent[node_i] != other.parent[node_i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Compare two parent vectors for inequality.
   *
   * Two trees are equal iff their number of nodes is equal and every nodes has
   * the same parent.
   *
   * @param other The other tree to compare too.
   * @return true The two trees are not equal.
   * @return false The two trees are equal.
   */
  bool operator!=(MutationTree<max_n_nodes> const &other) const {
    return !operator==(other);
  }

  std::string to_graphviz() const {
    std::stringstream stream;
    stream << "digraph G {" << std::endl;
    stream << "node [color=deeppink4, style=filled, fontcolor=white];"
           << std::endl;
    for (uint32_t node_i = 0; node_i < n_nodes - 1; node_i++) {
      stream << parent[node_i] << " -> " << node_i << ";" << std::endl;
    }
    stream << "}" << std::endl;
    return stream.str();
  }

  std::string to_newick() const {
    std::vector<std::vector<uint32_t>> children;

    // Initialize children list
    children.reserve(n_nodes);
    for (uint32_t node_i = 0; node_i < n_nodes; node_i++) {
      children.push_back(std::vector<uint32_t>());
    }

    // Populate children list (not including the root to avoid infinite
    // recursion).
    for (uint32_t node_i = 0; node_i < n_nodes - 1; node_i++) {
      children[parent[node_i]].push_back(node_i);
    }

    std::stringstream stream;
    add_node_to_newick_code(children, stream, get_root());
    stream << std::endl;
    return stream.str();
  }

  /**
   * @brief Swap the positions of two nodes within the tree.
   *
   * After this operation, the tree will have the same topology as before, but
   * node b will be where node a used to be, and vice versa. None of these nodes
   * may be the root of the tree since the root is not movable.
   *
   * @param node_a_i One of the nodes to swap.
   * @param node_b_i The other node to swap.
   */
  void swap_nodes(MutationTree<max_n_nodes> &out_tree, uint32_t node_a_i,
                  uint32_t node_b_i) const {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(node_a_i != get_root() && node_b_i != get_root());
#endif
    out_tree.n_nodes = n_nodes;

    for (uint32_t i = 0; i < max_n_nodes; i++) {
      uint32_t new_parent;

      if (i == node_a_i) {
        if (node_a_i == parent[node_b_i]) {
          new_parent = node_b_i;
        } else {
          new_parent = parent[node_b_i];
        }
      } else if (i == node_b_i) {
        if (node_b_i == parent[node_a_i]) {
          new_parent = node_a_i;
        } else {
          new_parent = parent[node_a_i];
        }
      } else {
        if (parent[i] == node_a_i) {
          new_parent = node_b_i;
        } else if (parent[i] == node_b_i) {
          new_parent = node_a_i;
        } else {
          new_parent = parent[i];
        }
      }

      AncestryVector ancestry_vector;
      if (i == node_a_i) {
        ancestry_vector = ancestor[node_b_i];
      } else if (i == node_b_i) {
        ancestry_vector = ancestor[node_a_i];
      } else {
        ancestry_vector = ancestor[i];
      }

      bool swap = ancestry_vector[node_a_i];
      ancestry_vector[node_a_i] = ancestry_vector[node_b_i];
      ancestry_vector[node_b_i] = swap;

      out_tree.parent[i] = new_parent;
      out_tree.ancestor[i] = ancestry_vector;
    }
  }

  /**
   * @brief Move the subtree below a node to a new parent.
   *
   * After that, the moved node's parent will be the requested new parent, but
   * apart from that, the tree will remain the same. If the method is not
   * compiled for a SYCL device, it will assert that the new parent is not a
   * descendant of the node to move, since this would detach one part of the
   * tree and introduce a cycle. The node to move must also not be the root of
   * the tree, as the root is not movable.
   *
   * @param node_i The index of the node to move.
   * @param new_parent_i The index of the node's new parent node.
   */
  void move_subtree(MutationTree<max_n_nodes> &out_tree, uint32_t node_i,
                    uint32_t new_parent_i) const {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(!is_descendant(new_parent_i, node_i) && node_i != get_root());
#endif
    out_tree.n_nodes = n_nodes;

    std::array<uint32_t, max_n_nodes> pv = parent;
    pv[node_i] = new_parent_i;
    out_tree.parent = pv;

    AncestryVector moved_node_descendant = ancestor[node_i];

    for (uint32_t i = 0; i < max_n_nodes; i++) {
      AncestryVector old_vector = ancestor[i];
      AncestryVector new_vector = old_vector;

#pragma unroll
      for (uint32_t j = 0; j < max_n_nodes; j++) {
        if (moved_node_descendant[j]) {
          // if (node_i -> j),
          // we have (i -> j) <=> (j -> new_parent) || (a -> i -> j)
          new_vector[j] = old_vector[new_parent_i] ||
                          (moved_node_descendant[i] && old_vector[j]);
        } else {
          // otherwise, we have (node_i !-> node_j).
          // Since this node is unaffected, everything remains the same.
          new_vector[j] = old_vector[j];
        }
      }

      out_tree.ancestor[i] = new_vector;
    }
  }

  /**
   * @brief Swap two subtrees in the tree.
   *
   * This operation takes the subtree below node a, hangs it next to node b, and
   * then hangs the subtree below node b to where node a used to be. If the
   * method is not compiled for a SYCL device, it will assert that the node a
   * and b are not descendants of each other, since this would detach one part
   * of the tree and introduce a cycle. None of these nodes may be the root of
   * the tree, since the root is not movable.
   *
   * @param node_a_i The index of one of the nodes to swap.
   * @param node_b_i The index of the other node to swap.
   */
  void swap_subtrees(MutationTree<max_n_nodes> &out_tree, uint32_t node_a_i,
                     uint32_t node_b_i) {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(node_a_i != get_root() && node_b_i != get_root());
    assert(!is_descendant(node_a_i, node_b_i));
    assert(!is_descendant(node_b_i, node_a_i));
#endif
    out_tree.n_nodes = n_nodes;

    std::array<uint32_t, max_n_nodes> pv = parent;
    std::swap(pv[node_a_i], pv[node_b_i]);
    out_tree.parent = pv;

    AncestryVector node_a_descendant = ancestor[node_a_i];
    AncestryVector node_b_descendant = ancestor[node_b_i];

    for (uint32_t i = 0; i < max_n_nodes; i++) {
      AncestryVector old_vector = ancestor[i];
      AncestryVector new_vector = old_vector;

#pragma unroll
      for (uint32_t j = 0; j < max_n_nodes; j++) {
        if (node_a_descendant[j] && !node_b_descendant[j]) {
          // if (a -> j && b !-> j),
          // we have (i -> j) <=> (j -> parent[b]) || (a -> i -> j)
          new_vector[j] = old_vector[parent[node_b_i]] ||
                          (node_a_descendant[i] && old_vector[j]);
        } else if (!node_a_descendant[j] && node_b_descendant[j]) {
          // if (a !-> j && b -> j),
          // we have (i -> j) <=> (j -> parent[a]) || (b -> i -> j)
          new_vector[j] = old_vector[parent[node_a_i]] ||
                          (node_b_descendant[i] && old_vector[j]);
        } else {
          // we have (a !-> j && b !-> j), (a -> j && b -> j) is impossible.
          // In this case, everything remains the same.
          new_vector[j] = old_vector[j];
        }
      }

      out_tree.ancestor[i] = new_vector;
    }
  }

private:
  void
  add_node_to_newick_code(std::vector<std::vector<uint32_t>> const &children,
                          std::stringstream &stream, uint32_t node_i) const {
    if (children[node_i].size() != 0) {
      stream << "(";
      for (uint32_t i = 0; i < children[node_i].size(); i++) {
        add_node_to_newick_code(children, stream, children[node_i][i]);
        if (i != children[node_i].size() - 1) {
          stream << ",";
        }
      }
      stream << ")";
    }
    stream << node_i;
  }

  std::array<uint32_t, max_n_nodes> parent;
  std::array<AncestryVector, max_n_nodes> ancestor;
  uint32_t n_nodes;
};
} // namespace ffSCITE