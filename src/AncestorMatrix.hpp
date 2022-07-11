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
#include "ParentVector.hpp"
#include <sycl/ext/intel/ac_types/ac_int.hpp>

namespace ffSCITE {
/**
 * @brief A datastructure to query whether two nodes in a tree are related.
 *
 * This is done by storing a nxn-matrix of bits. According to an ancestor matrix
 * A, the node i is an ancestor of j iff A[i][j] is true (or 1). Ancestor
 * matrices are usually constructed from a @ref ParentVector.
 *
 * @tparam max_n_nodes The maximal number of nodes in tree,
 * excluding the root.
 */
template <uint32_t max_n_nodes> class AncestorMatrix {
public:
  /**
   * @brief Shorthand to the corresponding parent vector type.
   */
  using ParentVectorImpl = ParentVector<max_n_nodes>;

  using AncestryVector = ac_int<max_n_nodes, false>;

  /**
   * @brief Default constructor
   *
   * Instantiate the ancestor matrix of a tree with the maximal number of nodes
   * where all nodes are connected directly to the root.
   */
  AncestorMatrix() : ancestor(), n_nodes(max_n_nodes) {}
  AncestorMatrix(uint32_t n_nodes) : ancestor(), n_nodes(n_nodes) {}
  AncestorMatrix(AncestorMatrix<max_n_nodes> const &other) = default;
  AncestorMatrix<max_n_nodes> &
  operator=(AncestorMatrix<max_n_nodes> const &other) = default;

  /**
   * @brief Construct the ancestor matrix for a parent vector's tree.
   *
   * This done by walking up from every node to the tree's root and marking all
   * nodes on the way. The number of nodes will be transferred from the parent
   * vector.
   *
   * @param parent_vector The parent vector to construct the ancestor matrix
   * from.
   */
  AncestorMatrix(ParentVectorImpl const &parent_vector)
      : ancestor(), n_nodes(parent_vector.get_n_nodes()) {
    for (uint32_t i = 0; i < max_n_nodes; i++) {
      if (i >= n_nodes) {
        continue;
      }

#pragma unroll
      // First, we assume that node i has no ancestors.
      for (uint32_t j = 0; j < max_n_nodes; j++) {
        ancestor[j][i] = false;
      }

      // Then we start from the node i and walk up to the root, marking all
      // nodes on the way as ancestors.
      uint32_t anc = i;
      while (anc != parent_vector.get_root()) {
        ancestor[anc][i] = true;
        anc = parent_vector[anc];
      }
      // Lastly, also mark the root as our ancestor.
      ancestor[parent_vector.get_root()][i] = true;
    }
  }

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

  AncestorMatrix swap_nodes(uint32_t node_a_i, uint32_t node_b_i) const {
    AncestorMatrix new_am(n_nodes);

    for (uint32_t i = 0; i < max_n_nodes; i++) {
      // Also iterating for non-existing nodes. Has no observable effects and is
      // simpler.

      AncestryVector vector;
      if (i == node_a_i) {
        vector = ancestor[node_b_i];
      } else if (i == node_b_i) {
        vector = ancestor[node_a_i];
      } else {
        vector = ancestor[i];
      }

      bool swap = vector[node_a_i];
      vector[node_a_i] = vector[node_b_i];
      vector[node_b_i] = swap;

      new_am.ancestor[i] = vector;
    }

    return new_am;
  }

  AncestorMatrix move_subtree(uint32_t node_i, uint32_t new_parent_i) const {
    AncestorMatrix new_am(n_nodes);

    AncestryVector moved_node_descendant = ancestor[node_i];

    for (uint32_t i = 0; i < max_n_nodes; i++) {
      AncestryVector old_vector = ancestor[i];
      AncestryVector new_vector;

#pragma unroll
      for (uint32_t j = 0; j < max_n_nodes; j++) {
        if (moved_node_descendant[j]) {
          new_vector[j] = old_vector[new_parent_i] ||
                          (moved_node_descendant[i] && old_vector[j]);
        } else {
          new_vector[j] = old_vector[j];
        }
      }

      new_am.ancestor[i] = new_vector;
    }

    return new_am;
  }

  /**
   * @brief Return the number of nodes in the tree.
   *
   * @return The number of nodes in the tree.
   */
  uint32_t get_n_nodes() const { return n_nodes; }

  uint32_t get_root() const { return n_nodes - 1; }

private:
  std::array<AncestryVector, max_n_nodes> ancestor;
  uint32_t n_nodes;
};
} // namespace ffSCITE