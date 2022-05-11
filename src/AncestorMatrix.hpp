#pragma once
#include "ParentVector.hpp"

namespace ffSCITE {
/**
 * \brief A datastructure to query whether two nodes in a tree are related.
 *
 * This is done by storing a nxn-matrix of bits. According to a ancestor matrix
 * A, the node i is an ancestor of j iff A[i][j] is true (or 1). Ancestor
 * matrices are usually constructed from a \ref ParentVector.
 *
 * \tparam max_n_nodes The maximal number of free, movable nodes in tree,
 * excluding the root.
 */
template <uint64_t max_n_nodes> class AncestorMatrix {
public:
  /**
   * \brief Shorthand to the corresponding parent vector type.
   */
  using ParentVectorImpl = ParentVector<max_n_nodes>;
  /**
   * \brief Shorthand to the node index type.
   */
  using uindex_node_t = typename ParentVectorImpl::uindex_node_t;

  /**
   * \brief Default constructor
   *
   * Instantiate the ancestor matrix of a tree with the maximal number of nodes
   * where all nodes are connected directly to the root.
   */
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

  /**
   * \brief Construct the ancestor matrix for a parent vector's tree.
   *
   * This done by walking up from every node to the tree's root and marking all
   * nodes on the way. The number of nodes will be transferred from the parent
   * vector.
   *
   * \param parent_vector The parent vector to construct the ancestor matrix
   * from.
   */
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

  /**
   * \brief Query whether node a is an ancestor of node b.
   *
   * \param node_a_i The index of the potential ancestor.
   * \param node_b_i The index of the potential descendant.
   * \return true iff node a is an ancestor of node b.
   */
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

  /**
   * \brief Return a boolean array describing a node's descendants.
   *
   * For example, if this method was invoked for node a, one can query whether
   * node b is a descendant of node a by checking whether the b-th bit in the
   * array is true. This form of an array of boolean values can be used to
   * iterate over the descendants of a node.
   *
   * \param node_i The index of the node who's descendants are queried.
   * \return The descendants bit array.
   */
  std::array<bool, max_n_nodes> get_descendants(uindex_node_t node_i) const {
    std::array<bool, max_n_nodes> descendants;
    for (uindex_node_t i = 0; i < n_nodes; i++) {
      descendants[i] = (node_i >= n_nodes) || ancestor[node_i][i];
    }
    return descendants;
  }

  /**
   * \brief Get the total number of a node's descendants.
   *
   * \param node_i The index of the node who's number of descendants is queried.
   * \return The number of descendants.
   */
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

  /**
   * \brief Return the number of free, movable nodes in the tree.
   *
   * \return The number of free, movable nodes in the tree.
   */
  uindex_node_t get_n_nodes() const { return n_nodes; }

private:
  std::array<std::array<bool, max_n_nodes>, max_n_nodes> ancestor;
  uindex_node_t n_nodes;
};
} // namespace ffSCITE