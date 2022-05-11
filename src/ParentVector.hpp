#pragma once
#include <array>
#include <bit>
#include <cstdint>
#include <random>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <unordered_map>
#include <vector>

namespace ffSCITE {
/**
 * \brief Canonical datatype to store and modify a mutation tree.
 *
 * A parent vector stores an rooted and enumerated tree, i.e. a tree where V =
 * {0, ..., n} for some natural number n. For this, it contains an array of
 * indices with n-1 elements and for every node i < n, its parent's index is
 * stored in position i. The index n is used to denote the root of the tree; it
 * does not have an entry in the parent array. This might cause some confusion:
 * The root isn't counted in most cases, so for example an instance of
 * `ParentVector<3>` has four nodes in total, but only three of them are
 * movable.
 *
 * This implementation of a parent vector also has a fixed upper size limit,
 * which aids the use of the class on FPGAs.
 *
 * Direct writing access to the parent vector is prohibited in order to ensure
 * that no cycles are introduced. The methods also assert the validity of the
 * requested operations if they are not compiled for SYCL devices.
 *
 * \tparam max_n_nodes The maximal number of free, movable nodes in the tree.
 */
template <uint64_t max_n_nodes> class ParentVector {
public:
  static constexpr uint64_t n_node_bits = std::bit_width(max_n_nodes + 1);
  using uindex_node_t = ac_int<n_node_bits, false>;

  /**
   * \brief Default constructor
   *
   * Instantiate a tree with the maximal number of nodes where all nodes are
   * connected directly to the root.
   */
  ParentVector() : parent(), n_nodes(max_n_nodes) {
    for (uindex_node_t i = 0; i < max_n_nodes; i++) {
      parent[i] = max_n_nodes;
    }
  }
  ParentVector(ParentVector const &other) = default;
  ParentVector &operator=(ParentVector const &other) = default;

  /**
   * \brief Instantiate a parent vector with a certain number of nodes.
   *
   * All nodes will be initially attached to the root.
   *
   * \param n_nodes The number of free, movable nodes in the tree.
   */
  ParentVector(uindex_node_t n_nodes) : parent(), n_nodes(n_nodes) {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(n_nodes <= max_n_nodes);
#endif

    for (uindex_node_t i = 0; i < max_n_nodes; i++) {
      parent[i] = max_n_nodes;
    }
  }

  /**
   * \brief Construct a parent vector from a tree's Prüfer Code.
   *
   * A Prüfer Code is a series of indices that uniquely describes a tree. It's
   * rather compact and can easily be used to sample a uniformly distributed
   * tree. The length of the code needs to be less than or equal to `max_n_nodes
   * - 1` since a Prüfer Code of length n describes a tree with n+2 nodes
   * including the root.
   *
   * \param pruefer_code The Prüfer Code to reconstruct.
   * \return The tree represented by the Prüfer Code.
   */
  static ParentVector<max_n_nodes>
  from_pruefer_code(std::vector<uindex_node_t> const &pruefer_code) {
    // Algorithm adapted from
    // https://en.wikipedia.org/wiki/Pr%C3%BCfer_sequence, 09th of May 2022,
    // 16:07, since the original reference implementation is sketchy.
    assert(pruefer_code.size() <= max_n_nodes - 1);
    ParentVector<max_n_nodes> pv(pruefer_code.size() + 1);
    uindex_node_t n_nodes = pv.n_nodes;

    // Compute the (resulting) degrees of every node.
    std::array<uindex_node_t, max_n_nodes + 1> degree;
    for (uindex_node_t i = 0; i < max_n_nodes + 1; i++) {
      degree[i] = 1;
    }
    for (uindex_node_t i = 0; i < pruefer_code.size(); i++) {
      degree[pruefer_code[i]]++;
    }

    // Build the tree.
    for (uindex_node_t i = 0; i < pruefer_code.size(); i++) {
      for (uindex_node_t j = 0; j < n_nodes; j++) {
        if (degree[j] == 1) {
          pv.parent[j] = pruefer_code[i];
          degree[pruefer_code[i]]--;
          degree[j]--;
          break;
        }
      }
    }

    // Construct the last edge. v is the root of tree as it's new parent has
    // never been assigned.
    uindex_node_t u = 0;
    for (uindex_node_t i = 0; i < n_nodes + 1; i++) {
      if (degree[i] == 1) {
        pv.parent[i] = n_nodes;
        break;
      }
    }

    return pv;
  }

  /**
   * \brief Generate a random, uniformly distributed tree.
   *
   * This is done by generating a random Prüfer Code and using
   * `from_pruefer_code` to construct the tree.
   *
   * \tparam The type of URNG to use.
   * \param rng The URNG instance to use.
   * \param n_nodes The number of free, movable nodes in the resulting tree.
   * \return A random, uniformly distributed tree.
   */
  template <typename RNG>
  static ParentVector<max_n_nodes> sample_random_tree(RNG &rng,
                                                      uindex_node_t n_nodes) {
    std::uniform_int_distribution<unsigned long> int_distribution(0, n_nodes);

    // Generate a pruefer code for the tree.
    std::vector<uindex_node_t> pruefer_code;
    for (uindex_node_t i = 0; i < n_nodes - 1; i++) {
      pruefer_code.push_back(int_distribution(rng));
    }

    return from_pruefer_code(pruefer_code);
  }

  uindex_node_t operator[](uindex_node_t node_i) const {
    return parent[node_i];
  }

  /**
   * \brief Get the number of free, movable nodes in the tree.
   *
   * \return The number of free, movable nodes in the tree.
   */
  uindex_node_t get_n_nodes() const { return n_nodes; }

  /**
   * \brief Check whether node a is a descendant of node b.
   *
   * This is done by following the parent vector links until either node b or
   * the root has been found. If node b was found, then a is a descendant of b,
   * otherwise not.
   *
   * \param node_a_i The index of node a.
   * \param node_b_i The index of node b.
   * \return true iff node a is a descendant of node b.
   */
  bool is_descendant(uindex_node_t node_a_i, uindex_node_t node_b_i) const {
    if (node_a_i == node_b_i) {
      return true;
    }

    while (!(node_a_i == node_b_i || node_a_i >= n_nodes)) {
      node_a_i = parent[node_a_i];
    }

    return node_a_i == node_b_i;
  }

  /**
   * \brief Swap the positions of two nodes within the tree.
   *
   * After this operation, the tree will have the same topology as before, but
   * node b will be where node a used to be, and vice versa.
   *
   * \param node_a_i One of the nodes to swap.
   * \param node_b_i The other node to swap.
   */
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

  /**
   * \brief Move the subtree below a node to a new parent.
   *
   * After that, the moved node's parent will be the requested new parent, but
   * apart from that, the tree will remain the same. If the method is not
   * compiled for a SYCL device, it will assert that the new parent is not a
   * descendant of the node to move, since this would detach one part of the
   * tree and introduce a cycle.
   *
   * \param node_i The index of the node to move.
   * \param new_parant_i The index of the node's new parent node.
   */
  void move_subtree(uindex_node_t node_i, uindex_node_t new_parent_i) {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(!is_descendant(new_parent_i, node_i));
#endif

    parent[node_i] = new_parent_i;
  }

  /**
   * \brief Compute an array that contains all nodes of the tree in
   * breadth-first order.
   *
   * Storing these indices in an array makes it later easy to statically
   * traverse the tree in breadth-first order.
   *
   * \return An array that contains all nodes of the tree in breadth-first
   * order.
   */
  std::array<uindex_node_t, max_n_nodes> calc_breadth_first_traversal() const {
    std::array<uindex_node_t, max_n_nodes> traversal;
    uindex_node_t back_of_buffer = 0;

    // Initialize with nodes connected to root, as the root will not be included
    // in the traversal
    for (uindex_node_t j = 0; j < n_nodes; j++) {
      if (parent[j] >= n_nodes) {
        traversal[back_of_buffer] = j;
        back_of_buffer++;
      }
    }

    // Now, follow the node's edges to fill the traversal.
    for (uindex_node_t traversal_i = 0; traversal_i < n_nodes; traversal_i++) {
      for (uindex_node_t j = 0; j < n_nodes; j++) {
        if (parent[j] == traversal[traversal_i]) {
          traversal[back_of_buffer] = j;
          back_of_buffer++;
        }
      }
    }

    return traversal;
  }

  /**
   * \brief Swap to subtrees in the tree.
   *
   * This operation takes the subtree below node a, hangs it next to node b, and
   * then hangs the subtree below node b to where node a used to be. If the
   * method is not compiled for a SYCL device, it will assert that the node a
   * and b are not descendants of each other, since this would detach one part
   * of the tree and introduce a cycle.
   *
   * \param node_a_i The index of one of the nodes to swap.
   * \param node_b_i The index of the other node to swap.
   */
  void swap_subtrees(uindex_node_t node_a_i, uindex_node_t node_b_i) {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(!is_descendant(node_a_i, node_b_i) &&
           !is_descendant(node_b_i, node_a_i));
#endif
    std::swap(parent[node_a_i], parent[node_b_i]);
  }

private:
  std::array<uindex_node_t, max_n_nodes> parent;
  uindex_node_t n_nodes;
};
} // namespace ffSCITE