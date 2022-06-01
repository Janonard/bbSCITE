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
#include <unordered_map>
#include <vector>

namespace ffSCITE {
/**
 * @brief Canonical datatype to store and modify a mutation tree.
 *
 * A parent vector stores an rooted and enumerated tree, i.e. a tree where V =
 * {0, ..., n-1} for some natural number n. For this, it contains an array of
 * indices with n elements and for every node i < n, its parent's index is
 * stored in position i. The index n-1 is used to denote the root of the tree
 * and it's parent entry is always itself, which helps to identify the root of
 * the tree. Due to this static nature of the root, it is not mutable, it can
 * not be swapped or attached to another node.
 *
 * This implementation of a parent vector also has a fixed upper size limit,
 * which aids the use of the class on FPGAs.
 *
 * Direct writing access to the parent vector is prohibited in order to ensure
 * that no cycles are introduced. The methods also assert the validity of the
 * requested operations if they are not compiled for SYCL devices.
 *
 * @tparam max_n_nodes The maximal number of nodes in the tree.
 */
template <uint64_t max_n_nodes> class ParentVector {
public:
  /**
   * @brief Default constructor
   *
   * Instantiate a tree with the maximal number of nodes where all nodes are
   * connected directly to the root.
   */
  ParentVector() : parent(), n_nodes(max_n_nodes) {
    for (uint64_t i = 0; i < max_n_nodes; i++) {
      parent[i] = get_root();
    }
  }
  ParentVector(ParentVector const &other) = default;
  ParentVector &operator=(ParentVector const &other) = default;

  /**
   * @brief Instantiate a parent vector with a certain number of nodes.
   *
   * All nodes will be initially attached to the root.
   *
   * @param n_nodes The number of nodes in the tree.
   */
  ParentVector(uint64_t n_nodes) : parent(), n_nodes(n_nodes) {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(n_nodes <= max_n_nodes);
#endif

    for (uint64_t i = 0; i < max_n_nodes; i++) {
      parent[i] = get_root();
    }
  }

  /**
   * @brief Return the node index of the root.
   *
   * This index is always the number of nodes minus 1.
   *
   * @return The node index of the root.
   */
  uint64_t get_root() const { return n_nodes - 1; }

  /**
   * @brief Construct a parent vector from a tree's Prüfer Code.
   *
   * A Prüfer Code is a series of indices that uniquely describes a tree. It's
   * rather compact and can easily be used to sample a uniformly distributed
   * tree. The length of the code needs to be less than or equal to `max_n_nodes
   * - 2` since a Prüfer Code of length n describes a tree with n+2 nodes. This
   * is asserted if the method is not compiled for a SYCL device.
   *
   * @param pruefer_code The Prüfer Code to reconstruct.
   * @return The tree represented by the Prüfer Code.
   */
  static ParentVector<max_n_nodes>
  from_pruefer_code(std::vector<uint64_t> const &pruefer_code) {
    // Algorithm adapted from
    // https://en.wikipedia.org/wiki/Pr%C3%BCfer_sequence, 09th of May 2022,
    // 16:07, since the original reference implementation is sketchy.
#if __SYCL_DEVICE_ONLY__ == 0
    assert(pruefer_code.size() <= max_n_nodes - 2);
#endif
    ParentVector<max_n_nodes> pv(pruefer_code.size() + 2);
    uint64_t n_nodes = pv.n_nodes;

    // Compute the (resulting) degrees of every node.
    std::array<uint64_t, max_n_nodes> degree;
    for (uint64_t i = 0; i < max_n_nodes; i++) {
      degree[i] = 1;
    }
    for (uint64_t i = 0; i < pruefer_code.size(); i++) {
      degree[pruefer_code[i]]++;
    }

    // Build the tree.
    for (uint64_t i = 0; i < pruefer_code.size(); i++) {
      for (uint64_t j = 0; j < n_nodes; j++) {
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
    uint64_t u = 0;
    for (uint64_t i = 0; i < n_nodes; i++) {
      if (degree[i] == 1) {
        pv.parent[i] = pv.get_root();
        break;
      }
    }

    return pv;
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
  static ParentVector<max_n_nodes> sample_random_tree(RNG &rng,
                                                      uint64_t n_nodes) {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(n_nodes <= max_n_nodes);
#endif

    oneapi::dpl::uniform_int_distribution<unsigned long> int_distribution(
        0, n_nodes - 1);

    // Generate a pruefer code for the tree.
    std::vector<uint64_t> pruefer_code;
    for (uint64_t i = 0; i < n_nodes - 2; i++) {
      pruefer_code.push_back(int_distribution(rng));
    }

    return from_pruefer_code(pruefer_code);
  }

  uint64_t operator[](uint64_t node_i) const { return parent[node_i]; }

  /**
   * @brief Get the number of nodes in the tree.
   *
   * @return The number of nodes in the tree.
   */
  uint64_t get_n_nodes() const { return n_nodes; }

  /**
   * @brief Check whether node a is a descendant of node b.
   *
   * This is done by following the parent vector links until either node b or
   * the root has been found. If node b was found, then a is a descendant of b,
   * otherwise not.
   *
   * @param node_a_i The index of node a.
   * @param node_b_i The index of node b.
   * @return true iff node a is a descendant of node b.
   */
  bool is_descendant(uint64_t node_a_i, uint64_t node_b_i) const {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(node_a_i < n_nodes && node_b_i < n_nodes);
#endif

    if (node_a_i == node_b_i) {
      return true;
    }

    while (!(node_a_i == node_b_i || node_a_i == get_root())) {
      node_a_i = parent[node_a_i];
    }

    return node_a_i == node_b_i;
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
  void swap_nodes(uint64_t node_a_i, uint64_t node_b_i) {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(node_a_i != get_root() && node_b_i != get_root());
#endif

    if (node_a_i == node_b_i) {
      return;
    }

    for (uint64_t i = 0; i < n_nodes; i++) {
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
  void move_subtree(uint64_t node_i, uint64_t new_parent_i) {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(!is_descendant(new_parent_i, node_i) && node_i != get_root());
#endif

    parent[node_i] = new_parent_i;
  }

  /**
   * @brief Compute an array that contains all nodes of the tree in
   * breadth-first order, starting from the root.
   *
   * Storing these indices in an array makes it later easy to statically
   * traverse the tree in breadth-first order.
   *
   * @return An array that contains all nodes of the tree in breadth-first
   * order.
   */
  std::array<uint64_t, max_n_nodes> calc_breadth_first_traversal() const {
    std::array<uint64_t, max_n_nodes> traversal;
    traversal[0] = get_root();
    uint64_t back_of_buffer = 1;

    // Follow the node's edges to fill the traversal.
    for (uint64_t traversal_i = 0; traversal_i < n_nodes; traversal_i++) {
      // don't look up the root's parent, it is already included in the walk.
      for (uint64_t j = 0; j < n_nodes - 1; j++) {
        if (parent[j] == traversal[traversal_i]) {
          traversal[back_of_buffer] = j;
          back_of_buffer++;
        }
      }
    }

    return traversal;
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
  void swap_subtrees(uint64_t node_a_i, uint64_t node_b_i) {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(node_a_i != get_root() && node_b_i != get_root() &&
           !is_descendant(node_a_i, node_b_i) &&
           !is_descendant(node_b_i, node_a_i));
#endif
    std::swap(parent[node_a_i], parent[node_b_i]);
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
  bool operator==(ParentVector<max_n_nodes> const &other) const {
    if (n_nodes != other.n_nodes) {
      return false;
    }
    for (uint64_t node_i = 0; node_i < n_nodes; node_i++) {
      if (parent[node_i] != other.parent[node_i]) {
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
  bool operator!=(ParentVector<max_n_nodes> const &other) const {
    return !operator==(other);
  }

private:
  std::array<uint64_t, max_n_nodes> parent;
  uint64_t n_nodes;
};
} // namespace ffSCITE