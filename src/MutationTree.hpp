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
#include "MoveType.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <oneapi/dpl/random>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <unordered_map>
#include <vector>

namespace ffSCITE {
/**
 * @brief Representation of a mutation tree, with relevant operations.
 *
 * A mutation tree is a tree which contains a dedicated root node and one node
 * for every gene. Then, cells are attached to a node of the tree, which is
 * equivalent to saying that this cell has a mutation at a gene iff this gene's
 * node is on the path from the root node to the gene. ffSCITE tries different
 * mutation trees and keeps the one that is most-likely to be true. Therefore,
 * this class contains operations to quickly query whether there is a path in
 * the tree from one node to another and to randomly modify the tree.
 *
 * Internally, the mutation tree is represented as an ancestor matrix: This is a
 * data structure with an entry for every node pair x and y that contains a 1
 * iff there is a path from x to y. Otherwise, it contains a 0. Technically,
 * ancestor matrices are implemented as an array of n-bit words called ancestry
 * vectors, where the x is the index of the word and y is the index of the bit
 * within the word. A MutationTree instance requires a reference to such an
 * array since oneAPI can not implement arrays inside classes as independent
 * memory blocks, at least at the time of this writing.
 *
 * @tparam max_n_genes The maximal number of genes this tree supports. The
 * maximal number of nodes in a tree is this number plus one since every
 * mutation tree has one node per gene and one additional root node.
 */
template <uint32_t max_n_genes> class MutationTree {
public:
  /**
   * @brief The maximal number of nodes in a mutation tree supported by the
   * class.
   *
   * This is always the maximal number of genes of the mutation tree plus one.
   */
  static constexpr uint32_t max_n_nodes = max_n_genes + 1;

  /**
   * @brief Type of the ancestor matrix rows.
   */
  using AncestryVector = ac_int<max_n_nodes, false>;

  /**
   * @brief Type of the internal ancestor matrix representation.
   */
  using AncestorMatrix = std::array<AncestryVector, max_n_nodes>;

  /**
   * @brief Parameter bundle for a tree modification.
   *
   * The move type enumeration describes the exact modification to execute. v,
   * w, descendant_of_v and nondescendant_of_v are all randomly sampled, where v
   * is sampled uniformly from all nodes, w is sampled uniformly from all nodes
   * except v, descendant_of_v is sampled uniformly from all descendants of v,
   * and nondescendant_of_v is sampled uniformly from all nondescendants of v.
   * new_beta contains a new, possibly modified beta value (probability of false
   * negatives).
   */
  struct ModificationParameters {
    MoveType move_type;
    uint32_t v, w, parent_of_v, parent_of_w, descendant_of_v,
        nondescendant_of_v;
    float new_beta;
  };

  /**
   * @brief Construct a new Mutation Tree without initializing the ancestor
   * matrix.
   *
   * This constructor assumes that the ancestor matrix has been properly
   * initialized and simply assumes it as the new internal state.
   *
   * This method works equally well on CPUs and on FPGAs.
   *
   * @param ancestor A reference to an ancestor matrix.
   * @param n_genes The number of genes represented by the mutation tree.
   * @param beta The probability for false negatives.
   */
  MutationTree(AncestorMatrix &ancestor, uint32_t n_genes, float beta)
      : ancestor(ancestor), n_nodes(n_genes + 1), beta(beta) {}

  /**
   * @brief Construct a new Mutation Tree by computing a modified version of an
   * old mutation tree.
   *
   * This constructor applies the changes parametrized by the parameters struct
   * and stores the resulting mutation tree in it's internal state.
   *
   * This method is designed and optimized for FPGAs, but works on CPUs too.
   *
   * @param am A reference to an ancestor matrix that is used as the internal
   * container of this tree.
   * @param old_tree The old tree to base the new tree of.
   * @param parameters The parameters of the modification.
   */
  MutationTree(AncestorMatrix &am, MutationTree<max_n_genes> const &old_tree,
               ModificationParameters parameters)
      : ancestor(am), n_nodes(old_tree.n_nodes), beta(old_tree.beta) {
    if (parameters.move_type == MoveType::ChangeBeta) {
      beta = parameters.new_beta;
    }

    uint32_t v = parameters.v;
    uint32_t w = parameters.w;

    AncestryVector v_descendant = old_tree.ancestor[v];
    AncestryVector w_descendant = old_tree.ancestor[w];

    uint32_t v_target, w_target;
    switch (parameters.move_type) {
    case MoveType::SwapSubtrees:
      v_target = parameters.parent_of_w;
      w_target =
          w_descendant[v] ? parameters.descendant_of_v : parameters.parent_of_v;
      break;
    case MoveType::PruneReattach:
      v_target = parameters.nondescendant_of_v;
      w_target = 0; // No target necessary.
      break;
    case MoveType::ChangeBeta:
    case MoveType::SwapNodes:
    default:
      v_target = w_target = 0; // No targets necessary.
    }

    for (uint32_t x = 0; x < max_n_nodes; x++) {
      // Compute the new ancestry vector.
      AncestryVector old_vector = old_tree.ancestor[x];
      AncestryVector new_vector = 0;

      // Declaring the swap variable for the "Swap Nodes" move here since you
      // can't declare it inside the switch statement.
      bool swap = true;

      switch (parameters.move_type) {
      case MoveType::ChangeBeta:
        new_vector = old_vector;
        break;

      case MoveType::SwapNodes:
        if (x == v) {
          new_vector = w_descendant;
        } else if (x == w) {
          new_vector = v_descendant;
        } else {
          new_vector = old_vector;
        }

        swap = new_vector[v];
        new_vector[v] = new_vector[w];
        new_vector[w] = swap;
        break;

      case MoveType::PruneReattach:
#pragma unroll
        for (uint32_t y = 0; y < max_n_nodes; y++) {
          if (v_descendant[y]) {
            // if (v -> y),
            // we have (x -> y) <=> (x -> v_target) || (v -> x -> y)
            new_vector[y] =
                old_vector[v_target] || (v_descendant[x] && old_vector[y]);
          } else {
            // otherwise, we have (v !-> y).
            // Since this node is unaffected, everything remains the same.
            new_vector[y] = old_vector[y];
          }
        }
        break;

      case MoveType::SwapSubtrees:
#pragma unroll
        for (uint32_t y = 0; y < max_n_nodes; y++) {
          if (w_descendant[v]) {
            ac_int<2, false> class_x;
            if (v_descendant[x]) {
              class_x = 2;
            } else if (w_descendant[x]) {
              class_x = 1;
            } else {
              class_x = 0;
            }

            ac_int<2, false> class_y;
            if (v_descendant[y]) {
              class_y = 2;
            } else if (w_descendant[y]) {
              class_y = 1;
            } else {
              class_y = 0;
            }

            if ((class_x == class_y) ||
                (class_x == 0 && (class_y == 1 || class_y == 2))) {
              new_vector[y] = old_vector[y];
            } else if (class_x == 2 && class_y == 1) {
              new_vector[y] = old_vector[w_target];
            } else {
              new_vector[y] = false;
            }
          } else {
            if (v_descendant[y] && !w_descendant[y]) {
              // if (v -> y && w !-> y),
              // we have (x -> y) <=> (x -> v_target) || (v -> x -> y)
              new_vector[y] =
                  old_vector[v_target] || (v_descendant[x] && old_vector[y]);
            } else if (!v_descendant[y] && w_descendant[y]) {
              // if (v !-> y && w -> y),
              // we have (x -> y) <=> (x -> w_target) || (w -> x -> y)
              new_vector[y] =
                  old_vector[w_target] || (w_descendant[x] && old_vector[y]);
            } else {
              // we have (v !-> y && w !-> y), (v -> y && w -> y) is impossible.
              // In this case, everything remains the same.
              new_vector[y] = old_vector[y];
            }
          }
        }
        break;

      default:
        break;
      }

      ancestor[x] = new_vector;
    }
  }

  MutationTree(MutationTree const &other) = default;
  MutationTree<max_n_genes> &
  operator=(MutationTree<max_n_genes> const &other) = default;

  /**
   * @brief Compute the ancestor matrix of the tree described by the given
   * parent vector.
   *
   * A parent vector is an array of indices which contains the parent of the
   * node i at the position i.
   *
   * This method is designed for CPUs and is not usable on FPGAs.
   *
   * @param parent_vector The parent vector to compute the ancestor matrix from.
   * @return AncestorMatrix The ancestor matrix of the tree described by the
   * parent vector.
   */
  static AncestorMatrix
  parent_vector_to_ancestor_matrix(std::vector<uint32_t> const &parent_vector) {
    AncestorMatrix ancestor;
    uint32_t n_nodes = parent_vector.size();
    uint32_t root = n_nodes - 1;

    for (uint32_t j = 0; j < max_n_nodes; j++) {
      // Zero all vectors. This is equivalent to setting everything to false.
      ancestor[j] = 0;
    }

    for (uint32_t i = 0; i < max_n_nodes; i++) {
      if (i < n_nodes) {
        // Then we start from the node i and walk up to the root, marking all
        // nodes on the way as ancestors.
        uint32_t anc = i;
        while (anc != root) {
          ancestor[anc][i] = true;
          anc = parent_vector[anc];
          // Otherwise, there is a circle in the graph!
          assert(anc != i && anc < n_nodes);
        }
      }

      // Lastly, also mark the root as our ancestor.
      ancestor[i][i] = ancestor[root][i] = true;
    }

    return ancestor;
  }

  /**
   * @brief Compute the parent vector of the tree described by the given Prüfer
   * code.
   *
   * A Prüfer code is a sequence of indices that describe a tree. Prüfer codes
   * are used to randomly generate trees since any sequence of integers with n
   * elements and entries <= (n+2) encode a tree with (n+2) nodes.
   *
   * A parent vector is an array of indices which contains the parent of the
   * node i at the position i.
   *
   * This method is designed for CPUs and is not usable on FPGAs.
   *
   * @param pruefer_code The Prüfer code of the tree.
   * @return std::vector<uint32_t> The parent vector of the tree.
   */
  static std::vector<uint32_t>
  pruefer_code_to_parent_vector(std::vector<uint32_t> const &pruefer_code) {
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

    return parent_vector;
  }

  /**
   * @brief Generate a random, uniformly distributed tree, encoded as a Prüfer
   * code.
   *
   * A Prüfer code is a sequence of indices that describe a tree. Prüfer codes
   * are used to randomly generate trees since any sequence of integers with n
   * elements and entries <= (n+2) encode a tree with (n+2) nodes.
   *
   * @tparam The type of URNG to use.
   * @param rng The URNG instance to use.
   * @param n_nodes The number of nodes in the resulting tree, must be lower
   * than or equal to `max_n_nodes`.
   * @return A random, uniformly distributed tree, encoded as a Prüfer code.
   */
  template <typename RNG>
  static std::vector<uint32_t> sample_random_pruefer_code(RNG &rng,
                                                          uint32_t n_genes) {
    uint32_t n_nodes = n_genes + 1;
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

    return pruefer_code;
  }

  /**
   * @brief Get the current beta value (Probability of false negatives).
   *
   * @return float The beta value (Probability of false negatives).
   */
  float get_beta() const { return beta; }

  /**
   * @brief Set the current beta value (Probability of false negatives).
   *
   * @param new_beta The new beta value (Probability of false negatives).
   */
  void set_beta(float new_beta) { beta = new_beta; }

  /**
   * @brief Get the index of the root node.
   *
   * @return uint32_t The index of the root node.
   */
  uint32_t get_root() const { return n_nodes - 1; }

  /**
   * @brief Check whether one node is the parent of the other node in this tree.
   *
   * This is equivalent to asking whether there is an edge from `parent` to
   * `child` in the tree. However, the root is its own parent per convention,
   * which means that this edge only exists if `parent` is the parent of `child`
   * and `child` is not the root of the tree.
   *
   * This method is designed and optimized for FPGAs, but works on CPUs too.
   *
   * @param parent The index of the possible parent node.
   * @param child  The index of the possible child node.
   * @return true `parent` is the parent of `child`.
   * @return false `parent``is not the parent of `child`.
   */
  bool is_parent(uint32_t parent, uint32_t child) const {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(parent < n_nodes && child < n_nodes);
#endif
    if (!ancestor[parent][child]) {
      return false;
    }

    if (parent == child) {
      return child == get_root();
    }

#pragma unroll
    for (uint32_t node_i = 0; node_i < max_n_nodes; node_i++) {
      if (node_i >= n_nodes || node_i == child) {
        continue;
      }
      if (ancestor[node_i][parent] != ancestor[node_i][child]) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Get the parent of a node in the tree.
   *
   * This method is designed and optimized for FPGAs, but works on CPUs too.
   *
   * @param node_i The node who's parent is searched.
   * @return uint32_t The parent of the node.
   */
  uint32_t get_parent(uint32_t node_i) const {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(node_i < max_n_nodes);
#endif
    for (uint32_t parent = 0; parent < max_n_nodes; parent++) {
      if (parent < n_nodes && is_parent(parent, node_i)) {
        return parent;
      }
    }
    assert(false);
    return 0; // Illegal option, will not occur if the tree is correct.
  }

  uint32_t get_n_nodes() const { return n_nodes; }

  /**
   * @brief Query whether node a is an ancestor of node b.
   *
   * This method is designed and optimized for FPGAs, but works on CPUs too.
   *
   * @param node_a_i The index of the potential ancestor.
   * @param node_b_i The index of the potential descendant.
   * @return true iff node a is an ancestor of node b.
   */
  bool is_ancestor(uint32_t node_a_i, uint32_t node_b_i) const {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(node_a_i < max_n_nodes && node_b_i < max_n_nodes);
#endif
    return ancestor[node_a_i][node_b_i];
  }

  /**
   * @brief Query whether node a is a descendant of node b.
   *
   * This method is designed and optimized for FPGAs, but works on CPUs too.
   *
   * @param node_a_i The index of the potential descendant.
   * @param node_b_i The index of the potential ancestor.
   * @return true iff node a is an ancestor of node b.
   */
  bool is_descendant(uint32_t node_a_i, uint32_t node_b_i) const {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(node_a_i < max_n_nodes && node_b_i < max_n_nodes);
#endif
    return ancestor[node_b_i][node_a_i];
  }

  /**
   * @brief Return an ancestry vector describing a node's descendants.
   *
   * For example, if this method was invoked for node a, one can query whether
   * node b is a descendant of node a by checking whether the bit with index b
   * in the array is true. This representation can be used to iterate over the
   * descendants of a node.
   *
   * This method is designed and optimized for FPGAs, but works on CPUs too.
   *
   * @param node_i The index of the node who's descendants are queried.
   * @return The descendants bit array.
   */
  AncestryVector get_descendants(uint32_t node_i) const {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(node_i < max_n_nodes);
#endif
    return ancestor[node_i];
  }

  /**
   * @brief Get the total number of descendants of a node.
   *
   * This method is designed and optimized for FPGAs, but works on CPUs too.
   *
   * @param node_i The index of the node who's number of descendants is queried.
   * @return The number of descendants.
   */
  uint32_t get_n_descendants(uint32_t node_i) const {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(node_i < max_n_nodes);
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
   * in the array is true. This representation can be used to iterate over the
   * ancestors of a node.
   *
   * This method is designed and optimized for FPGAs, but works on CPUs too.
   *
   * @param node_i The index of the node who's ancestors are queried.
   * @return The ancestors bit array.
   */
  AncestryVector get_ancestors(uint32_t node_i) const {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(node_i < max_n_nodes);
#endif
    AncestryVector ancestors = 0;
#pragma unroll
    for (uint32_t i = 0; i < max_n_nodes; i++) {
      if (i < n_nodes) {
        ancestors[i] = is_ancestor(i, node_i);
      }
    }
    return ancestors;
  }

  /**
   * @brief Get the total number of ancestors of a node.
   *
   * This method is designed and optimized for FPGAs, but works on CPUs too.
   *
   * @param node_i The index of the node who's number of ancestors is queried.
   * @return The number of ancestors.
   */
  uint32_t get_n_ancestors(uint32_t node_i) const {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(node_i < max_n_nodes);
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
   * @brief Compare two trees for equality.
   *
   * Two trees are equal iff their number of nodes is equal, they contain the
   * same edges, and contain the same beta value.
   *
   * This method is designed and optimized for FPGAs, but works on CPUs too.
   *
   * @param other The other tree to compare too.
   * @return true The two trees are equal.
   * @return false The two trees are not equal.
   */
  bool operator==(MutationTree<max_n_genes> const &other) const {
    if (n_nodes != other.n_nodes || beta != other.beta) {
      return false;
    }
#pragma unroll
    for (uint32_t node_i = 0; node_i < max_n_nodes; node_i++) {
      for (uint32_t node_j = 0; node_j < max_n_nodes; node_j++) {
        if (node_i < n_nodes && node_j < n_nodes &&
            ancestor[node_i][node_j] != other.ancestor[node_i][node_j]) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * @brief Compare two parent vectors for inequality.
   *
   * Two trees are equal iff their number of nodes is equal, they contain the
   * same edges, and contain the same beta value.
   *
   * This method is designed and optimized for FPGAs, but works on CPUs too.
   *
   * @param other The other tree to compare too.
   * @return true The two trees are not equal.
   * @return false The two trees are equal.
   */
  bool operator!=(MutationTree<max_n_genes> const &other) const {
    return !operator==(other);
  }

  /**
   * @brief Sample one of the random move types, with customized move
   * probabilities.
   *
   * The probabilities for the beta change, the "prune and reattach" move, and
   * the "swap nodes" move are parameters of this method, and it is assumed that
   * all of these probabilities are non-negative and that their sum is less than
   * or equal to one. The probability of the "swap subtrees" move is the
   * remaining probability given the other probabilities.
   *
   * This method works equally well on CPUs and on FPGAs.
   *
   * @tparam The type of URNG to use.
   * @param rng The URNG instance to use.
   * @param prob_beta_change The probability of a beta-changing move.
   * @param prob_prune_n_reattach The probability of a "prune and reattach"
   * move.
   * @param prob_swap_nodes The probability of a "swap nodes" move.
   * @return MoveType The sampled move type.
   */
  template <typename RNG>
  MoveType sample_move(RNG &rng, float prob_beta_change,
                       float prob_prune_n_reattach,
                       float prob_swap_nodes) const {
    float change_type_draw =
        oneapi::dpl::uniform_real_distribution(0.0, 1.0)(rng);
    if (change_type_draw <= prob_beta_change) {
      return MoveType::ChangeBeta;
    } else if (change_type_draw <= prob_beta_change + prob_prune_n_reattach) {
      return MoveType::PruneReattach;
    } else if (change_type_draw <=
               prob_beta_change + prob_prune_n_reattach + prob_swap_nodes) {
      return MoveType::SwapNodes;
    } else {
      return MoveType::SwapSubtrees;
    }
  }

  /**
   * @brief Sample one of the random move types, with default probabilities.
   *
   * The default probabilities for the different move types are:
   *
   * * beta change: 0
   * * "prune and reattach": 0.5
   * * "swap nodes": 0.45
   * * "swap subtrees": 0.05
   *
   * This method works equally well on CPUs and on FPGAs.
   *
   * @tparam The type of URNG to use.
   * @param rng The URNG instance to use.
   * @return MoveType The sampled move type.
   */
  template <typename RNG> MoveType sample_move(RNG &rng) const {
    return sample_move(rng, 0.0, 0.5, 0.45);
  }

  /**
   * @brief Sample two distinct nodes uniformly from the tree.
   *
   * The first node v is sampled uniformly from the entire tree and the second
   * node w is sampled from all nodes except v. v and w are either unrelated or
   * w is an ancestor of v.
   *
   * This method works equally well on CPUs and on FPGAs.
   *
   * @tparam The type of URNG to use.
   * @param rng The URNG instance to use.
   * @return std::array<uint32_t, 2> Two distinct nodes from the tree.
   */
  template <typename RNG>
  std::array<uint32_t, 2> sample_nonroot_nodepair(RNG &rng) const {
    // excluding n_nodes - 1, the root.
    uint32_t v = oneapi::dpl::uniform_int_distribution<uint32_t>(
        0, get_n_nodes() - 2)(rng);
    uint32_t w = oneapi::dpl::uniform_int_distribution<uint32_t>(
        0, get_n_nodes() - 3)(rng);
    if (w >= v) {
      w++;
    }
    if (is_ancestor(v, w)) {
      std::swap(v, w);
    }
    return {v, w};
  }

  /**
   * @brief Sample uniformly from all descendants of a node.
   *
   * Note that the sampled node may be `node_i` too.
   *
   * This method is designed and optimized for FPGAs, but works on CPUs too.
   *
   * @tparam The type of URNG to use.
   * @param rng The URNG instance to use.
   * @param node_i The pivot node who's descendants are considered.
   * @return uint32_t A descendant of `node_i`.
   */
  template <typename RNG>
  uint32_t sample_descendant(RNG &rng, uint32_t node_i) const {
    return sample_descendant_or_nondescendant(rng, node_i, true);
  }

  /**
   * @brief Sample uniformly from all non-descendants of a node.
   *
   * Note that the sampled node may be the root of the tree too if `node_i` is
   * not the root.
   *
   * This method is designed and optimized for FPGAs, but works on CPUs too.
   *
   * @tparam The type of URNG to use.
   * @param rng The URNG instance to use.
   * @param node_i The pivot node who's non-descendants are considered.
   * @return uint32_t A non-descendant of `node_i`.
   */
  template <typename RNG>
  uint32_t sample_nondescendant(RNG &rng, uint32_t node_i) const {
    return sample_descendant_or_nondescendant(rng, node_i, false);
  }

  /**
   * @brief Sample a new beta value (Probability of false negatives).
   *
   * This is done by adding a normally distributed value to the current beta
   * value and bounding it to [0,1].
   *
   * This method works equally well on CPUs and on FPGAs.
   *
   * @tparam The type of URNG to use.
   * @param rng The URNG instance to use.
   * @param beta_jump_sd The standard derivation of the added value.
   * @return float The newly sampled beta value.
   */
  template <typename RNG>
  float sample_new_beta(RNG &rng, float beta_jump_sd) const {
    float new_beta =
        beta + oneapi::dpl::normal_distribution<float>(0, beta_jump_sd)(rng);
    if (new_beta < 0) {
      new_beta = std::abs(new_beta);
    }
    if (new_beta > 1) {
      new_beta = new_beta - 2 * (new_beta - 1);
    }
    return new_beta;
  }

  /**
   * @brief Generate a graphviz file that plots the tree.
   *
   * This method is designed for CPUs and is not usable on FPGAs.
   *
   * @return std::string A graphviz file that plots the tree.
   */
  std::string to_graphviz() const {
    std::stringstream stream;
    stream << "digraph G {" << std::endl;
    stream << "node [color=deeppink4, style=filled, fontcolor=white];"
           << std::endl;
    for (uint32_t node_i = 0; node_i < n_nodes - 1; node_i++) {
      stream << get_parent(node_i) << " -> " << node_i << ";" << std::endl;
    }
    stream << "}" << std::endl;
    return stream.str();
  }

  /**
   * @brief Generate the Newick code of the tree.
   *
   * This method is designed for CPUs and is not usable on FPGAs.
   *
   * @return std::string The Newick code of the tree.
   */
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
      children[get_parent(node_i)].push_back(node_i);
    }

    std::stringstream stream;
    add_node_to_newick_code(children, stream, get_root());
    stream << std::endl;
    return stream.str();
  }

private:
  /**
   * @brief Sample uniformly from either all descendants or non-descendants of a
   * node.
   *
   * Depending on the `sample_descendant` flag, this method either samples a
   * node uniformly from all descendants of `node_i` or all non-descendants of
   * `node_i`.
   *
   * When sampling from all descendants, the sampled node may also be `node_i`
   * itself, and when sampling from all non-descendants, the sampled node may
   * also be the root of the tree if `node_i` is not the root.
   *
   * This method is designed and optimized for FPGAs, but works on CPUs too.
   *
   * @tparam The type of URNG to use.
   * @param rng The URNG instance to use.
   * @param node_i The pivot node who's descendants or non-descendants are
   * considered.
   * @param sample_descendant Set to true iff the sampled node should be sampled
   * from all descendants of `node_i`. Otherwise, set to false.
   * @return uint32_t A (non-)descendant of `node_i`.
   */
  template <typename RNG>
  uint32_t sample_descendant_or_nondescendant(RNG &rng, uint32_t node_i,
                                              bool sample_descendant) const {
    AncestryVector descendant = get_descendants(node_i);

    // If we have to sample a nondescendant, we invert the bitvector and
    // continue as if we were to sample a descendant.
    if (!sample_descendant) {
#pragma unroll
      for (uint32_t i = 0; i < max_n_nodes; i++) {
        descendant[i] = !descendant[i];
      }
    }

    // Count the (non)descendants.
    uint32_t n_descendants = 0;
#pragma unroll
    for (uint32_t i = 0; i < max_n_nodes; i++) {
      if (i < get_n_nodes() && descendant[i]) {
        n_descendants++;
      }
    }

    // Sample the occurrence of the (non)descendant to pick. The resulting node
    // will be the `sampled_occurrence_i`th (non)descendant.
    uint32_t sampled_occurrence_i =
        oneapi::dpl::uniform_int_distribution<uint32_t>(0,
                                                        n_descendants - 1)(rng);

    // Walk through the (non)descendant bitvector and pick the correct node
    // index.
    uint32_t sampled_node_i = 0;
#pragma unroll
    for (uint32_t i = 0; i < max_n_nodes; i++) {
      if (i < get_n_nodes() && descendant[i]) {
        if (sampled_occurrence_i == 0) {
          sampled_node_i = i;
          break;
        } else {
          sampled_occurrence_i--;
        }
      }
    }
    return sampled_node_i;
  }

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

  std::array<AncestryVector, max_n_nodes> &ancestor;
  uint32_t n_nodes;
  float beta;
};
} // namespace ffSCITE