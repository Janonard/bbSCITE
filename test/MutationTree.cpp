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
#include <MutationTree.hpp>
#include <RawMoveDistribution.hpp>

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <catch2/catch_all.hpp>
#include <map>
#include <set>

constexpr uint32_t max_n_genes = 31;
constexpr uint32_t max_n_nodes = max_n_genes + 1;
constexpr uint32_t n_fuzzing_iterations = 10000;
constexpr float hypothesis_test_alpha = 0.05;
using Tree = bbSCITE::MutationTree<max_n_genes>;
using ModificationParameters = Tree::ModificationParameters;
using AncestorMatrix = Tree::AncestorMatrix;

void require_tree_equality(Tree const &a,
                           std::vector<uint32_t> const &parent_vector) {
  REQUIRE(parent_vector.size() == a.get_n_nodes());
  auto matrix_tuple = Tree::parent_vector_to_matrix(parent_vector);
  AncestorMatrix b = std::get<0>(matrix_tuple);

  for (uint32_t i = 0; i < a.get_n_nodes(); i++) {
    REQUIRE(a.get_parent(i) == parent_vector[i]);
    for (uint32_t j = 0; j < a.get_n_nodes(); j++) {
      REQUIRE(a.is_ancestor(i, j) == bool(b[i][j]));
      REQUIRE(a.is_descendant(i, j) == bool(b[j][i]));
    }
  }
}

TEST_CASE("MutationTree::get_parent", "[MutationTree]") {
  /*
   * Testing tree:
   *
   *  ┌r┐
   * ┌2┐3
   * 0 1
   */
  auto matrix_tuple = Tree::parent_vector_to_matrix({2, 2, 4, 4, 4});
  AncestorMatrix am = std::get<0>(matrix_tuple);
  AncestorMatrix dm = std::get<1>(matrix_tuple);
  Tree tree(am, dm, 4, 0.42);

  REQUIRE(tree.get_parent(0) == 2);
  REQUIRE(tree.get_parent(1) == 2);
  REQUIRE(tree.get_parent(2) == 4);
  REQUIRE(tree.get_parent(3) == 4);
  REQUIRE(tree.get_parent(4) == 4);
}

TEST_CASE("MutationTree::is_ancestor", "[MutationTree]") {
  /*
   * Testing tree:
   *
   *  ┌r┐
   * ┌2┐3
   * 0 1
   */
  auto matrix_tuple = Tree::parent_vector_to_matrix({2, 2, 4, 4, 4});
  AncestorMatrix am = std::get<0>(matrix_tuple);
  AncestorMatrix dm = std::get<1>(matrix_tuple);
  Tree tree(am, dm, 4, 0.42);

  REQUIRE(tree.is_ancestor(0, 0));
  REQUIRE(!tree.is_ancestor(1, 0));
  REQUIRE(tree.is_ancestor(2, 0));
  REQUIRE(!tree.is_ancestor(3, 0));
  REQUIRE(tree.is_ancestor(4, 0));

  REQUIRE(!tree.is_ancestor(0, 1));
  REQUIRE(tree.is_ancestor(1, 1));
  REQUIRE(tree.is_ancestor(2, 1));
  REQUIRE(!tree.is_ancestor(3, 1));
  REQUIRE(tree.is_ancestor(4, 1));

  REQUIRE(!tree.is_ancestor(0, 2));
  REQUIRE(!tree.is_ancestor(1, 2));
  REQUIRE(tree.is_ancestor(2, 2));
  REQUIRE(!tree.is_ancestor(3, 2));
  REQUIRE(tree.is_ancestor(4, 2));

  REQUIRE(!tree.is_ancestor(0, 3));
  REQUIRE(!tree.is_ancestor(1, 3));
  REQUIRE(!tree.is_ancestor(2, 3));
  REQUIRE(tree.is_ancestor(3, 3));
  REQUIRE(tree.is_ancestor(4, 3));

  REQUIRE(!tree.is_ancestor(0, 4));
  REQUIRE(!tree.is_ancestor(1, 4));
  REQUIRE(!tree.is_ancestor(2, 4));
  REQUIRE(!tree.is_ancestor(3, 4));
  REQUIRE(tree.is_ancestor(4, 4));
}

TEST_CASE("MutationTree::is_descendant", "[MutationTree]") {
  /*
   * Testing tree:
   *
   *  ┌r┐
   * ┌2┐3
   * 0 1
   */
  auto matrix_tuple = Tree::parent_vector_to_matrix({2, 2, 4, 4, 4});
  AncestorMatrix am = std::get<0>(matrix_tuple);
  AncestorMatrix dm = std::get<1>(matrix_tuple);
  Tree tree(am, dm, 4, 0.42);

  REQUIRE(tree.is_descendant(0, 0));
  REQUIRE(!tree.is_descendant(0, 1));
  REQUIRE(tree.is_descendant(0, 2));
  REQUIRE(!tree.is_descendant(0, 3));
  REQUIRE(tree.is_descendant(0, 4));

  REQUIRE(!tree.is_descendant(1, 0));
  REQUIRE(tree.is_descendant(1, 1));
  REQUIRE(tree.is_descendant(1, 2));
  REQUIRE(!tree.is_descendant(1, 3));
  REQUIRE(tree.is_descendant(1, 4));

  REQUIRE(!tree.is_descendant(2, 0));
  REQUIRE(!tree.is_descendant(2, 1));
  REQUIRE(tree.is_descendant(2, 2));
  REQUIRE(!tree.is_descendant(2, 3));
  REQUIRE(tree.is_descendant(2, 4));

  REQUIRE(!tree.is_descendant(3, 0));
  REQUIRE(!tree.is_descendant(3, 1));
  REQUIRE(!tree.is_descendant(3, 2));
  REQUIRE(tree.is_descendant(3, 3));
  REQUIRE(tree.is_descendant(3, 4));

  REQUIRE(!tree.is_descendant(4, 0));
  REQUIRE(!tree.is_descendant(4, 1));
  REQUIRE(!tree.is_descendant(4, 2));
  REQUIRE(!tree.is_descendant(4, 3));
  REQUIRE(tree.is_descendant(4, 4));
}

TEST_CASE("MutationTree::get_descendants", "[MutationTree]") {
  // Construct a simple, binary tree with three levels and 7 nodes:
  //
  //  ┌-6-┐
  // ┌4┐ ┌5┐
  // 0 1 2 3
  auto matrix_tuple = Tree::parent_vector_to_matrix({4, 4, 5, 5, 6, 6, 6});
  AncestorMatrix am = std::get<0>(matrix_tuple);
  AncestorMatrix dm = std::get<1>(matrix_tuple);
  Tree tree(am, dm, 6, 0.42);

  auto descendant = tree.get_descendants(0);
  REQUIRE(descendant[0]);
  REQUIRE(!descendant[1]);
  REQUIRE(!descendant[2]);
  REQUIRE(!descendant[3]);
  REQUIRE(!descendant[4]);
  REQUIRE(!descendant[5]);
  REQUIRE(!descendant[6]);

  descendant = tree.get_descendants(1);
  REQUIRE(!descendant[0]);
  REQUIRE(descendant[1]);
  REQUIRE(!descendant[2]);
  REQUIRE(!descendant[3]);
  REQUIRE(!descendant[4]);
  REQUIRE(!descendant[5]);
  REQUIRE(!descendant[6]);

  descendant = tree.get_descendants(2);
  REQUIRE(!descendant[0]);
  REQUIRE(!descendant[1]);
  REQUIRE(descendant[2]);
  REQUIRE(!descendant[3]);
  REQUIRE(!descendant[4]);
  REQUIRE(!descendant[5]);
  REQUIRE(!descendant[6]);

  descendant = tree.get_descendants(3);
  REQUIRE(!descendant[0]);
  REQUIRE(!descendant[1]);
  REQUIRE(!descendant[2]);
  REQUIRE(descendant[3]);
  REQUIRE(!descendant[4]);
  REQUIRE(!descendant[5]);
  REQUIRE(!descendant[6]);

  descendant = tree.get_descendants(4);
  REQUIRE(descendant[0]);
  REQUIRE(descendant[1]);
  REQUIRE(!descendant[2]);
  REQUIRE(!descendant[3]);
  REQUIRE(descendant[4]);
  REQUIRE(!descendant[5]);
  REQUIRE(!descendant[6]);

  descendant = tree.get_descendants(5);
  REQUIRE(!descendant[0]);
  REQUIRE(!descendant[1]);
  REQUIRE(descendant[2]);
  REQUIRE(descendant[3]);
  REQUIRE(!descendant[4]);
  REQUIRE(descendant[5]);
  REQUIRE(!descendant[6]);

  descendant = tree.get_descendants(6);
  REQUIRE(descendant[0]);
  REQUIRE(descendant[1]);
  REQUIRE(descendant[2]);
  REQUIRE(descendant[3]);
  REQUIRE(descendant[4]);
  REQUIRE(descendant[5]);
  REQUIRE(descendant[6]);
}

TEST_CASE("MutationTree::get_n_descendants", "[MutationTree]") {
  // Construct a simple, binary tree with three levels and 7 nodes:
  //
  //  ┌-6-┐
  // ┌4┐ ┌5┐
  // 0 1 2 3
  auto matrix_tuple = Tree::parent_vector_to_matrix({4, 4, 5, 5, 6, 6, 6});
  AncestorMatrix am = std::get<0>(matrix_tuple);
  AncestorMatrix dm = std::get<0>(matrix_tuple);
  Tree tree(am, dm, 6, 0.42);

  REQUIRE(tree.get_n_descendants(0) == 1);
  REQUIRE(tree.get_n_descendants(1) == 1);
  REQUIRE(tree.get_n_descendants(2) == 1);
  REQUIRE(tree.get_n_descendants(3) == 1);
  REQUIRE(tree.get_n_descendants(4) == 3);
  REQUIRE(tree.get_n_descendants(5) == 3);
  REQUIRE(tree.get_n_descendants(6) == 7);
}

TEST_CASE("MutationTree::get_ancestors", "[MutationTree]") {
  // Construct a simple, binary tree with three levels and 7 nodes:
  //
  //  ┌-6-┐
  // ┌4┐ ┌5┐
  // 0 1 2 3
  auto matrix_tuple = Tree::parent_vector_to_matrix({4, 4, 5, 5, 6, 6, 6});
  AncestorMatrix am = std::get<0>(matrix_tuple);
  AncestorMatrix dm = std::get<1>(matrix_tuple);
  Tree tree(am, dm, 6, 0.42);

  auto ancestor = tree.get_ancestors(0);
  REQUIRE(ancestor[0]);
  REQUIRE(!ancestor[1]);
  REQUIRE(!ancestor[2]);
  REQUIRE(!ancestor[3]);
  REQUIRE(ancestor[4]);
  REQUIRE(!ancestor[5]);
  REQUIRE(ancestor[6]);

  ancestor = tree.get_ancestors(1);
  REQUIRE(!ancestor[0]);
  REQUIRE(ancestor[1]);
  REQUIRE(!ancestor[2]);
  REQUIRE(!ancestor[3]);
  REQUIRE(ancestor[4]);
  REQUIRE(!ancestor[5]);
  REQUIRE(ancestor[6]);

  ancestor = tree.get_ancestors(2);
  REQUIRE(!ancestor[0]);
  REQUIRE(!ancestor[1]);
  REQUIRE(ancestor[2]);
  REQUIRE(!ancestor[3]);
  REQUIRE(!ancestor[4]);
  REQUIRE(ancestor[5]);
  REQUIRE(ancestor[6]);

  ancestor = tree.get_ancestors(3);
  REQUIRE(!ancestor[0]);
  REQUIRE(!ancestor[1]);
  REQUIRE(!ancestor[2]);
  REQUIRE(ancestor[3]);
  REQUIRE(!ancestor[4]);
  REQUIRE(ancestor[5]);
  REQUIRE(ancestor[6]);

  ancestor = tree.get_ancestors(4);
  REQUIRE(!ancestor[0]);
  REQUIRE(!ancestor[1]);
  REQUIRE(!ancestor[2]);
  REQUIRE(!ancestor[3]);
  REQUIRE(ancestor[4]);
  REQUIRE(!ancestor[5]);
  REQUIRE(ancestor[6]);

  ancestor = tree.get_ancestors(5);
  REQUIRE(!ancestor[0]);
  REQUIRE(!ancestor[1]);
  REQUIRE(!ancestor[2]);
  REQUIRE(!ancestor[3]);
  REQUIRE(!ancestor[4]);
  REQUIRE(ancestor[5]);
  REQUIRE(ancestor[6]);

  ancestor = tree.get_ancestors(6);
  REQUIRE(!ancestor[0]);
  REQUIRE(!ancestor[1]);
  REQUIRE(!ancestor[2]);
  REQUIRE(!ancestor[3]);
  REQUIRE(!ancestor[4]);
  REQUIRE(!ancestor[5]);
  REQUIRE(ancestor[6]);
}

TEST_CASE("MutationTree::get_n_ancestors", "[MutationTree]") {
  // Construct a simple, binary tree with three levels and 7 nodes:
  //
  //  ┌-6-┐
  // ┌4┐ ┌5┐
  // 0 1 2 3
  auto matrix_tuple = Tree::parent_vector_to_matrix({4, 4, 5, 5, 6, 6, 6});
  AncestorMatrix am = std::get<0>(matrix_tuple);
  AncestorMatrix dm = std::get<1>(matrix_tuple);
  Tree tree(am, dm, 6, 0.42);

  REQUIRE(tree.get_n_ancestors(0) == 3);
  REQUIRE(tree.get_n_ancestors(1) == 3);
  REQUIRE(tree.get_n_ancestors(2) == 3);
  REQUIRE(tree.get_n_ancestors(3) == 3);
  REQUIRE(tree.get_n_ancestors(4) == 2);
  REQUIRE(tree.get_n_ancestors(5) == 2);
  REQUIRE(tree.get_n_ancestors(6) == 1);
}

TEST_CASE("MutationTree::pruefer_code_to_parent_vector", "[MutationTree]") {
  // Construct a simple, binary tree with three levels and 15 nodes:
  //
  //     ┌--14--┐
  //  ┌-12┐    ┌13-┐
  // ┌8┐ ┌9┐ ┌10┐ ┌11┐
  // 0 1 2 3 4  5 6  7
  std::vector<uint32_t> pruefer_code = {8,  8,  9,  9,  10, 10, 11,
                                        11, 12, 12, 13, 13, 14};

  std::vector<uint32_t> parent_vector =
      Tree::pruefer_code_to_parent_vector(pruefer_code);

  REQUIRE(parent_vector == std::vector<uint32_t>({8, 8, 9, 9, 10, 10, 11, 11,
                                                  12, 12, 13, 13, 14, 14, 14}));
}

TEST_CASE("MutationTree::parent_vector_to_matrix", "[MutationTree]") {
  // Construct a simple, binary tree with three levels and 15 nodes:
  //
  //     ┌--14--┐
  //  ┌-12┐    ┌13-┐
  // ┌8┐ ┌9┐ ┌10┐ ┌11┐
  // 0 1 2 3 4  5 6  7
  std::vector<uint32_t> parent_vector = {8,  8,  9,  9,  10, 10, 11, 11,
                                         12, 12, 13, 13, 14, 14, 14};
  auto matrix_tuple = Tree::parent_vector_to_matrix(parent_vector);
  AncestorMatrix am = std::get<0>(matrix_tuple);
  AncestorMatrix dm = std::get<1>(matrix_tuple);

  REQUIRE(am[0] == 0b000000000000001);
  REQUIRE(am[1] == 0b000000000000010);
  REQUIRE(am[2] == 0b000000000000100);
  REQUIRE(am[3] == 0b000000000001000);
  REQUIRE(am[4] == 0b000000000010000);
  REQUIRE(am[5] == 0b000000000100000);
  REQUIRE(am[6] == 0b000000001000000);
  REQUIRE(am[7] == 0b000000010000000);
  REQUIRE(am[8] == 0b000000100000011);
  REQUIRE(am[9] == 0b000001000001100);
  REQUIRE(am[10] == 0b000010000110000);
  REQUIRE(am[11] == 0b000100011000000);
  REQUIRE(am[12] == 0b001001100001111);
  REQUIRE(am[13] == 0b010110011110000);
  REQUIRE(am[14] % (1 << 15) == 0b111111111111111);

  REQUIRE(dm[0] == 0b101000100000001);
  REQUIRE(dm[1] == 0b101000100000010);
  REQUIRE(dm[2] == 0b101001000000100);
  REQUIRE(dm[3] == 0b101001000001000);
  REQUIRE(dm[4] == 0b110010000010000);
  REQUIRE(dm[5] == 0b110010000100000);
  REQUIRE(dm[6] == 0b110100001000000);
  REQUIRE(dm[7] == 0b110100010000000);
  REQUIRE(dm[8] == 0b101000100000000);
  REQUIRE(dm[9] == 0b101001000000000);
  REQUIRE(dm[10] == 0b110010000000000);
  REQUIRE(dm[11] == 0b110100000000000);
  REQUIRE(dm[12] == 0b101000000000000);
  REQUIRE(dm[13] == 0b110000000000000);
  REQUIRE(dm[14] == 0b100000000000000);
}

TEST_CASE("MutationTree update constructor, swap nodes", "[MutationTree]") {
  /*
   * Testing tree:
   *
   *  ┌r┐
   * ┌2┐3
   * 0 1
   */
  auto matrix_tuple = Tree::parent_vector_to_matrix({2, 2, 4, 4, 4});
  AncestorMatrix am = std::get<0>(matrix_tuple);
  AncestorMatrix dm = std::get<1>(matrix_tuple);
  Tree tree(am, dm, 4, 0.42);

  // Identity operation
  Tree::ModificationParameters parameters{
      .move_type = bbSCITE::MoveType::SwapNodes,
      .v = 2,
      .w = 2,
      .parent_of_v = 4,
      .parent_of_w = 4,
      .descendant_of_v = 0,
      .nondescendant_of_v = 3,
      .new_beta = 0.42,
  };
  AncestorMatrix identity_am, identity_dm;
  Tree identity_tree(identity_am, identity_dm, tree, parameters);

  /*
   * Expected tree:
   *
   *  ┌r┐
   * ┌2┐3
   * 0 1
   */
  require_tree_equality(identity_tree, {2, 2, 4, 4, 4});

  // Swap of unrelated nodes
  parameters = {
      .move_type = bbSCITE::MoveType::SwapNodes,
      .v = 2,
      .w = 3,
      .parent_of_v = 4,
      .parent_of_w = 4,
      .descendant_of_v = 0,
      .nondescendant_of_v = 3,
      .new_beta = 0.42,
  };
  AncestorMatrix unrelated_swap_am, unrelated_swap_dm;
  Tree unrelated_swap_tree(unrelated_swap_am, unrelated_swap_dm, identity_tree,
                           parameters);

  /*
   * Expected tree:
   *
   *  ┌r┐
   * ┌3┐2
   * 0 1
   */
  require_tree_equality(unrelated_swap_tree, {3, 3, 4, 4, 4});

  // Swap of parent and child
  parameters = {
      .move_type = bbSCITE::MoveType::SwapNodes,
      .v = 0,
      .w = 3,
      .parent_of_v = 3,
      .parent_of_w = 4,
      .descendant_of_v = 0,
      .nondescendant_of_v = 3,
      .new_beta = 0.42,
  };
  AncestorMatrix child_swap_am, child_swap_dm;
  Tree child_swap_tree(child_swap_am, child_swap_dm, unrelated_swap_tree,
                       parameters);

  /*
   * Expected tree:
   *
   *  ┌r┐
   * ┌0┐2
   * 3 1
   */
  require_tree_equality(child_swap_tree, {4, 0, 4, 0, 4});
}

TEST_CASE("MutationTree update constructor (prune and reattach)",
          "[MutationTree]") {
  /*
   * Original tree:
   *
   *   ┌-7-┐
   *  ┌5┐ ┌6
   * ┌2┐3 4
   * 0 1
   */
  auto matrix_tuple = Tree::parent_vector_to_matrix({2, 2, 5, 5, 6, 7, 7, 7});
  AncestorMatrix am = std::get<0>(matrix_tuple);
  AncestorMatrix dm = std::get<1>(matrix_tuple);
  Tree tree(am, dm, 7, 0.42);

  Tree::ModificationParameters parameters{
      .move_type = bbSCITE::MoveType::PruneReattach,
      .v = 2,
      .w = 0,
      .parent_of_v = 5,
      .parent_of_w = 2,
      .descendant_of_v = 0,
      .nondescendant_of_v = 6,
      .new_beta = 0.42,
  };
  AncestorMatrix modified_am, modified_dm;
  Tree modified_tree(modified_am, modified_dm, tree, parameters);

  /*
   * Resulting tree:
   *
   * ┌-7-┐
   * 5┐ ┌6┐
   *  3 4┌2┐
   *     0 1
   */
  require_tree_equality(modified_tree, {2, 2, 6, 5, 6, 7, 7, 7});
}

TEST_CASE("MutationTree update constructor (swap unrelated subtrees)",
          "[MutationTree]") {
  /*
   * Original tree:
   *
   *   ┌-7-┐
   *  ┌5┐ ┌6
   * ┌2┐3 4
   * 0 1
   */
  auto matrix_tuple = Tree::parent_vector_to_matrix({2, 2, 5, 5, 6, 7, 7, 7});
  AncestorMatrix am = std::get<0>(matrix_tuple);
  AncestorMatrix dm = std::get<1>(matrix_tuple);
  Tree tree(am, dm, 7, 0.42);

  Tree::ModificationParameters parameters{
      .move_type = bbSCITE::MoveType::SwapSubtrees,
      .v = 2,
      .w = 6,
      .parent_of_v = 5,
      .parent_of_w = 7,
      .descendant_of_v = 0,
      .nondescendant_of_v = 3,
      .new_beta = 0.42,
  };
  AncestorMatrix swapped_am, swapped_dm;
  Tree swapped_tree(swapped_am, swapped_dm, tree, parameters);

  /*
   * Resulting tree:
   *
   *   ┌-7-┐
   *  ┌5┐ ┌2┐
   * ┌6 3 0 1
   * 4
   */
  require_tree_equality(swapped_tree, {2, 2, 7, 5, 6, 7, 5, 7});
}

TEST_CASE("MutationTree update constructor (swap related subtrees)",
          "[MutationTree]") {
  /*
   * Original tree:
   *
   *   ┌-7-┐
   *  ┌5┐ ┌6
   * ┌2┐3 4
   * 0 1
   */
  auto matrix_tuple = Tree::parent_vector_to_matrix({2, 2, 5, 5, 6, 7, 7, 7});
  AncestorMatrix am = std::get<0>(matrix_tuple);
  AncestorMatrix dm = std::get<1>(matrix_tuple);
  Tree tree(am, dm, 7, 0.42);

  Tree::ModificationParameters parameters{
      .move_type = bbSCITE::MoveType::SwapSubtrees,
      .v = 2,
      .w = 5,
      .parent_of_v = 5,
      .parent_of_w = 7,
      .descendant_of_v = 0,
      .nondescendant_of_v = 3,
      .new_beta = 0.42,
  };
  AncestorMatrix swapped_am, swapped_dm;
  Tree swapped_tree(swapped_am, swapped_dm, tree, parameters);

  /*
   * Resulting tree:
   *
   *    ┌-7-┐
   *   ┌2┐ ┌6
   *  ┌0 1 4
   * ┌5
   * 3
   */
  require_tree_equality(swapped_tree, {2, 2, 7, 5, 6, 0, 7, 7});
}

const std::string required_graphviz_tree =
    "digraph G {\n"
    "node [color=deeppink4, style=filled, fontcolor=white];\n"
    "2 -> 0;\n"
    "2 -> 1;\n"
    "5 -> 2;\n"
    "5 -> 3;\n"
    "6 -> 4;\n"
    "7 -> 5;\n"
    "7 -> 6;\n"
    "}\n";

TEST_CASE("MutationTree::to_graphviz", "[MutationTree]") {
  /*
   * Tree:
   *
   *   ┌-7-┐
   *  ┌5┐ ┌6
   * ┌2┐3 4
   * 0 1
   */
  auto matrix_tuple = Tree::parent_vector_to_matrix({2, 2, 5, 5, 6, 7, 7, 7});
  AncestorMatrix am = std::get<0>(matrix_tuple);
  AncestorMatrix dm = std::get<1>(matrix_tuple);
  Tree tree(am, dm, 7, 0.42);

  std::string graphviz_string = tree.to_graphviz();
  REQUIRE(graphviz_string == required_graphviz_tree);
}

const std::string required_newick_tree = "(((0,1)2,3)5,(4)6)7\n";

TEST_CASE("MutationTree::to_newick", "[MutationTree]") {
  /*
   * Tree:
   *
   *   ┌-7-┐
   *  ┌5┐ ┌6
   * ┌2┐3 4
   * 0 1
   */
  auto matrix_tuple = Tree::parent_vector_to_matrix({2, 2, 5, 5, 6, 7, 7, 7});
  AncestorMatrix am = std::get<0>(matrix_tuple);
  AncestorMatrix dm = std::get<1>(matrix_tuple);
  Tree tree(am, dm, 7, 0.42);

  std::string newick_string = tree.to_newick();
  REQUIRE(newick_string == required_newick_tree);
}

TEST_CASE("MutationTree update constructor (fuzzing)", "[MutationTree]") {
  uint32_t n_genes = 16;

  std::mt19937 rng;
  rng.seed(std::random_device()());
  bbSCITE::RawMoveDistribution move_distribution(n_genes + 1, 0, 0.33, 0.33,
                                                 1.0);

  std::vector<uint32_t> pruefer_code =
      Tree::sample_random_pruefer_code(rng, n_genes);
  std::vector<uint32_t> parent_vector =
      Tree::pruefer_code_to_parent_vector(pruefer_code);
  auto matrix_tuple = Tree::parent_vector_to_matrix(parent_vector);
  AncestorMatrix am = std::get<0>(matrix_tuple);
  AncestorMatrix dm = std::get<1>(matrix_tuple);

  Tree tree(am, dm, n_genes, 0.42);

  for (uint32_t i_operation = 0; i_operation < n_fuzzing_iterations;
       i_operation++) {
    auto move_sample = move_distribution(rng);
    auto move_params = tree.realize_raw_move_sample(move_sample);

    uint32_t v = move_params.v;
    uint32_t w = move_params.w;
    uint32_t parent_of_v = move_params.parent_of_v;
    uint32_t parent_of_w = move_params.parent_of_w;
    uint32_t descendant_of_v = move_params.descendant_of_v;
    uint32_t nondescendant_of_v = move_params.nondescendant_of_v;
    bbSCITE::MoveType move_type = move_params.move_type;
    if (move_type == bbSCITE::MoveType::ChangeBeta) {
      continue;
    }

    // Execute the move manually
    std::vector<uint32_t> modified_vector = parent_vector;

    switch (move_type) {
    case bbSCITE::MoveType::SwapNodes:
      for (uint32_t i_node = 0; i_node < modified_vector.size(); i_node++) {
        if (i_node != v && i_node != w) {
          if (modified_vector[i_node] == v) {
            modified_vector[i_node] = w;
          } else if (modified_vector[i_node] == w) {
            modified_vector[i_node] = v;
          }
        }
      }
      if (modified_vector[v] == w) {
        modified_vector[v] = modified_vector[w];
        modified_vector[w] = v;
      } else if (modified_vector[w] == v) {
        modified_vector[w] = modified_vector[v];
        modified_vector[v] = w;
      } else {
        std::swap(modified_vector[v], modified_vector[w]);
      }
      break;
    case bbSCITE::MoveType::PruneReattach:
      modified_vector[v] = nondescendant_of_v;
      break;
    case bbSCITE::MoveType::SwapSubtrees:
      modified_vector[v] = parent_of_w;
      if (tree.is_ancestor(w, v)) {
        modified_vector[w] = descendant_of_v;
      } else {
        modified_vector[w] = parent_of_v;
      }
      break;
    case bbSCITE::MoveType::ChangeBeta:
    default:
      break;
    }

    // Construct the updated tree
    AncestorMatrix modified_am, modified_dm;
    Tree modified_tree(modified_am, modified_dm, tree, move_params);

    // Verify the results
    require_tree_equality(modified_tree, modified_vector);
  }
}