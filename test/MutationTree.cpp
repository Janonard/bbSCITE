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
#include <catch2/catch_all.hpp>

using Tree = ffSCITE::MutationTree<14>;

void require_tree_equality(Tree const &a, Tree const &b) {
  REQUIRE(a.get_n_nodes() == b.get_n_nodes());

  for (uint32_t i = 0; i < a.get_n_nodes(); i++) {
    REQUIRE(a.get_parent(i) == b.get_parent(i));

    for (uint32_t j = 0; j < a.get_n_nodes(); j++) {
      REQUIRE(a.is_ancestor(i, j) == b.is_ancestor(i, j));
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
  Tree tree({2, 2, 4, 4, 4}, 0.42);

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
  Tree tree({2, 2, 4, 4, 4}, 0.42);

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
  Tree tree({2, 2, 4, 4, 4}, 0.42);

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
  Tree tree = Tree({4, 4, 5, 5, 6, 6, 6}, 0.42);

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
  Tree tree = Tree({4, 4, 5, 5, 6, 6, 6}, 0.42);

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
  Tree tree = Tree({4, 4, 5, 5, 6, 6, 6}, 0.42);

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
  Tree tree = Tree({4, 4, 5, 5, 6, 6, 6}, 0.42);

  REQUIRE(tree.get_n_ancestors(0) == 3);
  REQUIRE(tree.get_n_ancestors(1) == 3);
  REQUIRE(tree.get_n_ancestors(2) == 3);
  REQUIRE(tree.get_n_ancestors(3) == 3);
  REQUIRE(tree.get_n_ancestors(4) == 2);
  REQUIRE(tree.get_n_ancestors(5) == 2);
  REQUIRE(tree.get_n_ancestors(6) == 1);
}

TEST_CASE("MutationTree::from_pruefer_code", "[MutationTree]") {
  // Construct a simple, binary tree with three levels and 15 nodes:
  //
  //     ┌--14--┐
  //  ┌-12┐    ┌13-┐
  // ┌8┐ ┌9┐ ┌10┐ ┌11┐
  // 0 1 2 3 4  5 6  7
  std::vector<uint32_t> pruefer_code = {8,  8,  9,  9,  10, 10, 11,
                                        11, 12, 12, 13, 13, 14};

  Tree tree = Tree::from_pruefer_code(pruefer_code, 0.42);

  require_tree_equality(
      tree,
      Tree({8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 14}, 0.42));
}

TEST_CASE("MutationTree::execute_move (SwapNodes)", "[MutationTree]") {
  /*
   * Testing tree:
   *
   *  ┌r┐
   * ┌2┐3
   * 0 1
   */
  Tree original_tree({2, 2, 4, 4, 4}, 0.42);

  // Identity operation
  Tree identity_tree;
  original_tree.execute_move(identity_tree, ffSCITE::MoveType::SwapNodes, 2, 2,
                             0, 0);

  /*
   * Expected tree:
   *
   *  ┌r┐
   * ┌2┐3
   * 0 1
   */
  require_tree_equality(identity_tree, Tree({2, 2, 4, 4, 4}, 0.42));

  // Swap of unrelated nodes
  Tree unrelated_swap_tree;
  identity_tree.execute_move(unrelated_swap_tree, ffSCITE::MoveType::SwapNodes,
                             2, 3, 0, 0);

  /*
   * Expected tree:
   *
   *  ┌r┐
   * ┌3┐2
   * 0 1
   */
  require_tree_equality(unrelated_swap_tree, Tree({3, 3, 4, 4, 4}, 0.42));

  // Swap of parent and child
  Tree child_swap_tree;
  unrelated_swap_tree.execute_move(child_swap_tree,
                                   ffSCITE::MoveType::SwapNodes, 0, 3, 0, 0);

  /*
   * Expected tree:
   *
   *  ┌r┐
   * ┌0┐2
   * 3 1
   */
  require_tree_equality(child_swap_tree, Tree({4, 0, 4, 0, 4}, 0.42));
}

TEST_CASE("MutationTree::execute_move (Prune and Reattach)", "[MutationTree]") {
  /*
   * Original tree:
   *
   *   ┌-7-┐
   *  ┌5┐ ┌6
   * ┌2┐3 4
   * 0 1
   */
  Tree original_tree({2, 2, 5, 5, 6, 7, 7, 7}, 0.42);

  Tree modified_tree;
  original_tree.execute_move(modified_tree, ffSCITE::MoveType::PruneReattach, 2,
                             0, 6, 0);

  /*
   * Resulting tree:
   *
   * ┌-7-┐
   * 5┐ ┌6┐
   *  3 4┌2┐
   *     0 1
   */
  require_tree_equality(modified_tree, Tree({2, 2, 6, 5, 6, 7, 7, 7}, 0.42));
}

TEST_CASE("MutationTree::execute_move (Swap Subtrees)", "[MutationTree]") {
  /*
   * Original tree:
   *
   *   ┌-7-┐
   *  ┌5┐ ┌6
   * ┌2┐3 4
   * 0 1
   */
  Tree original_tree({2, 2, 5, 5, 6, 7, 7, 7}, 0.42);

  Tree swapped_tree;
  original_tree.execute_move(swapped_tree, ffSCITE::MoveType::SwapSubtrees, 2,
                             6, 7, 5);

  /*
   * Resulting tree:
   *
   *   ┌-7-┐
   *  ┌5┐ ┌2┐
   * ┌6 3 0 1
   * 4
   */
  require_tree_equality(swapped_tree, Tree({2, 2, 7, 5, 6, 7, 5, 7}, 0.42));
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
  Tree tree = Tree::from_pruefer_code({2, 2, 5, 5, 6, 7}, 0.42);

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
  Tree tree = Tree::from_pruefer_code({2, 2, 5, 5, 6, 7}, 0.42);

  std::string newick_string = tree.to_newick();
  REQUIRE(newick_string == required_newick_tree);
}
