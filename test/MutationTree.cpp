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
#include <ChangeProposer.hpp>
#include <MutationTree.hpp>
#include <catch2/catch_all.hpp>

constexpr uint32_t max_n_genes = 64;
constexpr uint32_t max_n_nodes = max_n_genes + 1;
using Tree = ffSCITE::MutationTree<max_n_genes>;
using AncestorMatrix = Tree::AncestorMatrix;

void require_tree_equality(Tree const &a,
                           std::vector<uint32_t> const &parent_vector) {
  REQUIRE(parent_vector.size() == a.get_n_nodes());
  AncestorMatrix b = Tree::parent_vector_to_ancestor_matrix(parent_vector);

  for (uint32_t i = 0; i < a.get_n_nodes(); i++) {
    for (uint32_t j = 0; j < a.get_n_nodes(); j++) {
      REQUIRE(a.is_ancestor(i, j) == b[i][j]);
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
  AncestorMatrix am = Tree::parent_vector_to_ancestor_matrix({2, 2, 4, 4, 4});
  Tree tree(am, 4, 0.42);

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
  AncestorMatrix am = Tree::parent_vector_to_ancestor_matrix({2, 2, 4, 4, 4});
  Tree tree(am, 4, 0.42);

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
  AncestorMatrix am = Tree::parent_vector_to_ancestor_matrix({2, 2, 4, 4, 4});
  Tree tree(am, 4, 0.42);

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
  AncestorMatrix am =
      Tree::parent_vector_to_ancestor_matrix({4, 4, 5, 5, 6, 6, 6});
  Tree tree(am, 6, 0.42);

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
  AncestorMatrix am =
      Tree::parent_vector_to_ancestor_matrix({4, 4, 5, 5, 6, 6, 6});
  Tree tree(am, 6, 0.42);

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
  AncestorMatrix am =
      Tree::parent_vector_to_ancestor_matrix({4, 4, 5, 5, 6, 6, 6});
  Tree tree(am, 6, 0.42);

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
  AncestorMatrix am =
      Tree::parent_vector_to_ancestor_matrix({4, 4, 5, 5, 6, 6, 6});
  Tree tree(am, 6, 0.42);

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

TEST_CASE("MutationTree::parent_vector_to_ancestor_matrix", "[MutationTree]") {
  // Construct a simple, binary tree with three levels and 15 nodes:
  //
  //     ┌--14--┐
  //  ┌-12┐    ┌13-┐
  // ┌8┐ ┌9┐ ┌10┐ ┌11┐
  // 0 1 2 3 4  5 6  7
  std::vector<uint32_t> parent_vector = {8,  8,  9,  9,  10, 10, 11, 11,
                                         12, 12, 13, 13, 14, 14, 14};
  AncestorMatrix am = Tree::parent_vector_to_ancestor_matrix(parent_vector);

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
}

TEST_CASE("MutationTree::execute_move (SwapNodes)", "[MutationTree]") {
  /*
   * Testing tree:
   *
   *  ┌r┐
   * ┌2┐3
   * 0 1
   */
  AncestorMatrix am = Tree::parent_vector_to_ancestor_matrix({2, 2, 4, 4, 4});
  Tree tree(am, 4, 0.42);

  // Identity operation
  AncestorMatrix identity_am;
  Tree identity_tree(identity_am, 5, 0.42);
  tree.execute_move(identity_tree, ffSCITE::MoveType::SwapNodes, 2, 2, 0, 0);

  /*
   * Expected tree:
   *
   *  ┌r┐
   * ┌2┐3
   * 0 1
   */
  require_tree_equality(identity_tree, {2, 2, 4, 4, 4});

  // Swap of unrelated nodes
  AncestorMatrix unrelated_swap_am;
  Tree unrelated_swap_tree(unrelated_swap_am, 5, 0.42);
  identity_tree.execute_move(unrelated_swap_tree, ffSCITE::MoveType::SwapNodes,
                             2, 3, 0, 0);

  /*
   * Expected tree:
   *
   *  ┌r┐
   * ┌3┐2
   * 0 1
   */
  require_tree_equality(unrelated_swap_tree, {3, 3, 4, 4, 4});

  // Swap of parent and child
  AncestorMatrix child_swap_am;
  Tree child_swap_tree(child_swap_am, 5, 0.42);
  unrelated_swap_tree.execute_move(child_swap_tree,
                                   ffSCITE::MoveType::SwapNodes, 0, 3, 0, 0);

  /*
   * Expected tree:
   *
   *  ┌r┐
   * ┌0┐2
   * 3 1
   */
  require_tree_equality(child_swap_tree, {4, 0, 4, 0, 4});
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
  AncestorMatrix am =
      Tree::parent_vector_to_ancestor_matrix({2, 2, 5, 5, 6, 7, 7, 7});
  Tree tree(am, 7, 0.42);

  AncestorMatrix modified_am;
  Tree modified_tree(modified_am, 8, 0.42);
  tree.execute_move(modified_tree, ffSCITE::MoveType::PruneReattach, 2, 0, 6,
                    0);

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

TEST_CASE("MutationTree::execute_move (Swap Subtrees)", "[MutationTree]") {
  /*
   * Original tree:
   *
   *   ┌-7-┐
   *  ┌5┐ ┌6
   * ┌2┐3 4
   * 0 1
   */
  AncestorMatrix am =
      Tree::parent_vector_to_ancestor_matrix({2, 2, 5, 5, 6, 7, 7, 7});
  Tree tree(am, 7, 0.42);

  AncestorMatrix swapped_am;
  Tree swapped_tree(swapped_am, 8, 0.42);
  tree.execute_move(swapped_tree, ffSCITE::MoveType::SwapSubtrees, 2, 6, 7, 5);

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
  AncestorMatrix am =
      Tree::parent_vector_to_ancestor_matrix({2, 2, 5, 5, 6, 7, 7, 7});
  Tree tree(am, 7, 0.42);

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
  AncestorMatrix am =
      Tree::parent_vector_to_ancestor_matrix({2, 2, 5, 5, 6, 7, 7, 7});
  Tree tree(am, 7, 0.42);

  std::string newick_string = tree.to_newick();
  REQUIRE(newick_string == required_newick_tree);
}

TEST_CASE("MutationTree::execute_move (fuzzing)", "[MutationTree]") {
  std::mt19937 twister;
  twister.seed(std::random_device()());

  ffSCITE::ChangeProposer<max_n_genes, std::mt19937> change_proposer(twister);

  constexpr uint32_t n_operations = 1000;

  std::vector<uint32_t> pruefer_code =
      Tree::sample_random_pruefer_code(twister, max_n_genes);
  std::vector<uint32_t> parent_vector =
      Tree::pruefer_code_to_parent_vector(pruefer_code);
  AncestorMatrix am = Tree::parent_vector_to_ancestor_matrix(parent_vector);

  Tree tree(am, max_n_genes, 0.42);

  for (uint32_t i_operation = 0; i_operation < n_operations; i_operation++) {
    // =========
    // Node swap
    // =========
    {
      std::array<uint32_t, 2> nodes_to_swap =
          change_proposer.sample_nonroot_nodepair(max_n_nodes);
      uint32_t v = nodes_to_swap[0];
      uint32_t w = nodes_to_swap[1];

      // Execute the move on the tree
      AncestorMatrix modified_am;
      Tree modified_tree(modified_am, max_n_genes, 0.42);
      tree.execute_move(modified_tree, ffSCITE::MoveType::SwapNodes, v, w, 0,
                        0);

      // Execute the move on the parent vector
      for (uint32_t i_node = 0; i_node < max_n_nodes; i_node++) {
        if (i_node != v && i_node != w) {
          if (parent_vector[i_node] == v) {
            parent_vector[i_node] = w;
          } else if (parent_vector[i_node] == w) {
            parent_vector[i_node] = v;
          }
        }
      }
      if (parent_vector[v] == w) {
        parent_vector[v] = parent_vector[w];
        parent_vector[w] = v;
      } else if (parent_vector[w] == v) {
        parent_vector[w] = parent_vector[v];
        parent_vector[v] = w;
      } else {
        std::swap(parent_vector[v], parent_vector[w]);
      }

      // Verify the results
      require_tree_equality(modified_tree, parent_vector);
      am = modified_am;
    }

    // ==================
    // Prune and reattach
    // ==================
    {
      std::array<uint32_t, 2> params =
          change_proposer.sample_prune_and_reattach_parameters(tree);
      uint32_t v = params[0];
      uint32_t v_target = params[1];

      // Execute the move on the tree
      AncestorMatrix modified_am;
      Tree modified_tree(modified_am, max_n_genes, 0.42);
      tree.execute_move(modified_tree, ffSCITE::MoveType::PruneReattach, v, 0,
                        v_target, 0);

      // Execute the move on the parent vector
      parent_vector[v] = v_target;

      // Verify the results
      require_tree_equality(modified_tree, parent_vector);
      am = modified_am;
    }

    // ========
    // Treeswap
    // ========
    {
      float neighborhood_correction;
      std::array<uint32_t, 4> params =
          change_proposer.sample_treeswap_parameters(tree,
                                                     neighborhood_correction);
      uint32_t v = params[0];
      uint32_t w = params[1];
      uint32_t v_target = params[2];
      uint32_t w_target = params[3];

      if (tree.is_ancestor(w, v)) {
        REQUIRE(tree.is_ancestor(v, w_target));
        REQUIRE(tree.is_parent(v_target, w));
      } else {
        REQUIRE(!tree.is_ancestor(v, w));
        REQUIRE(tree.is_parent(v_target, w));
        REQUIRE(tree.is_parent(w_target, v));
      }

      // Execute the move on the tree
      AncestorMatrix modified_am;
      Tree modified_tree(modified_am, max_n_genes, 0.42);
      tree.execute_move(modified_tree, ffSCITE::MoveType::SwapSubtrees, v, w,
                        v_target, w_target);

      // Execute the move on the parent vector
      parent_vector[v] = v_target;
      parent_vector[w] = w_target;

      // Verify the results
      require_tree_equality(modified_tree, parent_vector);
      am = modified_am;
    }
  }
}