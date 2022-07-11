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
#include <AncestorMatrix.hpp>
#include <catch2/catch_all.hpp>
using PV = ffSCITE::ParentVector<8>;
using AM = ffSCITE::AncestorMatrix<8>;

AM create_test_ancestor_matrix() {
  // Construct a simple, binary tree with three levels and 7 nodes:
  //
  //  ┌-6-┐
  // ┌4┐ ┌5┐
  // 0 1 2 3
  PV pv = PV::from_pruefer_code({4, 4, 5, 5, 6});
  AM ancestor_matrix(pv);
  return ancestor_matrix;
}

TEST_CASE("AncestorMatrix::is_ancestor", "[AncestorMatrix]") {
  AM ancestor_matrix = create_test_ancestor_matrix();

  REQUIRE(ancestor_matrix.is_ancestor(0, 0));
  REQUIRE(!ancestor_matrix.is_ancestor(1, 0));
  REQUIRE(!ancestor_matrix.is_ancestor(2, 0));
  REQUIRE(!ancestor_matrix.is_ancestor(3, 0));
  REQUIRE(ancestor_matrix.is_ancestor(4, 0));
  REQUIRE(!ancestor_matrix.is_ancestor(5, 0));
  REQUIRE(ancestor_matrix.is_ancestor(6, 0));

  REQUIRE(!ancestor_matrix.is_ancestor(0, 1));
  REQUIRE(ancestor_matrix.is_ancestor(1, 1));
  REQUIRE(!ancestor_matrix.is_ancestor(2, 1));
  REQUIRE(!ancestor_matrix.is_ancestor(3, 1));
  REQUIRE(ancestor_matrix.is_ancestor(4, 1));
  REQUIRE(!ancestor_matrix.is_ancestor(5, 1));
  REQUIRE(ancestor_matrix.is_ancestor(6, 1));

  REQUIRE(!ancestor_matrix.is_ancestor(0, 2));
  REQUIRE(!ancestor_matrix.is_ancestor(1, 2));
  REQUIRE(ancestor_matrix.is_ancestor(2, 2));
  REQUIRE(!ancestor_matrix.is_ancestor(3, 2));
  REQUIRE(!ancestor_matrix.is_ancestor(4, 2));
  REQUIRE(ancestor_matrix.is_ancestor(5, 2));
  REQUIRE(ancestor_matrix.is_ancestor(6, 2));

  REQUIRE(!ancestor_matrix.is_ancestor(0, 3));
  REQUIRE(!ancestor_matrix.is_ancestor(1, 3));
  REQUIRE(!ancestor_matrix.is_ancestor(2, 3));
  REQUIRE(ancestor_matrix.is_ancestor(3, 3));
  REQUIRE(!ancestor_matrix.is_ancestor(4, 3));
  REQUIRE(ancestor_matrix.is_ancestor(5, 3));
  REQUIRE(ancestor_matrix.is_ancestor(6, 3));

  REQUIRE(!ancestor_matrix.is_ancestor(0, 4));
  REQUIRE(!ancestor_matrix.is_ancestor(1, 4));
  REQUIRE(!ancestor_matrix.is_ancestor(2, 4));
  REQUIRE(!ancestor_matrix.is_ancestor(3, 4));
  REQUIRE(ancestor_matrix.is_ancestor(4, 4));
  REQUIRE(!ancestor_matrix.is_ancestor(5, 4));
  REQUIRE(ancestor_matrix.is_ancestor(6, 4));

  REQUIRE(!ancestor_matrix.is_ancestor(0, 5));
  REQUIRE(!ancestor_matrix.is_ancestor(1, 5));
  REQUIRE(!ancestor_matrix.is_ancestor(2, 5));
  REQUIRE(!ancestor_matrix.is_ancestor(3, 5));
  REQUIRE(!ancestor_matrix.is_ancestor(4, 5));
  REQUIRE(ancestor_matrix.is_ancestor(5, 5));
  REQUIRE(ancestor_matrix.is_ancestor(6, 5));

  REQUIRE(!ancestor_matrix.is_ancestor(0, 6));
  REQUIRE(!ancestor_matrix.is_ancestor(1, 6));
  REQUIRE(!ancestor_matrix.is_ancestor(2, 6));
  REQUIRE(!ancestor_matrix.is_ancestor(3, 6));
  REQUIRE(!ancestor_matrix.is_ancestor(4, 6));
  REQUIRE(!ancestor_matrix.is_ancestor(5, 6));
  REQUIRE(ancestor_matrix.is_ancestor(6, 6));
}

TEST_CASE("AncestorMatrix::is_descendant", "[AncestorMatrix]") {
  AM ancestor_matrix = create_test_ancestor_matrix();

  REQUIRE(ancestor_matrix.is_descendant(0, 0));
  REQUIRE(!ancestor_matrix.is_descendant(1, 0));
  REQUIRE(!ancestor_matrix.is_descendant(2, 0));
  REQUIRE(!ancestor_matrix.is_descendant(3, 0));
  REQUIRE(!ancestor_matrix.is_descendant(4, 0));
  REQUIRE(!ancestor_matrix.is_descendant(5, 0));

  REQUIRE(!ancestor_matrix.is_descendant(0, 1));
  REQUIRE(ancestor_matrix.is_descendant(1, 1));
  REQUIRE(!ancestor_matrix.is_descendant(2, 1));
  REQUIRE(!ancestor_matrix.is_descendant(3, 1));
  REQUIRE(!ancestor_matrix.is_descendant(4, 1));
  REQUIRE(!ancestor_matrix.is_descendant(5, 1));

  REQUIRE(!ancestor_matrix.is_descendant(0, 2));
  REQUIRE(!ancestor_matrix.is_descendant(1, 2));
  REQUIRE(ancestor_matrix.is_descendant(2, 2));
  REQUIRE(!ancestor_matrix.is_descendant(3, 2));
  REQUIRE(!ancestor_matrix.is_descendant(4, 2));
  REQUIRE(!ancestor_matrix.is_descendant(5, 2));

  REQUIRE(!ancestor_matrix.is_descendant(0, 3));
  REQUIRE(!ancestor_matrix.is_descendant(1, 3));
  REQUIRE(!ancestor_matrix.is_descendant(2, 3));
  REQUIRE(ancestor_matrix.is_descendant(3, 3));
  REQUIRE(!ancestor_matrix.is_descendant(4, 3));
  REQUIRE(!ancestor_matrix.is_descendant(5, 3));

  REQUIRE(ancestor_matrix.is_descendant(0, 4));
  REQUIRE(ancestor_matrix.is_descendant(1, 4));
  REQUIRE(!ancestor_matrix.is_descendant(2, 4));
  REQUIRE(!ancestor_matrix.is_descendant(3, 4));
  REQUIRE(ancestor_matrix.is_descendant(4, 4));
  REQUIRE(!ancestor_matrix.is_descendant(5, 4));

  REQUIRE(!ancestor_matrix.is_descendant(0, 5));
  REQUIRE(!ancestor_matrix.is_descendant(1, 5));
  REQUIRE(ancestor_matrix.is_descendant(2, 5));
  REQUIRE(ancestor_matrix.is_descendant(3, 5));
  REQUIRE(!ancestor_matrix.is_descendant(4, 5));
  REQUIRE(ancestor_matrix.is_descendant(5, 5));

  REQUIRE(ancestor_matrix.is_descendant(0, 6));
  REQUIRE(ancestor_matrix.is_descendant(1, 6));
  REQUIRE(ancestor_matrix.is_descendant(2, 6));
  REQUIRE(ancestor_matrix.is_descendant(3, 6));
  REQUIRE(ancestor_matrix.is_descendant(4, 6));
  REQUIRE(ancestor_matrix.is_descendant(5, 6));
  REQUIRE(ancestor_matrix.is_descendant(6, 6));
}

TEST_CASE("AncestorMatrix::get_descendants", "[AncestorMatrix]") {
  AM ancestor_matrix = create_test_ancestor_matrix();

  auto descendant = ancestor_matrix.get_descendants(0);
  REQUIRE(descendant[0]);
  REQUIRE(!descendant[1]);
  REQUIRE(!descendant[2]);
  REQUIRE(!descendant[3]);
  REQUIRE(!descendant[4]);
  REQUIRE(!descendant[5]);
  REQUIRE(!descendant[6]);

  descendant = ancestor_matrix.get_descendants(1);
  REQUIRE(!descendant[0]);
  REQUIRE(descendant[1]);
  REQUIRE(!descendant[2]);
  REQUIRE(!descendant[3]);
  REQUIRE(!descendant[4]);
  REQUIRE(!descendant[5]);
  REQUIRE(!descendant[6]);

  descendant = ancestor_matrix.get_descendants(2);
  REQUIRE(!descendant[0]);
  REQUIRE(!descendant[1]);
  REQUIRE(descendant[2]);
  REQUIRE(!descendant[3]);
  REQUIRE(!descendant[4]);
  REQUIRE(!descendant[5]);
  REQUIRE(!descendant[6]);

  descendant = ancestor_matrix.get_descendants(3);
  REQUIRE(!descendant[0]);
  REQUIRE(!descendant[1]);
  REQUIRE(!descendant[2]);
  REQUIRE(descendant[3]);
  REQUIRE(!descendant[4]);
  REQUIRE(!descendant[5]);
  REQUIRE(!descendant[6]);

  descendant = ancestor_matrix.get_descendants(4);
  REQUIRE(descendant[0]);
  REQUIRE(descendant[1]);
  REQUIRE(!descendant[2]);
  REQUIRE(!descendant[3]);
  REQUIRE(descendant[4]);
  REQUIRE(!descendant[5]);
  REQUIRE(!descendant[6]);

  descendant = ancestor_matrix.get_descendants(5);
  REQUIRE(!descendant[0]);
  REQUIRE(!descendant[1]);
  REQUIRE(descendant[2]);
  REQUIRE(descendant[3]);
  REQUIRE(!descendant[4]);
  REQUIRE(descendant[5]);
  REQUIRE(!descendant[6]);

  descendant = ancestor_matrix.get_descendants(6);
  REQUIRE(descendant[0]);
  REQUIRE(descendant[1]);
  REQUIRE(descendant[2]);
  REQUIRE(descendant[3]);
  REQUIRE(descendant[4]);
  REQUIRE(descendant[5]);
  REQUIRE(descendant[6]);
}

TEST_CASE("AncestorMatrix::get_n_descendants", "[AncestorMatrix]") {
  AM ancestor_matrix = create_test_ancestor_matrix();

  REQUIRE(ancestor_matrix.get_n_descendants(0) == 1);
  REQUIRE(ancestor_matrix.get_n_descendants(1) == 1);
  REQUIRE(ancestor_matrix.get_n_descendants(2) == 1);
  REQUIRE(ancestor_matrix.get_n_descendants(3) == 1);
  REQUIRE(ancestor_matrix.get_n_descendants(4) == 3);
  REQUIRE(ancestor_matrix.get_n_descendants(5) == 3);
  REQUIRE(ancestor_matrix.get_n_descendants(6) == 7);
}

TEST_CASE("AncestorMatrix::get_ancestors", "[AncestorMatrix]") {
  AM ancestor_matrix = create_test_ancestor_matrix();

  auto ancestor = ancestor_matrix.get_ancestors(0);
  REQUIRE(ancestor[0]);
  REQUIRE(!ancestor[1]);
  REQUIRE(!ancestor[2]);
  REQUIRE(!ancestor[3]);
  REQUIRE(ancestor[4]);
  REQUIRE(!ancestor[5]);
  REQUIRE(ancestor[6]);

  ancestor = ancestor_matrix.get_ancestors(1);
  REQUIRE(!ancestor[0]);
  REQUIRE(ancestor[1]);
  REQUIRE(!ancestor[2]);
  REQUIRE(!ancestor[3]);
  REQUIRE(ancestor[4]);
  REQUIRE(!ancestor[5]);
  REQUIRE(ancestor[6]);

  ancestor = ancestor_matrix.get_ancestors(2);
  REQUIRE(!ancestor[0]);
  REQUIRE(!ancestor[1]);
  REQUIRE(ancestor[2]);
  REQUIRE(!ancestor[3]);
  REQUIRE(!ancestor[4]);
  REQUIRE(ancestor[5]);
  REQUIRE(ancestor[6]);

  ancestor = ancestor_matrix.get_ancestors(3);
  REQUIRE(!ancestor[0]);
  REQUIRE(!ancestor[1]);
  REQUIRE(!ancestor[2]);
  REQUIRE(ancestor[3]);
  REQUIRE(!ancestor[4]);
  REQUIRE(ancestor[5]);
  REQUIRE(ancestor[6]);

  ancestor = ancestor_matrix.get_ancestors(4);
  REQUIRE(!ancestor[0]);
  REQUIRE(!ancestor[1]);
  REQUIRE(!ancestor[2]);
  REQUIRE(!ancestor[3]);
  REQUIRE(ancestor[4]);
  REQUIRE(!ancestor[5]);
  REQUIRE(ancestor[6]);

  ancestor = ancestor_matrix.get_ancestors(5);
  REQUIRE(!ancestor[0]);
  REQUIRE(!ancestor[1]);
  REQUIRE(!ancestor[2]);
  REQUIRE(!ancestor[3]);
  REQUIRE(!ancestor[4]);
  REQUIRE(ancestor[5]);
  REQUIRE(ancestor[6]);

  ancestor = ancestor_matrix.get_ancestors(6);
  REQUIRE(!ancestor[0]);
  REQUIRE(!ancestor[1]);
  REQUIRE(!ancestor[2]);
  REQUIRE(!ancestor[3]);
  REQUIRE(!ancestor[4]);
  REQUIRE(!ancestor[5]);
  REQUIRE(ancestor[6]);
}

TEST_CASE("AncestorMatrix::get_n_ancestors", "[AncestorMatrix]") {
  AM ancestor_matrix = create_test_ancestor_matrix();

  REQUIRE(ancestor_matrix.get_n_ancestors(0) == 3);
  REQUIRE(ancestor_matrix.get_n_ancestors(1) == 3);
  REQUIRE(ancestor_matrix.get_n_ancestors(2) == 3);
  REQUIRE(ancestor_matrix.get_n_ancestors(3) == 3);
  REQUIRE(ancestor_matrix.get_n_ancestors(4) == 2);
  REQUIRE(ancestor_matrix.get_n_ancestors(5) == 2);
  REQUIRE(ancestor_matrix.get_n_ancestors(6) == 1);
}

TEST_CASE("AncestorMatrix::swap_nodes", "[AncestorMatrix]") {
  /*
   * Original tree:
   *
   *   ┌-7-┐
   *  ┌5┐ ┌6
   * ┌2┐3 4
   * 0 1
   */
  PV pv = PV::from_pruefer_code({2, 2, 5, 5, 6, 7});
  AM original_am(pv);

  pv.swap_nodes(2, 4);
  AM rebuilt_am(pv);

  AM updated_am = original_am.swap_nodes(2, 4);

  for (uint32_t i = 0; i < 8; i++) {
    for (uint32_t j = 0; j < 8; j++) {
      REQUIRE(rebuilt_am.is_ancestor(i, j) == updated_am.is_ancestor(i, j));
    }
  }
}

TEST_CASE("AncestorMatrix::move_subtree", "[AncestorMatrix]") {
  /*
   * Original tree:
   *
   *   ┌-7-┐
   *  ┌5┐ ┌6
   * ┌2┐3 4
   * 0 1
   */
  PV pv = PV::from_pruefer_code({2, 2, 5, 5, 6, 7});
  AM original_am(pv);

  pv.move_subtree(2, 6);
  AM rebuilt_am(pv);

  AM updated_am = original_am.move_subtree(2, 6);

  for (uint32_t i = 0; i < 8; i++) {
    for (uint32_t j = 0; j < 8; j++) {
      REQUIRE(rebuilt_am.is_ancestor(i, j) == updated_am.is_ancestor(i, j));
    }
  }
}

TEST_CASE("AncestorMatrix::swap_subtrees", "[AncestorMatrix]") {
  /*
   * Original tree:
   *
   *   ┌-7-┐
   *  ┌5┐ ┌6
   * ┌2┐3 4
   * 0 1
   */
  PV pv = PV::from_pruefer_code({2, 2, 5, 5, 6, 7});
  AM original_am(pv);

  pv.swap_subtrees(2, 6);
  AM rebuilt_am(pv);

  AM updated_am = original_am.swap_subtrees(2, 5, 6, 7);

  for (uint32_t i = 0; i < 8; i++) {
    for (uint32_t j = 0; j < 8; j++) {
      REQUIRE(rebuilt_am.is_ancestor(i, j) == updated_am.is_ancestor(i, j));
    }
  }
}