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
#include <ParentVector.hpp>
#include <catch2/catch_all.hpp>

using PV = ffSCITE::ParentVector<15>;

PV build_test_tree() {
  /*
   * Initial tree:
   *
   * ┌-┌r┐-┐
   * 0 1 2 3
   */
  PV pv(5);
  REQUIRE(pv[0] == pv.get_root());
  REQUIRE(pv[1] == pv.get_root());
  REQUIRE(pv[2] == pv.get_root());
  REQUIRE(pv[3] == pv.get_root());
  REQUIRE(pv[4] == pv.get_root());

  /*
   * Actual testing tree:
   *
   *  ┌r┐
   * ┌2┐3
   * 0 1
   */
  pv.move_subtree(0, 2);
  pv.move_subtree(1, 2);

  REQUIRE(pv[0] == 2);
  REQUIRE(pv[1] == 2);
  REQUIRE(pv[2] == pv.get_root());
  REQUIRE(pv[3] == pv.get_root());
  REQUIRE(pv[4] == pv.get_root());

  return pv;
}

TEST_CASE("ParentVector: Ancestry Queries", "[ParentVector]") {
  PV pv = build_test_tree();

  REQUIRE(pv.is_descendant(0, 0));
  REQUIRE(!pv.is_descendant(0, 1));
  REQUIRE(pv.is_descendant(0, 2));
  REQUIRE(!pv.is_descendant(0, 3));

  REQUIRE(!pv.is_descendant(1, 0));
  REQUIRE(pv.is_descendant(1, 1));
  REQUIRE(pv.is_descendant(1, 2));
  REQUIRE(!pv.is_descendant(1, 3));

  REQUIRE(!pv.is_descendant(2, 0));
  REQUIRE(!pv.is_descendant(2, 1));
  REQUIRE(pv.is_descendant(2, 2));
  REQUIRE(!pv.is_descendant(2, 3));

  REQUIRE(!pv.is_descendant(3, 0));
  REQUIRE(!pv.is_descendant(3, 1));
  REQUIRE(!pv.is_descendant(3, 2));
  REQUIRE(pv.is_descendant(3, 3));

  REQUIRE(!pv.is_descendant(4, 0));
  REQUIRE(!pv.is_descendant(4, 1));
  REQUIRE(!pv.is_descendant(4, 2));
  REQUIRE(!pv.is_descendant(4, 3));
}

TEST_CASE("ParentVector::from_pruefer_code", "[ParentVector]") {
  // Construct a simple, binary tree with three levels and 15 nodes:
  //
  //     ┌--14--┐
  //  ┌-12┐    ┌13-┐
  // ┌8┐ ┌9┐ ┌10┐ ┌11┐
  // 0 1 2 3 4  5 6  7
  std::vector<uint64_t> pruefer_code = {8,  8,  9,  9,  10, 10, 11,
                                        11, 12, 12, 13, 13, 14};

  PV pv = PV::from_pruefer_code(pruefer_code);

  REQUIRE(pv[0] == 8);
  REQUIRE(pv[1] == 8);
  REQUIRE(pv[2] == 9);
  REQUIRE(pv[3] == 9);
  REQUIRE(pv[4] == 10);
  REQUIRE(pv[5] == 10);
  REQUIRE(pv[6] == 11);
  REQUIRE(pv[7] == 11);
  REQUIRE(pv[8] == 12);
  REQUIRE(pv[9] == 12);
  REQUIRE(pv[10] == 13);
  REQUIRE(pv[11] == 13);
  REQUIRE(pv[12] == pv.get_root());
  REQUIRE(pv[13] == pv.get_root());
  REQUIRE(pv[14] == pv.get_root());
}

TEST_CASE("ParentVector::swap_nodes", "[ParentVector]") {
  PV pv = build_test_tree();

  // Identity operation
  pv.swap_nodes(2, 2);

  /*
   * Expected tree:
   *
   *  ┌r┐
   * ┌2┐3
   * 0 1
   */
  REQUIRE(pv[0] == 2);
  REQUIRE(pv[1] == 2);
  REQUIRE(pv[2] == pv.get_root());
  REQUIRE(pv[3] == pv.get_root());
  REQUIRE(pv[4] == pv.get_root());

  // Swap of unrelated nodes
  pv.swap_nodes(2, 3);

  /*
   * Expected tree:
   *
   *  ┌r┐
   * ┌3┐2
   * 0 1
   */
  REQUIRE(pv[0] == 3);
  REQUIRE(pv[1] == 3);
  REQUIRE(pv[2] == pv.get_root());
  REQUIRE(pv[3] == pv.get_root());
  REQUIRE(pv[4] == pv.get_root());

  // Swap of parent and child
  pv.swap_nodes(0, 3);

  /*
   * Expected tree:
   *
   *  ┌r┐
   * ┌0┐2
   * 3 1
   */
  REQUIRE(pv[0] == pv.get_root());
  REQUIRE(pv[1] == 0);
  REQUIRE(pv[2] == pv.get_root());
  REQUIRE(pv[3] == 0);
  REQUIRE(pv[4] == pv.get_root());
}

TEST_CASE("ParentVector::calc_breadth_first_traversal", "[ParentVector]") {
  PV pv = build_test_tree();

  auto traversal = pv.calc_breadth_first_traversal();
  REQUIRE(traversal[0] == 4);
  REQUIRE(traversal[1] == 2);
  REQUIRE(traversal[2] == 3);
  REQUIRE(traversal[3] == 0);
  REQUIRE(traversal[4] == 1);
}

TEST_CASE("ParentVector::swap_subtrees", "[ParentVector]") {
  /*
   * Original tree:
   *
   *   ┌-7-┐
   *  ┌5┐ ┌6
   * ┌2┐3 4
   * 0 1
   */
  PV pv = PV::from_pruefer_code({2, 2, 5, 5, 6, 7});

  REQUIRE(pv[0] == 2);
  REQUIRE(pv[1] == 2);
  REQUIRE(pv[2] == 5);
  REQUIRE(pv[3] == 5);
  REQUIRE(pv[4] == 6);
  REQUIRE(pv[5] == pv.get_root());
  REQUIRE(pv[6] == pv.get_root());
  REQUIRE(pv[7] == pv.get_root());

  /*
   * Resulting tree:
   *
   *   ┌-7-┐
   *  ┌5┐ ┌2┐
   * ┌6 3 0 1
   * 4
   */
  pv.swap_subtrees(2, 6);

  REQUIRE(pv[0] == 2);
  REQUIRE(pv[1] == 2);
  REQUIRE(pv[2] == pv.get_root());
  REQUIRE(pv[3] == 5);
  REQUIRE(pv[4] == 6);
  REQUIRE(pv[5] == pv.get_root());
  REQUIRE(pv[6] == 5);
  REQUIRE(pv[7] == pv.get_root());
}
