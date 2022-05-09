#include <ParentVector.hpp>
#include <catch2/catch_all.hpp>

using PV = ffSCITE::ParentVector<16>;

PV build_test_tree() {
  /*
   * Initial tree:
   *
   * ┌-┌r┐-┐
   * 0 1 2 3
   */
  PV pv(4);
  REQUIRE(pv[0] == PV::root);
  REQUIRE(pv[1] == PV::root);
  REQUIRE(pv[2] == PV::root);
  REQUIRE(pv[3] == PV::root);

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
  REQUIRE(pv[2] == PV::root);
  REQUIRE(pv[3] == PV::root);

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
  REQUIRE(pv[2] == PV::root);
  REQUIRE(pv[3] == PV::root);

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
  REQUIRE(pv[2] == PV::root);
  REQUIRE(pv[3] == PV::root);

  // Swap of parent and child
  pv.swap_nodes(0, 3);

  /*
   * Expected tree:
   *
   *  ┌r┐
   * ┌0┐2
   * 3 1
   */
  REQUIRE(pv[0] == PV::root);
  REQUIRE(pv[1] == 0);
  REQUIRE(pv[2] == PV::root);
  REQUIRE(pv[3] == 0);
}
