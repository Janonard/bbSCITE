#include <ParentVector.hpp>
#include <catch2/catch_all.hpp>

using PV = ffSCITE::ParentVector<16>;

TEST_CASE("ParentVector: Elementwise access", "[ParentVector]") {
  PV pv(16);
  for (uint64_t i = 0; i < 16; i++) {
    pv[i] = i + 1;
  }

  for (uint64_t i = 0; i < 16; i++) {
    REQUIRE(pv[i] == i + 1);
  }
}

PV build_test_tree() {
  /*
   * Build the following tree:
   *
   *  ┌3┐
   * ┌2┐4
   * 0 1
   */
  PV pv(5);
  pv[0] = 2;
  pv[1] = 2;
  pv[2] = 3;
  pv[4] = 3;
  pv[3] = 5;
  return pv;
}

TEST_CASE("ParentVector: Ancestry Queries", "[ParentVector]") {
  PV pv = build_test_tree();

  for (uint64_t i = 0; i < 5; i++) {
    if (i == 3) {
      REQUIRE(pv.is_root(i));
    } else {
      REQUIRE(!pv.is_root(i));
    }
  }

  REQUIRE(pv.is_descendant(0, 0));
  REQUIRE(!pv.is_descendant(0, 1));
  REQUIRE(pv.is_descendant(0, 2));
  REQUIRE(pv.is_descendant(0, 3));
  REQUIRE(!pv.is_descendant(0, 4));

  REQUIRE(!pv.is_descendant(1, 0));
  REQUIRE(pv.is_descendant(1, 1));
  REQUIRE(pv.is_descendant(1, 2));
  REQUIRE(pv.is_descendant(1, 3));
  REQUIRE(!pv.is_descendant(1, 4));

  REQUIRE(!pv.is_descendant(2, 0));
  REQUIRE(!pv.is_descendant(2, 1));
  REQUIRE(pv.is_descendant(2, 2));
  REQUIRE(pv.is_descendant(2, 3));
  REQUIRE(!pv.is_descendant(2, 4));

  REQUIRE(!pv.is_descendant(3, 0));
  REQUIRE(!pv.is_descendant(3, 1));
  REQUIRE(!pv.is_descendant(3, 2));
  REQUIRE(pv.is_descendant(3, 3));
  REQUIRE(!pv.is_descendant(3, 4));

  REQUIRE(!pv.is_descendant(4, 0));
  REQUIRE(!pv.is_descendant(4, 1));
  REQUIRE(!pv.is_descendant(4, 2));
  REQUIRE(pv.is_descendant(4, 3));
  REQUIRE(pv.is_descendant(4, 4));
}

TEST_CASE("ParentVector::swap_nodes", "[ParentVector]") {
  PV pv = build_test_tree();

  // Identity operation
  pv.swap_nodes(3, 3);

  /*
   * Expected tree:
   *
   *  ┌3┐
   * ┌2┐4
   * 0 1
   */
  REQUIRE(pv[0] == 2);
  REQUIRE(pv[1] == 2);
  REQUIRE(pv[2] == 3);
  REQUIRE(pv[3] >= 5);
  REQUIRE(pv[4] == 3);

  // Swap of unrelated nodes
  pv.swap_nodes(2, 4);

  /*
   * Expected tree:
   *
   *  ┌3┐
   * ┌4┐2
   * 0 1
   */
  REQUIRE(pv[0] == 4);
  REQUIRE(pv[1] == 4);
  REQUIRE(pv[2] == 3);
  REQUIRE(pv[3] >= 5);
  REQUIRE(pv[4] == 3);

  // Swap of parent and child
  pv.swap_nodes(0, 4);

  /*
   * Expected tree:
   *
   *  ┌3┐
   * ┌0┐2
   * 4 1
   */
  REQUIRE(pv[0] == 3);
  REQUIRE(pv[1] == 0);
  REQUIRE(pv[2] == 3);
  REQUIRE(pv[3] >= 5);
  REQUIRE(pv[4] == 0);
}