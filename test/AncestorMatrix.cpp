#include <AncestorMatrix.hpp>
#include <catch2/catch_all.hpp>
using PV = ffSCITE::ParentVector<6>;
using AM = ffSCITE::AncestorMatrix<6>;

AM create_test_ancestor_matrix() {
  // Construct a simple, binary tree with three levels and 7 nodes (6 without
  // root):
  //
  //  ┌-6-┐
  // ┌4┐ ┌5┐
  // 0 1 2 3
  PV pv(6);
  pv.from_pruefer_code({4, 4, 5, 5, 6});
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

TEST_CASE("AncestorMatrix::get_descendants", "[AncestorMatrix]") {
  AM ancestor_matrix = create_test_ancestor_matrix();

  auto descendant = ancestor_matrix.get_descendants(0);
  REQUIRE(descendant[0]);
  REQUIRE(!descendant[1]);
  REQUIRE(!descendant[2]);
  REQUIRE(!descendant[3]);
  REQUIRE(!descendant[4]);
  REQUIRE(!descendant[5]);

  descendant = ancestor_matrix.get_descendants(1);
  REQUIRE(!descendant[0]);
  REQUIRE(descendant[1]);
  REQUIRE(!descendant[2]);
  REQUIRE(!descendant[3]);
  REQUIRE(!descendant[4]);
  REQUIRE(!descendant[5]);

  descendant = ancestor_matrix.get_descendants(2);
  REQUIRE(!descendant[0]);
  REQUIRE(!descendant[1]);
  REQUIRE(descendant[2]);
  REQUIRE(!descendant[3]);
  REQUIRE(!descendant[4]);
  REQUIRE(!descendant[5]);

  descendant = ancestor_matrix.get_descendants(3);
  REQUIRE(!descendant[0]);
  REQUIRE(!descendant[1]);
  REQUIRE(!descendant[2]);
  REQUIRE(descendant[3]);
  REQUIRE(!descendant[4]);
  REQUIRE(!descendant[5]);

  descendant = ancestor_matrix.get_descendants(4);
  REQUIRE(descendant[0]);
  REQUIRE(descendant[1]);
  REQUIRE(!descendant[2]);
  REQUIRE(!descendant[3]);
  REQUIRE(descendant[4]);
  REQUIRE(!descendant[5]);

  descendant = ancestor_matrix.get_descendants(5);
  REQUIRE(!descendant[0]);
  REQUIRE(!descendant[1]);
  REQUIRE(descendant[2]);
  REQUIRE(descendant[3]);
  REQUIRE(!descendant[4]);
  REQUIRE(descendant[5]);

  descendant = ancestor_matrix.get_descendants(6);
  REQUIRE(descendant[0]);
  REQUIRE(descendant[1]);
  REQUIRE(descendant[2]);
  REQUIRE(descendant[3]);
  REQUIRE(descendant[4]);
  REQUIRE(descendant[5]);
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