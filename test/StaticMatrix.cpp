#include "StaticMatrix.hpp"
#include <catch2/catch_all.hpp>

TEST_CASE("StaticMatrix::operator[]", "[StaticMatrix]") {
  ffSCITE::StaticMatrix<uint64_t, 4, 4> matrix(0);

  for (uint64_t c = 0; c < 4; c++) {
    for (uint64_t r = 0; r < 4; r++) {
      REQUIRE(matrix[{c, r}] == 0);
      matrix[{c, r}] = c * 4 + r;
    }
  }

  for (uint64_t r = 0; r < 4; r++) {
    for (uint64_t c = 0; c < 4; c++) {
      REQUIRE(matrix[{c, r}] == c * 4 + r);
    }
  }
}

TEST_CASE("StaticMatrix::operator+", "[StaticMatrix]") {
  ffSCITE::StaticMatrix<uint64_t, 4, 4> matrix_a(0);
  ffSCITE::StaticMatrix<uint64_t, 4, 4> matrix_b(0);
  ffSCITE::StaticMatrix<uint64_t, 4, 4> matrix_c(4);

  for (uint64_t c = 0; c < 4; c++) {
    for (uint64_t r = 0; r < 4; r++) {
      matrix_a[{c, r}] = c * 4;
      matrix_b[{c, r}] = r;
    }
  }

  matrix_c += matrix_a + matrix_b;

  for (uint64_t c = 0; c < 4; c++) {
    for (uint64_t r = 0; r < 4; r++) {
      REQUIRE(matrix_a[{c, r}] == c * 4);
      REQUIRE(matrix_b[{c, r}] == r);
      REQUIRE(matrix_c[{c, r}] == 4 + c * 4 + r);
    }
  }
}

TEST_CASE("StaticMatrix::operator*", "[StaticMatrix]") {
  ffSCITE::StaticMatrix<uint64_t, 4, 4> matrix_a(0);
  ffSCITE::StaticMatrix<uint64_t, 4, 4> matrix_b(0);

  for (uint64_t c = 0; c < 4; c++) {
    for (uint64_t r = 0; r < 4; r++) {
      matrix_a[{c, r}] = c * 4 + r;
    }
  }

  matrix_b = uint64_t(8) * matrix_a;

  for (uint64_t c = 0; c < 4; c++) {
    for (uint64_t r = 0; r < 4; r++) {
      REQUIRE(matrix_a[{c, r}] == c * 4 + r);
      REQUIRE(matrix_b[{c, r}] == 8 * (c * 4 + r));
    }
  }
}