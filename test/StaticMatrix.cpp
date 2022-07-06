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
#include "StaticMatrix.hpp"
#include <catch2/catch_all.hpp>

TEST_CASE("StaticMatrix::operator[]", "[StaticMatrix]") {
  ffSCITE::StaticMatrix<uint32_t, 4, 4> matrix(0);

  for (uint32_t c = 0; c < 4; c++) {
    for (uint32_t r = 0; r < 4; r++) {
      REQUIRE(matrix[{c, r}] == 0);
      matrix[{c, r}] = c * 4 + r;
    }
  }

  for (uint32_t r = 0; r < 4; r++) {
    for (uint32_t c = 0; c < 4; c++) {
      REQUIRE(matrix[{c, r}] == c * 4 + r);
    }
  }
}

TEST_CASE("StaticMatrix::operator+", "[StaticMatrix]") {
  ffSCITE::StaticMatrix<uint32_t, 4, 4> matrix_a(0);
  ffSCITE::StaticMatrix<uint32_t, 4, 4> matrix_b(0);
  ffSCITE::StaticMatrix<uint32_t, 4, 4> matrix_c(4);

  for (uint32_t c = 0; c < 4; c++) {
    for (uint32_t r = 0; r < 4; r++) {
      matrix_a[{c, r}] = c * 4;
      matrix_b[{c, r}] = r;
    }
  }

  matrix_c += matrix_a + matrix_b;

  for (uint32_t c = 0; c < 4; c++) {
    for (uint32_t r = 0; r < 4; r++) {
      REQUIRE(matrix_a[{c, r}] == c * 4);
      REQUIRE(matrix_b[{c, r}] == r);
      REQUIRE(matrix_c[{c, r}] == 4 + c * 4 + r);
    }
  }
}

TEST_CASE("StaticMatrix::operator*", "[StaticMatrix]") {
  ffSCITE::StaticMatrix<uint32_t, 4, 4> matrix_a(0);
  ffSCITE::StaticMatrix<uint32_t, 4, 4> matrix_b(0);

  for (uint32_t c = 0; c < 4; c++) {
    for (uint32_t r = 0; r < 4; r++) {
      matrix_a[{c, r}] = c * 4 + r;
    }
  }

  matrix_b = uint32_t(8) * matrix_a;

  for (uint32_t c = 0; c < 4; c++) {
    for (uint32_t r = 0; r < 4; r++) {
      REQUIRE(matrix_a[{c, r}] == c * 4 + r);
      REQUIRE(matrix_b[{c, r}] == 8 * (c * 4 + r));
    }
  }
}