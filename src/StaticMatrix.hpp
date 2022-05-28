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
#pragma once
#include <cstdint>
#include <tuple>
#include <utility>

namespace ffSCITE {
/**
 * \brief An abstract matrix width static sizes and some vector operations.
 *
 * \tparam T The type of the entries
 * \tparam width The width of the matrix
 * \tparam height The height of the matrix
 */
template <typename T, uint64_t width, uint64_t height> class StaticMatrix {
public:
  StaticMatrix() : internal() {}
  ~StaticMatrix() {}

  StaticMatrix(T value) : internal() {
    for (uint64_t c = 0; c < width; c++) {
      for (uint64_t r = 0; r < height; r++) {
        internal[c][r] = value;
      }
    }
  }

  StaticMatrix(StaticMatrix<T, width, height> const &other) = default;
  StaticMatrix &
  operator=(StaticMatrix<T, width, height> const &other) = default;

  T const &operator[](std::tuple<uint64_t, uint64_t> idx) const {
    return internal[std::get<0>(idx)][std::get<1>(idx)];
  }

  T &operator[](std::tuple<uint64_t, uint64_t> idx) {
    return internal[std::get<0>(idx)][std::get<1>(idx)];
  }

  StaticMatrix<T, width, height> &
  operator+=(StaticMatrix<T, width, height> rhs) {
    for (uint64_t c = 0; c < width; c++) {
      for (uint64_t r = 0; r < height; r++) {
        internal[c][r] += rhs.internal[c][r];
      }
    }
    return *this;
  }

  StaticMatrix<T, width, height> &operator*=(T scalar) {
    for (uint64_t c = 0; c < width; c++) {
      for (uint64_t r = 0; r < height; r++) {
        internal[c][r] *= scalar;
      }
    }
    return *this;
  }

private:
  T internal[width][height];
};

template <typename T, uint64_t width, uint64_t height>
StaticMatrix<T, width, height> operator+(StaticMatrix<T, width, height> lhs,
                                         StaticMatrix<T, width, height> rhs) {
  lhs += rhs;
  return lhs;
}

template <typename T, uint64_t width, uint64_t height>
StaticMatrix<T, width, height> operator*(StaticMatrix<T, width, height> lhs,
                                         T scalar) {
  lhs *= scalar;
  return lhs;
}

template <typename T, uint64_t width, uint64_t height>
StaticMatrix<T, width, height> operator*(T scalar,
                                         StaticMatrix<T, width, height> rhs) {
  rhs *= scalar;
  return rhs;
}
} // namespace ffSCITE