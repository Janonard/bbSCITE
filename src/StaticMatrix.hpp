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
 * @brief An abstract matrix width static sizes and some vector operations.
 *
 * @tparam T The type of the entries
 * @tparam width The width of the matrix
 * @tparam height The height of the matrix
 */
template <typename T, uint32_t width, uint32_t height> class StaticMatrix {
public:
  StaticMatrix() : internal() {}
  ~StaticMatrix() {}

  /**
   * @brief Construct a new Static Matrix and initialize it with the given
   * value.
   *
   * This method is designed and optimized for FPGAs, but works on CPUs too.
   *
   * @param value The initial value of all entries.
   */
  StaticMatrix(T value) : internal() {
#pragma unroll
    for (uint32_t c = 0; c < width; c++) {
#pragma unroll
      for (uint32_t r = 0; r < height; r++) {
        internal[c][r] = value;
      }
    }
  }

  StaticMatrix(StaticMatrix<T, width, height> const &other) = default;
  StaticMatrix &
  operator=(StaticMatrix<T, width, height> const &other) = default;

  /**
   * @brief Access an element in the matrix.
   *
   * It is assumed that the index is within the bounds of this matrix.
   *
   * This method works equally well on CPUs and on FPGAs.
   *
   * @param idx The two-dimensional index of the query.
   * @return T const& A reference to the queried element.
   */
  T const &operator[](std::tuple<uint32_t, uint32_t> idx) const {
    return internal[std::get<0>(idx)][std::get<1>(idx)];
  }

  /**
   * @brief Access an element in the matrix.
   *
   * It is assumed that the index is within the bounds of this matrix.
   *
   * This method works equally well on CPUs and on FPGAs.
   *
   * @param idx The two-dimensional index of the query.
   * @return T & A reference to the queried element.
   */
  T &operator[](std::tuple<uint32_t, uint32_t> idx) {
    return internal[std::get<0>(idx)][std::get<1>(idx)];
  }

  /**
   * @brief Vector-add another matrix to this matrix.
   *
   * It is assumed that the add-assign operation for entries exists and is
   * well-defined.
   *
   * This method is designed and optimized for FPGAs, but works on CPUs too.
   *
   * @param rhs The other matrix to add.
   * @return StaticMatrix<T, width, height>& A reference to this, resulting
   * matrix.
   */
  StaticMatrix<T, width, height> &
  operator+=(StaticMatrix<T, width, height> rhs) {
#pragma unroll
    for (uint32_t c = 0; c < width; c++) {
#pragma unroll
      for (uint32_t r = 0; r < height; r++) {
        internal[c][r] += rhs.internal[c][r];
      }
    }
    return *this;
  }

  /**
   * @brief Multiply the given value to the entries of this matrix.
   *
   * It is assumed that the multiply-assign operation for entries exists and is
   * well-defined.
   *
   * This method is designed and optimized for FPGAs, but works on CPUs too.
   *
   * @param scalar The scalar to multiply with all entries.
   * @return StaticMatrix<T, width, height>& A reference to this, resulting
   * matrix.
   */
  StaticMatrix<T, width, height> &operator*=(T scalar) {
#pragma unroll
    for (uint32_t c = 0; c < width; c++) {
#pragma unroll
      for (uint32_t r = 0; r < height; r++) {
        internal[c][r] *= scalar;
      }
    }
    return *this;
  }

private:
  T internal[width][height];
};

template <typename T, uint32_t width, uint32_t height>
StaticMatrix<T, width, height> operator+(StaticMatrix<T, width, height> lhs,
                                         StaticMatrix<T, width, height> rhs) {
  lhs += rhs;
  return lhs;
}

template <typename T, uint32_t width, uint32_t height>
StaticMatrix<T, width, height> operator*(StaticMatrix<T, width, height> lhs,
                                         T scalar) {
  lhs *= scalar;
  return lhs;
}

template <typename T, uint32_t width, uint32_t height>
StaticMatrix<T, width, height> operator*(T scalar,
                                         StaticMatrix<T, width, height> rhs) {
  rhs *= scalar;
  return rhs;
}
} // namespace ffSCITE