#pragma once
#include <bit>
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
  /**
   * \brief The width of the index type for the cells.
   */
  static constexpr uint64_t n_cell_bits = std::bit_width(width * height);
  /**
   * \brief The index type to address the cells in the matrix.
   */
  using uindex_t = ac_int<n_cell_bits, false>;

  StaticMatrix() : internal() {}
  ~StaticMatrix() {}

  StaticMatrix(T value) : internal() {
    for (uindex_t c = 0; c < width; c++) {
      for (uindex_t r = 0; r < height; r++) {
        internal[c][r] = value;
      }
    }
  }

  StaticMatrix(StaticMatrix<T, width, height> const &other) = default;
  StaticMatrix &operator=(StaticMatrix<T, width, height> const &other) = default;

  T const &operator[](std::tuple<uindex_t, uindex_t> idx) const {
    return internal[std::get<0>(idx)][std::get<1>(idx)];
  }

  T &operator[](std::tuple<uindex_t, uindex_t> idx) {
    return internal[std::get<0>(idx)][std::get<1>(idx)];
  }

  StaticMatrix<T, width, height> &
  operator+=(StaticMatrix<T, width, height> rhs) {
    for (uindex_t c = 0; c < width; c++) {
      for (uindex_t r = 0; r < height; r++) {
        internal[c][r] += rhs.internal[c][r];
      }
    }
    return *this;
  }

  StaticMatrix < T, width, height> &operator*=(T scalar) {
    for (uindex_t c = 0; c < width; c++) {
      for (uindex_t r = 0; r < height; r++) {
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