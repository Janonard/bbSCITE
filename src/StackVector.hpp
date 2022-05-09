#include <array>
#include <bit>
#include <cstdint>
#include <sycl/ext/intel/ac_types/ac_int.hpp>

namespace ffSCITE {
template <typename T, uint64_t max_n_elements> class StackVector {
public:
  static constexpr uint64_t n_index_bits = std::bit_width(max_n_elements);
  using uindex_t = ac_int<n_index_bits, false>;

  StackVector() : elements(), n_elements(0) {}

  T const &operator[](uindex_t i) const {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(i < n_elements);
#endif
    return elements[i];
  }

  T &operator[](uindex_t i) {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(i < n_elements);
#endif
    return elements[i];
  }

  uindex_t get_n_elements() const { return n_elements; }

  void push_back(T new_element) {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(n_elements < max_n_elements);
#endif
    elements[n_elements] = new_element;
    n_elements++;
  }

  T const &back() const { return elements[n_elements - 1]; }

  T &back() { return elements[n_elements - 1]; }

  T pop_back() {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(n_elements > 0);
#endif
    n_elements--;
    return elements[n_elements];
  }

private:
  std::array<T, max_n_elements> elements;
  uindex_t n_elements;
};
} // namespace ffSCITE