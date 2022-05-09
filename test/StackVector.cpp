#include <StackVector.hpp>
#include <catch2/catch_all.hpp>

using namespace ffSCITE;

TEST_CASE("StackVector: pushing and popping", "[StackVector]") {
  StackVector<int, 16> vector;

  REQUIRE(vector.get_n_elements() == 0);
  for (int i = 0; i < 16; i++) {
    vector.push_back(i);
    REQUIRE(vector[i] == i);
    REQUIRE(vector.get_n_elements() == i + 1);
  }

  for (int i = 0; i < 16; i++) {
    vector[i] = 16 - vector[i] - 1;
  }

  for (int i = 0; i < 16; i++) {
    REQUIRE(vector[i] == 16 - i - 1);
  }

  for (int i = 0; i < 16; i++) {
    int value = vector.pop_back();
    REQUIRE(value == i);
    REQUIRE(vector.get_n_elements() == 16 - i - 1);
  }
}