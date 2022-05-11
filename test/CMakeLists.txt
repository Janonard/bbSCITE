Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.0.0-preview5
)

FetchContent_MakeAvailable(Catch2)

find_package(Boost 1.76.0 REQUIRED)

set(UNIT_TEST_SOURCES AncestorMatrix.cpp ChangeProposer.cpp ParentVector.cpp)

add_executable(unit_test ${UNIT_TEST_SOURCES})

target_link_libraries(unit_test PRIVATE Catch2::Catch2WithMain)
target_include_directories(unit_test PUBLIC ../src)
set_property(TARGET unit_test PROPERTY CXX_STANDARD 20)
