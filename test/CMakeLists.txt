Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.0.0-preview3
)

FetchContent_MakeAvailable(Catch2)

set(TEST_SOURCES AncestorMatrix.cpp ParentVector.cpp)

add_executable(test ${TEST_SOURCES})

target_link_libraries(test PRIVATE Catch2::Catch2WithMain)
target_include_directories(test PUBLIC ../src)
set_property(TARGET test PROPERTY CXX_STANDARD 20)
