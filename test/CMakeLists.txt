Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.0.1
)

FetchContent_MakeAvailable(Catch2)

set(UNIT_TEST_SOURCES Application.cpp MutationTree.cpp TreeScorer.cpp)

add_executable(unit_test ${UNIT_TEST_SOURCES})

target_compile_definitions(unit_test PUBLIC EMULATOR)
target_link_libraries(unit_test PRIVATE Catch2::Catch2WithMain)
target_include_directories(unit_test PUBLIC ../src)

set_property(TARGET unit_test PROPERTY CXX_STANDARD 20)
set_property(TARGET unit_test PROPERTY COMPILE_OPTIONS -qactypes -fintelfpga -Xsv)
set_property(TARGET unit_test PROPERTY LINK_OPTIONS -qactypes -fintelfpga -Xsv)