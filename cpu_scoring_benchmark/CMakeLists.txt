set(CPU_SCORING_BENCHMARK_SOURCES main.cpp)

add_executable(cpu_scoring_benchmark ${CPU_SCORING_BENCHMARK_SOURCES})
target_include_directories(cpu_scoring_benchmark PUBLIC ../src)
set_property(TARGET cpu_scoring_benchmark PROPERTY CXX_STANDARD 20)
set_property(TARGET cpu_scoring_benchmark APPEND PROPERTY COMPILE_OPTIONS -qactypes)