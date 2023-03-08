set(CPU_SCORING_BENCHMARK_SOURCES main.cpp)

add_executable(cpu_scoring_benchmark ${CPU_SCORING_BENCHMARK_SOURCES})
target_include_directories(cpu_scoring_benchmark PUBLIC ../src)
set_property(TARGET cpu_scoring_benchmark PROPERTY CXX_STANDARD 20)

target_compile_options(cpu_scoring_benchmark PUBLIC -qactypes)

if(DEFINED PC2_SYSTEM)
    target_compile_options(cpu_scoring_benchmark PUBLIC -fsycl-targets=spir64_x86_64-unknown-unknown)
    target_link_options(cpu_scoring_benchmark PUBLIC -fsycl-targets=spir64_x86_64-unknown-unknown -Xsdevice=cpu)

    if(${PC2_SYSTEM} STREQUAL "Noctua2")
        target_link_options(cpu_scoring_benchmark PUBLIC -Xsmarch=avx2)
        target_compile_definitions(cpu_scoring_benchmark PUBLIC PC2_SYSTEM=2)
    elseif(${PC2_SYSTEM} STREQUAL "Noctua1")
        target_link_options(cpu_scoring_benchmark PUBLIC -Xsmarch=avx512)
        target_compile_definitions(cpu_scoring_benchmark PUBLIC PC2_SYSTEM=1)
    endif()

    add_custom_command(
        OUTPUT cpu_scoring_offload.asm
        COMMAND objcopy ARGS -j __CLANG_OFFLOAD_BUNDLE__sycl-spir64_x86_64  $<TARGET_FILE:cpu_scoring_benchmark> cpu_scoring_offload.efi
        COMMAND objdump ARGS -D -M intel-mnemonics cpu_scoring_offload.efi > cpu_scoring_offload.asm
        DEPENDS cpu_scoring_benchmark)
    add_custom_target(cpu_scoring_offload DEPENDS cpu_scoring_offload.asm)
endif()