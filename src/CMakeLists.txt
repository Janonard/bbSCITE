set(FFSCITE_SOURCES main.cpp)

add_executable(ffSCITE_emu ${FFSCITE_SOURCES})
add_executable(ffSCITE_report ${FFSCITE_SOURCES})
add_executable(ffSCITE ${FFSCITE_SOURCES})

# Common options
set_property(TARGET ffSCITE_emu ffSCITE_report ffSCITE APPEND PROPERTY COMPILE_OPTIONS -qactypes -fintelfpga)
set_property(TARGET ffSCITE_emu ffSCITE_report ffSCITE APPEND PROPERTY LINK_OPTIONS -fintelfpga -Xsv)
set_property(TARGET ffSCITE_emu ffSCITE_report ffSCITE PROPERTY CXX_STANDARD 20)

# Specific options
set_property(TARGET ffSCITE_emu APPEND PROPERTY LINK_OPTIONS)
target_compile_definitions(ffSCITE_emu PUBLIC EMULATOR)
set_property(TARGET ffSCITE ffSCITE_report APPEND PROPERTY LINK_OPTIONS -Xshardware -Xsclock=300MHz -Xsseed=2)
set_property(TARGET ffSCITE_report APPEND PROPERTY LINK_OPTIONS -fsycl-link)

if(DEFINED ENV{FPGA_BOARD_NAME})
  set_property(TARGET ffSCITE_report ffSCITE APPEND PROPERTY LINK_OPTIONS -Xsboard=$ENV{FPGA_BOARD_NAME} -Xsboard-package=$ENV{AOCL_BOARD_PACKAGE_ROOT})
endif()
