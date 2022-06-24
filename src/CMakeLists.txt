set(FFSCITE_SOURCES main.cpp)

add_executable(ffSCITE_emu ${FFSCITE_SOURCES})
add_executable(ffSCITE_report ${FFSCITE_SOURCES})
add_executable(ffSCITE ${FFSCITE_SOURCES})

# Common options
set_property(TARGET ffSCITE_emu ffSCITE_report ffSCITE APPEND PROPERTY COMPILE_OPTIONS -fintelfpga)
set_property(TARGET ffSCITE_emu ffSCITE_report ffSCITE APPEND PROPERTY LINK_OPTIONS -fintelfpga -Xsv)

# Specific options
set_property(TARGET ffSCITE_emu APPEND PROPERTY LINK_OPTIONS -DEMULATOR)
set_property(TARGET ffSCITE_report APPEND PROPERTY LINK_OPTIONS -Xshardware -fsycl-link)
set_property(TARGET ffSCITE APPEND PROPERTY LINK_OPTIONS -Xshardware)

if(DEFINED ENV{FPGA_BOARD_NAME})
  set_property(TARGET ffSCITE_report ffSCITE APPEND PROPERTY LINK_OPTIONS -Xsboard=$ENV{FPGA_BOARD_NAME} -Xsboard-package=$ENV{AOCL_BOARD_PACKAGE_ROOT})
endif()
