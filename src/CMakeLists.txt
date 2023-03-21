set(FFSCITE_SOURCES main.cpp)

foreach(vectorlen 64 96 128)
  set(base_target ffSCITE${vectorlen})
  set(emu_target ffSCITE${vectorlen}_emu)
  set(report_target ffSCITE${vectorlen}_report)
  set(all_targets ${base_target} ${emu_target} ${report_target})
  set(hw_targets ${base_target} ${report_target})

  foreach(target ${all_targets})
    add_executable(${target} ${FFSCITE_SOURCES})
  endforeach()

  # Common options
  set_property(TARGET ${all_targets} APPEND PROPERTY COMPILE_OPTIONS -qactypes -fintelfpga)
  set_property(TARGET ${all_targets} APPEND PROPERTY COMPILE_DEFINITIONS FFSCITE_VECTORLEN=${vectorlen})
  set_property(TARGET ${all_targets} APPEND PROPERTY LINK_OPTIONS -fintelfpga -Xsv)
  set_property(TARGET ${all_targets} PROPERTY CXX_STANDARD 20)

  # Specific options
  target_compile_definitions(${emu_target} PUBLIC EMULATOR)
  target_compile_definitions(${base_target} PUBLIC HARDWARE)
  set_property(TARGET ${hw_targets} APPEND PROPERTY LINK_OPTIONS -Xshardware)
  set_property(TARGET ${report_target} APPEND PROPERTY LINK_OPTIONS -fsycl-link)

  if(DEFINED ENV{FPGA_BOARD_NAME})
    set_property(TARGET ${hw_targets} APPEND PROPERTY LINK_OPTIONS -Xsboard=$ENV{FPGA_BOARD_NAME} -Xsboard-package=$ENV{AOCL_BOARD_PACKAGE_ROOT})
  endif()

  if(${vectorlen} EQUAL 128)
    set_property(TARGET ${hw_targets} APPEND PROPERTY LINK_OPTIONS -Xsclock=200MHz)
  endif()
endforeach()
