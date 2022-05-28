set(FFSCITE_SOURCES main.cpp)

add_executable(ffSCITE ${FFSCITE_SOURCES})

set_property(TARGET ffSCITE PROPERTY CXX_STANDARD 20)
set_property(TARGET ffSCITE PROPERTY COMPILE_OPTIONS -qactypes -fintelfpga -Xsv)
set_property(TARGET ffSCITE PROPERTY LINK_OPTIONS -qactypes -fintelfpga -Xsv)
