set(FFSCITE_SOURCES main.cpp)
set(FFSCITE_HEADERS ChainState.hpp ChangeProposer.hpp ParentVector.hpp)

add_executable(ffSCITE ${FFSCITE_SOURCES} ${FFSCITE_HEADER})

set_property(TARGET ffSCITE PROPERTY CXX_STANDARD 20)
