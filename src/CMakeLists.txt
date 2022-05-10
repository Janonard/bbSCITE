set(FFSCITE_SOURCES main.cpp)
set(FFSCITE_HEADERS ChainState.hpp ChangeProposer.hpp ParentVector.hpp AncestorMatrix.hpp)

add_executable(ffSCITE ${FFSCITE_SOURCES} ${FFSCITE_HEADERS})

set_property(TARGET ffSCITE PROPERTY CXX_STANDARD 20)
