#include "ChangeProposer.hpp"

using namespace ffSCITE;

int main() {
    std::mt19937 twister;
    twister.seed(42);
    std::cout << twister() << std::endl;
    ChangeProposer<15, std::mt19937> proposer(twister, 0.55, 0.4, 0.04);
    ChainState<15> state(twister, 15, 0.5);

    for (uint64_t i = 0; i < 128; i++) {
        proposer.propose_change(state);
    }

    return 0;
}