#include "ChainState.hpp"
#include <random>

namespace ffSCITE {
template <uint64_t max_n_nodes, typename RNG> class ChangeProposer {
public:
  using uindex_node_t = typename ParentVector<max_n_nodes>::uindex_node_t;

  ChangeProposer(RNG rng, double prob_beta_change, double prob_prune_n_reattach,
                 double prob_swap_nodes)
      : rng(rng), prob_beta_change(prob_beta_change),
        prob_prune_n_reattach(prob_prune_n_reattach),
        prob_swap_nodes(prob_swap_nodes) {
#if __SYCL_DEVICE_ONLY__ == 0
    assert(prob_beta_change + prob_prune_n_reattach + prob_swap_nodes < 1.0);
#endif
  }

  enum class MoveType {
    ChangeBeta,
    PruneReattach,
    SwapNodes,
    SwapSubtrees,
  };

  MoveType sample_move() {
    double change_type_draw = std::uniform_real_distribution(0.0, 1.0)(rng);
    if (change_type_draw <= prob_beta_change) {
      return MoveType::ChangeBeta;
    } else if (change_type_draw <= prob_beta_change + prob_prune_n_reattach) {
      return MoveType::PruneReattach;
    } else if (change_type_draw <=
               prob_beta_change + prob_prune_n_reattach + prob_swap_nodes) {
      return MoveType::SwapNodes;
    } else {
      return MoveType::SwapSubtrees;
    }
  }

  void propose_change(ChainState<max_n_nodes> &state) {
    switch (sample_move()) {
    case MoveType::ChangeBeta:
      std::cout << "Change Beta" << std::endl;
      break;
    case MoveType::PruneReattach:
      std::cout << "Prune and Reattach" << std::endl;
      break;
    case MoveType::SwapNodes:
      std::cout << "Swap Nodes" << std::endl;
      break;
    case MoveType::SwapSubtrees:
      std::cout << "Swap Subtrees" << std::endl;
    default:
      break;
    }
  }

private:
  RNG rng;
  double prob_beta_change, prob_prune_n_reattach,
      prob_swap_nodes; // prob_swap_subtrees omitted.
};
} // namespace ffSCITE