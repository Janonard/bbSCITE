#pragma once
#include "ChangeProposer.hpp"
#include "StateScorer.hpp"
#include <CL/sycl.hpp>

namespace ffSCITE {
template <uint64_t max_n_cells, uint64_t max_n_genes, typename ChangeProposer,
          typename StateScorer,
          cl::sycl::access::target access_target =
              cl::sycl::access::target::device>
class MCMCKernel {
public:
  using ChainStateImpl = ChainState<max_n_genes>;
  using MutationDataMatrix =
      StaticMatrix<ac_int<2, false>, max_n_cells, max_n_genes>;
  using Accessor =
      cl::sycl::accessor<std::tuple<ChainStateImpl, double>, 1,
                         cl::sycl::access::mode::read_write, access_target>;

  MCMCKernel(ChangeProposer change_proposer, StateScorer state_scorer,
             double gamma, Accessor current_state_ac, Accessor best_state_ac,
             uint64_t n_steps)
      : change_proposer(change_proposer), state_scorer(state_scorer),
        current_state_ac(current_state_ac), best_state_ac(best_state_ac),
        gamma(gamma), n_steps(n_steps) {
    assert(current_state_ac.get_range() == 1);
    assert(best_state_ac.get_range() == 1);
  }

  void operator()() const {
    ChainStateImpl best_state = std::get<0>(best_state_ac[0]);
    double best_score = std::get<1>(best_state_ac[0]);

    ChainStateImpl current_state = std::get<0>(current_state_ac[0]);
    double current_score = std::get<1>(best_state_ac[0]);

    // Copy of score to avoid mutation within the constant operator.
    ChangeProposer change_proposer = this->change_proposer;
    StateScorer state_scorer = this->state_scorer;

    for (uint64_t i = 0; i < n_steps; i++) {
      double neighborhood_correction = 1.0;

      ChainStateImpl proposed_state = current_state;
      change_proposer.propose_change(proposed_state, neighborhood_correction);
      double proposed_score = state_scorer.logscore_state(proposed_state);

      double acceptance_probability =
          neighborhood_correction *
          std::exp((proposed_score - current_score) * gamma);
      bool accept_move = oneapi::dpl::bernoulli_distribution(
          acceptance_probability)(change_proposer.get_rng());
      if (accept_move) {
        current_state = proposed_state;
        current_score = proposed_score;
      }

      if (proposed_score > best_score) {
        best_state = proposed_state;
        best_score = proposed_score;
      }
    }

    best_state_ac[0] = {best_state, best_score};
    current_state_ac[0] = {current_state, current_score};
  }

  ChangeProposer &get_change_proposer() { return change_proposer; }

  StateScorer &get_state_scorer() { return state_scorer; }

private:
  ChangeProposer change_proposer;
  StateScorer state_scorer;
  Accessor best_state_ac, current_state_ac;

  double gamma;
  uint64_t n_steps;
};
} // namespace ffSCITE