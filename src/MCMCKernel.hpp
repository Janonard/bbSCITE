#pragma once
#include "ChangeProposer.hpp"
#include "StateScorer.hpp"
#include <CL/sycl.hpp>

namespace ffSCITE {
/**
 * \brief SYCL kernel that simulates a SCITE markov chain.
 *
 * It picks up a chain with a given current and best states, simulates the chain
 * from this point for a set number of iterations and writes the resulting state
 * back. It takes a change proposer and a state scorer as it's type argument so
 * that they can easily be swapped out to test different approaches.
 *
 * \tparam max_n_cells The maximum number of cells supported by the kernel.
 * \tparam max_n_genes The maximum number of genes supported by the kernel.
 * \tparam ChangeProposer The type of the change proposing strategy.
 * \tparam StateScorer The type of the state scoring strategy.
 * \tparam access_target The target from where the states are accessed.
 */
template <uint64_t max_n_cells, uint64_t max_n_genes, typename ChangeProposer,
          typename StateScorer,
          cl::sycl::access::target access_target =
              cl::sycl::access::target::device>
class MCMCKernel {
public:
  /**
   * \brief Shorthand for the used chain state.
   */
  using ChainStateImpl = ChainState<max_n_genes>;
  /**
   * \brief Shorthand for the mutation data matrix type.
   */
  using MutationDataMatrix =
      StaticMatrix<ac_int<2, false>, max_n_cells, max_n_genes>;
  /**
   * \brief Shorthand for the chain state buffer accessor type.
   */
  using Accessor =
      cl::sycl::accessor<std::tuple<ChainStateImpl, double>, 1,
                         cl::sycl::access::mode::read_write, access_target>;

  /**
   * \brief Construct a new MCMCKernel object
   *
   * \param change_proposer The configured instance of the change proposer to
   * use.
   * \param state_scorer The configured instance of the state scorer to
   * use.
   * \param gamma A factor for the width of the optimization space
   * exploration.
   * \param current_state_ac Accessor to the current state of the
   * chain.
   * \param best_state_ac Accessor to the best known state of the chain.
   * \param n_steps The number of steps to execute.
   */
  MCMCKernel(ChangeProposer change_proposer, StateScorer state_scorer,
             double gamma, Accessor current_state_ac, Accessor best_state_ac,
             uint64_t n_steps)
      : change_proposer(change_proposer), state_scorer(state_scorer),
        current_state_ac(current_state_ac), best_state_ac(best_state_ac),
        gamma(gamma), n_steps(n_steps) {
    assert(current_state_ac.get_range() == 1);
    assert(best_state_ac.get_range() == 1);
  }

  /**
   * @brief Run the MCMC chain.
   *
   * This will read the current and best states from device memory, execute the
   * configured number of steps and writes the results back.
   */
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

  /**
   * @brief Get the change proposer instance.
   *
   * @return ChangeProposer& A reference to the change proposer.
   */
  ChangeProposer &get_change_proposer() { return change_proposer; }

  /**
   * @brief Get the state scorer instance.
   *
   * @return StateScorer& A reference to the state scorer.
   */
  StateScorer &get_state_scorer() { return state_scorer; }

private:
  ChangeProposer change_proposer;
  StateScorer state_scorer;
  Accessor best_state_ac, current_state_ac;

  double gamma;
  uint64_t n_steps;
};
} // namespace ffSCITE