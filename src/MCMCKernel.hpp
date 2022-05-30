/*
 * Copyright © 2022 Jan-Oliver Opdenhövel
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <https://www.gnu.org/licenses/>.
 */
#pragma once
#include "ChangeProposer.hpp"
#include "Parameters.hpp"
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
   * \brief Shorthand for the chain state buffer accessor type.
   */
  using StateAccessor =
      cl::sycl::accessor<ChainStateImpl, 1, cl::sycl::access::mode::read_write,
                         access_target>;

  using ScoreAccessor =
      cl::sycl::accessor<double, 1, cl::sycl::access::mode::read_write,
                         access_target>;

  using IndexAccessor =
      cl::sycl::accessor<uint64_t, 1, cl::sycl::access::mode::read_write,
                         access_target>;

  /**
   * \brief
   *
   *
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
             double gamma, StateAccessor best_states_ac,
             ScoreAccessor best_score_ac, IndexAccessor n_best_states_ac,
             StateAccessor current_states_ac, ScoreAccessor current_scores_ac,
             uint64_t n_steps)
      : change_proposer(change_proposer), state_scorer(state_scorer),
        best_states_ac(best_states_ac), current_states_ac(current_states_ac),
        best_score_ac(best_score_ac), current_scores_ac(current_scores_ac),
        n_best_states_ac(n_best_states_ac), gamma(gamma), n_steps(n_steps) {
    assert(best_score_ac.get_range()[0] == 1);
    assert(current_states_ac.get_range() == current_scores_ac.get_range());
  }

  /**
   * @brief Run the MCMC chain.
   *
   * This will read the current and best states from device memory, execute the
   * configured number of steps and writes the results back.
   */
  void operator()() const {
    // Copy of score to avoid mutation within the constant operator.
    ChangeProposer change_proposer = this->change_proposer;
    StateScorer state_scorer = this->state_scorer;

    best_score_ac[0] = -std::numeric_limits<double>::infinity();
    n_best_states_ac[0] = 0;

    for (uint64_t chain_i = 0; chain_i < current_states_ac.get_range()[0];
         chain_i++) {
      current_scores_ac[chain_i] =
          state_scorer.logscore_state(current_states_ac[chain_i]);
    }

    for (uint64_t i = 0; i < n_steps; i++) {
      for (uint64_t chain_i = 0; chain_i < current_states_ac.get_range()[0];
           chain_i++) {
        double neighborhood_correction = 1.0;

        ChainStateImpl proposed_state = current_states_ac[chain_i];
        change_proposer.propose_change(proposed_state, neighborhood_correction);
        double proposed_score = state_scorer.logscore_state(proposed_state);

        double acceptance_probability =
            neighborhood_correction *
            std::exp((proposed_score - current_scores_ac[chain_i]) * gamma);
        bool accept_move = oneapi::dpl::bernoulli_distribution(
            acceptance_probability)(change_proposer.get_rng());
        if (accept_move) {
          current_states_ac[chain_i] = proposed_state;
          current_scores_ac[chain_i] = proposed_score;
        }

        if (proposed_score >= best_score_ac[0]) {
          if (proposed_score > best_score_ac[0]) {
            n_best_states_ac[0] = 0;
            best_score_ac[0] = proposed_score;
          }
          if (n_best_states_ac[0] < best_states_ac.get_range()[0]) {
            best_states_ac[n_best_states_ac[0]] = proposed_state;
            n_best_states_ac[0]++;
          }
        }
      }
    }
  }

  static std::vector<ChainStateImpl>
  run_simulation(cl::sycl::buffer<ac_int<2, false>, 2> data_buffer,
                 cl::sycl::queue working_queue, Parameters const &parameters) {
    using MCMCKernelImpl = MCMCKernel<max_n_cells, max_n_genes, ChangeProposer,
                                      StateScorer, access_target>;

    uint64_t n_cells = data_buffer.get_range()[0];
    uint64_t n_genes = data_buffer.get_range()[1];

    oneapi::dpl::minstd_rand0 twister;
    twister.seed(parameters.get_seed());

    cl::sycl::buffer<ChainStateImpl, 1> best_states_buffer(
        (cl::sycl::range<1>(parameters.get_max_n_best_states())));
    cl::sycl::buffer<double, 1> best_score_buffer((cl::sycl::range<1>(1)));
    cl::sycl::buffer<uint64_t, 1> n_best_states_buffer((cl::sycl::range<1>(1)));

    cl::sycl::buffer<ChainStateImpl, 1> current_states_buffer(
        (cl::sycl::range<1>(parameters.get_n_chains())));
    cl::sycl::buffer<double, 1> current_scores_buffer(
        (cl::sycl::range<1>(parameters.get_n_chains())));

    {
      auto current_states_ac =
          current_states_buffer
              .template get_access<cl::sycl::access::mode::read_write>();
      auto current_scores_ac =
          current_scores_buffer
              .template get_access<cl::sycl::access::mode::read_write>();

      for (uint64_t rep_i = 0; rep_i < parameters.get_n_chains(); rep_i++) {
        current_states_ac[rep_i] = ChainStateImpl::sample_random_state(
            twister, n_genes, parameters.get_beta_mean());
      }
    }

    working_queue.submit([&](cl::sycl::handler &cgh) {
      auto best_states_ac =
          best_states_buffer
              .template get_access<cl::sycl::access::mode::read_write>(cgh);
      auto best_score_ac =
          best_score_buffer
              .template get_access<cl::sycl::access::mode::read_write>(cgh);
      auto n_best_states_ac =
          n_best_states_buffer
              .template get_access<cl::sycl::access::mode::read_write>(cgh);

      auto current_states_ac =
          current_states_buffer
              .template get_access<cl::sycl::access::mode::read_write>(cgh);
      auto current_scores_ac =
          current_scores_buffer
              .template get_access<cl::sycl::access::mode::read_write>(cgh);

      auto data_ac =
          data_buffer.template get_access<cl::sycl::access::mode::read>(cgh);

      ChangeProposer change_proposer(twister, parameters.get_prob_beta_change(),
                                     parameters.get_prob_prune_n_reattach(),
                                     parameters.get_prob_swap_nodes(),
                                     parameters.get_beta_jump_sd());
      StateScorer state_scorer(parameters.get_alpha_mean(),
                               parameters.get_beta_mean(),
                               parameters.get_beta_sd(), data_ac);
      MCMCKernel kernel(change_proposer, state_scorer, parameters.get_gamma(),
                        best_states_ac, best_score_ac, n_best_states_ac,
                        current_states_ac, current_scores_ac,
                        parameters.get_chain_length());
      cgh.single_task(kernel);
    });

    auto best_states_ac =
        best_states_buffer.template get_access<cl::sycl::access::mode::read>();
    auto n_best_states_ac =
        n_best_states_buffer
            .template get_access<cl::sycl::access::mode::read>();
    std::vector<ChainStateImpl> best_states_vec;
    best_states_vec.reserve(n_best_states_ac[0]);
    for (uint64_t i = 0; i < n_best_states_ac[0]; i++) {
      best_states_vec.push_back(best_states_ac[i]);
    }

    return best_states_vec;
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
  StateAccessor best_states_ac, current_states_ac;
  ScoreAccessor best_score_ac, current_scores_ac;
  IndexAccessor n_best_states_ac;

  double gamma;
  uint64_t n_steps;
};
} // namespace ffSCITE