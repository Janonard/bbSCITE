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
class DummyRNG {
public:
  using result_type = uint32_t;

  DummyRNG() : state(0) {}

  uint32_t operator()() {
    state++;
    return state;
  }

  uint32_t min() { return 0; }

  uint32_t max() { return std::numeric_limits<uint32_t>::max(); }

  void seed(uint32_t seed) { state = seed; }

private:
  uint32_t state;
};

/**
 * @brief SYCL kernel that simulates a SCITE markov chain.
 *
 * It picks up a chain with a given current and best states, simulates the chain
 * from this point for a set number of iterations and writes the resulting state
 * back. It takes a change proposer and a state scorer as it's type argument so
 * that they can easily be swapped out to test different approaches.
 *
 * @tparam max_n_cells The maximum number of cells supported by the kernel.
 * @tparam max_n_genes The maximum number of genes supported by the kernel.
 * @tparam ChangeProposer The type of the change proposing strategy.
 * @tparam StateScorer The type of the state scoring strategy.
 * @tparam access_target The target from where the states are accessed.
 */
template <uint32_t max_n_cells, uint32_t max_n_genes, typename RNG>
class MCMCKernel {
public:
  /**
   * @brief Shorthand for the used chain state.
   */
  using ChainStateImpl = ChainState<max_n_genes>;

  /**
   * @brief Shorthand for the chain state buffer accessor type.
   */
  using StateAccessor =
      cl::sycl::accessor<ChainStateImpl, 1, cl::sycl::access::mode::read_write>;

  /**
   * @brief Shorthand for the double buffer accessor type.
   */
  using ScoreAccessor =
      cl::sycl::accessor<double, 1, cl::sycl::access::mode::read_write>;

  /**
   * @brief Shorthand for the index buffer accessor type.
   */
  using IndexAccessor =
      cl::sycl::accessor<uint32_t, 1, cl::sycl::access::mode::read_write>;

  using ChangeProposerImpl = ChangeProposer<max_n_genes, RNG>;

  using StateScorerImpl = StateScorer<max_n_cells, max_n_genes>;

  using HostStateScorerImpl =
      StateScorer<max_n_cells, max_n_genes,
                  cl::sycl::access::target::host_buffer>;

  using DataEntry = typename StateScorerImpl::DataEntry;

  using DataMatrix = typename StateScorerImpl::DataMatrix;

  using OccurrenceMatrix = typename StateScorerImpl::OccurrenceMatrix;

  using MutationDataAccessor = typename StateScorerImpl::MutationDataAccessor;

  /**
   * @brief Initialize the MCMC kernel instance.
   *
   * @param change_proposer The configured instance of the change proposer to
   * use.
   * @param state_scorer The configured instance of the state scorer to
   * use.
   * @param gamma A factor for the width of the optimization space
   * exploration.
   * @param best_states_ac Accessor to the buffer for the optimal states.
   * @param best_score_ac Accessor to the score of the optimal states.
   * @param current_states_ac Accessor to the current states of the
   * chains.
   * @param current_scores_ac Accessor to the scores of the current states of
   * the chains.
   * @param n_best_states_ac Accessor to the number of best states.
   * @param n_steps The number of steps to execute.
   */
  MCMCKernel(StateAccessor best_states_ac, StateAccessor current_states_ac,
             ScoreAccessor best_score_ac, ScoreAccessor current_scores_ac,
             IndexAccessor n_best_states_ac, MutationDataAccessor data_ac,
             RNG rng, double prob_beta_change, double prob_prune_n_reattach,
             double prob_swap_nodes, double beta_jump_sd, double alpha_mean,
             double beta_mean, double beta_sd, double gamma, uint32_t n_steps)
      : best_states_ac(best_states_ac), current_states_ac(current_states_ac),
        best_score_ac(best_score_ac), current_scores_ac(current_scores_ac),
        n_best_states_ac(n_best_states_ac), prob_beta_change(prob_beta_change),
        data_ac(data_ac), prob_prune_n_reattach(prob_prune_n_reattach),
        prob_swap_nodes(prob_swap_nodes), beta_jump_sd(beta_jump_sd),
        alpha_mean(alpha_mean), beta_mean(beta_mean), beta_sd(beta_sd),
        gamma(gamma), n_steps(n_steps) {
    assert(best_score_ac.get_range()[0] == 1);
    assert(current_states_ac.get_range() == current_scores_ac.get_range());
  }

  /**
   * @brief Run the MCMC chain.
   *
   * This will read the current and best states from device memory, execute the
   * configured number of steps for every chain and writes the results back.
   */
  void operator()() const {
    // Copy of score to avoid mutation within the constant operator.
    [[intel::fpga_register]] ChangeProposerImpl change_proposer(
        rng, prob_beta_change, prob_prune_n_reattach, prob_swap_nodes,
        beta_jump_sd);

    DataMatrix data;
    StateScorerImpl state_scorer(alpha_mean, beta_mean, beta_sd, data_ac, data);

    double best_score = -std::numeric_limits<double>::infinity();
    uint32_t n_best_states = 0;

    for (uint32_t i = 0; i < n_steps; i++) {
      for (uint32_t chain_i = 0; chain_i < current_states_ac.get_range()[0];
           chain_i++) {
        double neighborhood_correction = 1.0;
        [[intel::fpga_register]] ChainStateImpl current_state =
            current_states_ac[chain_i];
        double current_score = current_scores_ac[chain_i];

        [[intel::fpga_register]] ChainStateImpl proposed_state = current_state;
        change_proposer.propose_change(proposed_state, neighborhood_correction);
        double proposed_score = state_scorer.logscore_state(proposed_state);

        double acceptance_probability =
            neighborhood_correction *
            std::exp((proposed_score - current_score) * gamma);
        bool accept_move = oneapi::dpl::bernoulli_distribution(
            acceptance_probability)(change_proposer.get_rng());
        if (accept_move) {
          current_states_ac[chain_i] = proposed_state;
          current_scores_ac[chain_i] = proposed_score;
        }

        if (proposed_score > best_score || n_best_states == 0) {
          best_states_ac[0] = proposed_state;
          best_score = proposed_score;
          n_best_states = 1;
        }
      }
    }

    best_score_ac[0] = best_score;
    n_best_states_ac[0] = n_best_states;
  }

  /**
   * @brief Run the MCMC experiment with the given data and parameters.
   *
   * @param data_buffer The mutation data from the input file. The number of
   * cells and genes is inferred from its range.
   * @param working_queue The configured SYCL queue to execute the simulation
   * with.
   * @param parameters The configuration parameters of the simulation
   * @return std::vector<ChainStateImpl> A vector with all of the optimal
   * states.
   * @return cl::sycl::event The SYCL event of the MCMC kernel execution.
   */
  static std::tuple<std::vector<ChainStateImpl>, cl::sycl::event>
  run_simulation(cl::sycl::buffer<ac_int<2, false>, 2> data_buffer,
                 cl::sycl::queue working_queue, Parameters const &parameters) {
    using MCMCKernelImpl = MCMCKernel<max_n_cells, max_n_genes, RNG>;

    uint32_t n_cells = data_buffer.get_range()[0];
    uint32_t n_genes = data_buffer.get_range()[1];

    RNG twister;
    twister.seed(parameters.get_seed());

    cl::sycl::buffer<ChainStateImpl, 1> best_states_buffer(
        (cl::sycl::range<1>(parameters.get_max_n_best_states())));
    cl::sycl::buffer<double, 1> best_score_buffer((cl::sycl::range<1>(1)));
    cl::sycl::buffer<uint32_t, 1> n_best_states_buffer((cl::sycl::range<1>(1)));

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
      auto data_ac =
          data_buffer.template get_access<cl::sycl::access::mode::read>();
      DataMatrix data;

      HostStateScorerImpl host_scorer(parameters.get_alpha_mean(),
                                      parameters.get_beta_mean(),
                                      parameters.get_beta_sd(), data_ac, data);

      for (uint32_t rep_i = 0; rep_i < parameters.get_n_chains(); rep_i++) {
        current_states_ac[rep_i] = ChainStateImpl::sample_random_state(
            twister, n_genes, parameters.get_beta_mean());
        current_scores_ac[rep_i] =
            host_scorer.logscore_state(current_states_ac[rep_i]);
      }
    }

    cl::sycl::event event = working_queue.submit([&](cl::sycl::handler &cgh) {
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

      double beta_jump_sd =
          parameters.get_beta_sd() / parameters.get_beta_jump_scaling_chi();
      MCMCKernel kernel(best_states_ac, current_states_ac, best_score_ac,
                        current_scores_ac, n_best_states_ac, data_ac, twister,
                        parameters.get_prob_beta_change(),
                        parameters.get_prob_prune_n_reattach(),
                        parameters.get_prob_swap_nodes(), beta_jump_sd,
                        parameters.get_alpha_mean(), parameters.get_beta_mean(),
                        parameters.get_beta_sd(), parameters.get_gamma(),
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
    for (uint32_t i = 0; i < n_best_states_ac[0]; i++) {
      best_states_vec.push_back(best_states_ac[i]);
    }

    return {best_states_vec, event};
  }

private:
  StateAccessor best_states_ac, current_states_ac;
  ScoreAccessor best_score_ac, current_scores_ac;
  IndexAccessor n_best_states_ac;
  MutationDataAccessor data_ac;

  RNG rng;
  double prob_beta_change, prob_prune_n_reattach, prob_swap_nodes, beta_jump_sd;
  double alpha_mean, beta_mean, beta_sd;
  double gamma;
  uint32_t n_steps;
};
} // namespace ffSCITE