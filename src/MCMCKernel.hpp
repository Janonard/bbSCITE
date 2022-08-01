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
#include "TreeScorer.hpp"
#include <CL/sycl.hpp>

namespace ffSCITE {
/**
 * @brief SYCL kernel that simulates a SCITE markov chain.
 *
 * It picks up a chain with a given current and best trees, simulates the chain
 * from this point for a set number of iterations and writes the resulting tree
 * back. It takes a change proposer and a tree scorer as it's type argument so
 * that they can easily be swapped out to test different approaches.
 *
 * @tparam max_n_cells The maximum number of cells supported by the kernel.
 * @tparam max_n_genes The maximum number of genes supported by the kernel.
 * @tparam ChangeProposer The type of the change proposing strategy.
 * @tparam TreeScorer The type of the tree scoring strategy.
 * @tparam access_target The target from where the trees are accessed.
 */
template <uint32_t max_n_cells, uint32_t max_n_genes, typename RNG>
class MCMCKernel {
public:
  /**
   * @brief Shorthand for the used tree.
   */
  using MutationTreeImpl = MutationTree<max_n_genes>;

  using AncestorMatrix = typename MutationTreeImpl::AncestorMatrix;

  using AncestorMatrixAccessor =
      cl::sycl::accessor<AncestorMatrix, 1, cl::sycl::access::mode::read_write>;

  using DoubleAccessor =
      cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write>;

  /**
   * @brief Shorthand for the index buffer accessor type.
   */
  using IndexAccessor =
      cl::sycl::accessor<uint32_t, 1, cl::sycl::access::mode::read_write>;

  using ChangeProposerImpl = ChangeProposer<max_n_genes, RNG>;

  using TreeScorerImpl = TreeScorer<max_n_cells, max_n_genes>;

  using HostTreeScorerImpl = TreeScorer<max_n_cells, max_n_genes,
                                        cl::sycl::access::target::host_buffer>;

  using MutationDataWord = typename TreeScorerImpl::MutationDataWord;

  using MutationDataMatrix = typename TreeScorerImpl::MutationDataMatrix;

  using MutationDataAccessor =
      cl::sycl::accessor<MutationDataWord, 1, cl::sycl::access::mode::read>;

  using OccurrenceMatrix = typename TreeScorerImpl::OccurrenceMatrix;

  using AncestorInPipe = cl::sycl::pipe<class AncestorInPipeID, AncestorMatrix>;
  using AncestorOutPipe =
      cl::sycl::pipe<class AncestorOutPipeID, AncestorMatrix>;
  using BetaInPipe = cl::sycl::pipe<class BetaInPipeID, float>;
  using BetaOutPipe = cl::sycl::pipe<class BetaOutPipeID, float>;
  using ScoreInPipe = cl::sycl::pipe<class ScoreInPipeID, float>;
  using ScoreOutPipe = cl::sycl::pipe<class ScoreOutPipeID, float>;

  struct Accessors {
    AncestorMatrixAccessor best_am;
    DoubleAccessor best_beta;
    DoubleAccessor best_score;
    IndexAccessor n_best_trees;
    MutationDataAccessor mutation_data;
  };

  MCMCKernel(Accessors accessors, RNG rng, float prob_beta_change,
             float prob_prune_n_reattach, float prob_swap_nodes,
             float beta_jump_sd, float alpha_mean, float beta_mean,
             float beta_sd, float gamma, uint32_t n_chains, uint32_t n_steps,
             uint32_t n_cells, uint32_t n_genes)
      : acs(accessors), prob_beta_change(prob_beta_change),
        prob_prune_n_reattach(prob_prune_n_reattach),
        prob_swap_nodes(prob_swap_nodes), beta_jump_sd(beta_jump_sd),
        alpha_mean(alpha_mean), beta_mean(beta_mean), beta_sd(beta_sd),
        gamma(gamma), n_chains(n_chains), n_steps(n_steps), n_cells(n_cells),
        n_genes(n_genes) {
    assert(acs.best_am.get_range() == acs.best_beta.get_range());
    assert(acs.best_score.get_range()[0] == 1);
  }

  /**
   * @brief Run the MCMC chain.
   *
   * This will read the current and best trees from device memory, execute the
   * configured number of steps for every chain and writes the results back.
   */
  void operator()() const {
    [[intel::fpga_register]] ChangeProposerImpl change_proposer(
        rng, prob_beta_change, prob_prune_n_reattach, prob_swap_nodes,
        beta_jump_sd);

    [[intel::fpga_memory]] MutationDataMatrix data;
    TreeScorerImpl tree_scorer(alpha_mean, beta_mean, beta_sd, n_cells, n_genes,
                               acs.mutation_data, data);

    AncestorMatrix best_am;
    float best_beta;
    float best_score = -std::numeric_limits<float>::infinity();
    uint32_t n_best_trees = 0;

    [[intel::loop_coalesce(2)]] for (uint32_t i = 0; i < n_steps; i++) {
      for (uint32_t chain_i = 0; chain_i < n_chains; chain_i++) {
        float neighborhood_correction = 1.0;
        [[intel::fpga_memory]] AncestorMatrix current_am =
            AncestorInPipe::read();
        float current_beta = BetaInPipe::read();
        float current_score = ScoreInPipe::read();
        MutationTreeImpl current_tree(current_am, n_genes, current_beta);

        ChainStepParameters step_parameters =
            change_proposer.sample_step_parameters(current_tree);

        [[intel::fpga_memory]] AncestorMatrix proposed_am;
        MutationTreeImpl proposed_tree(proposed_am, current_tree,
                                       step_parameters);
        float proposed_score = tree_scorer.logscore_tree(proposed_tree);

        float acceptance_probability =
            step_parameters.tree_swap_neighborhood_correction *
            std::exp((proposed_score - current_score) * gamma);
        bool accept_move =
            acceptance_probability > step_parameters.acceptance_level;

        AncestorOutPipe::write(accept_move ? proposed_am : current_am);
        BetaOutPipe::write(accept_move ? proposed_tree.get_beta()
                                       : current_beta);
        ScoreOutPipe::write(accept_move ? proposed_score : current_score);

        if (proposed_score > best_score || n_best_trees == 0) {
          best_am = proposed_am;
          best_beta = proposed_tree.get_beta();
          best_score = proposed_score;
          n_best_trees = 1;
        }
      }
    }

    acs.best_am[0] = best_am;
    acs.best_beta[0] = best_beta;
    acs.best_score[0] = best_score;
    acs.n_best_trees[0] = n_best_trees;
  }

  /**
   * @brief Run the MCMC experiment with the given data and parameters.
   *
   * @param data_buffer The mutation data from the input file. The number of
   * cells and genes is inferred from its range.
   * @param working_queue The configured SYCL queue to execute the simulation
   * with.
   * @param parameters The configuration parameters of the simulation
   * @return std::vector<MutationTreeImpl> A vector with all of the optimal
   * trees.
   * @return cl::sycl::event The SYCL event of the MCMC kernel execution.
   */
  static std::tuple<std::vector<AncestorMatrix>, std::vector<float>,
                    cl::sycl::event>
  run_simulation(cl::sycl::buffer<MutationDataWord, 1> data_buffer,
                 cl::sycl::queue working_queue, Parameters const &parameters,
                 uint32_t n_cells, uint32_t n_genes) {
    using MCMCKernelImpl = MCMCKernel<max_n_cells, max_n_genes, RNG>;

    using namespace cl::sycl;

    uint32_t n_chains = parameters.get_n_chains();
    uint32_t n_steps = parameters.get_chain_length();
    uint32_t max_n_trees = parameters.get_max_n_best_trees();

    RNG twister;
    twister.seed(parameters.get_seed());

    buffer<AncestorMatrix, 1> current_am_buffer((range<1>(n_chains)));
    buffer<float, 1> current_beta_buffer((range<1>(n_chains)));
    buffer<float, 1> current_score_buffer((range<1>(n_chains)));

    buffer<AncestorMatrix, 1> best_am_buffer((range<1>(max_n_trees)));
    buffer<float, 1> best_beta_buffer((range<1>(max_n_trees)));
    buffer<float, 1> best_score_buffer((range<1>(1)));
    buffer<uint32_t, 1> n_best_trees_buffer((range<1>(1)));

    {
      auto current_am_ac =
          current_am_buffer.template get_access<access::mode::discard_write>();
      auto current_beta_ac =
          current_beta_buffer
              .template get_access<access::mode::discard_write>();
      auto current_score_ac =
          current_score_buffer
              .template get_access<access::mode::discard_write>();
      auto data_ac = data_buffer.template get_access<access::mode::read>();
      MutationDataMatrix data;

      HostTreeScorerImpl host_scorer(
          parameters.get_alpha_mean(), parameters.get_beta_mean(),
          parameters.get_beta_sd(), n_cells, n_genes, data_ac, data);

      for (uint32_t rep_i = 0; rep_i < parameters.get_n_chains(); rep_i++) {
        std::vector<uint32_t> pruefer_code =
            MutationTreeImpl::sample_random_pruefer_code(twister, n_genes);
        std::vector<uint32_t> parent_vector =
            MutationTreeImpl::pruefer_code_to_parent_vector(pruefer_code);
        current_am_ac[rep_i] =
            MutationTreeImpl::parent_vector_to_ancestor_matrix(parent_vector);

        current_beta_ac[rep_i] = parameters.get_beta_mean();

        MutationTreeImpl tree(current_am_ac[rep_i], n_genes,
                              current_beta_ac[rep_i]);
        current_score_ac[rep_i] = host_scorer.logscore_tree(tree);
      }
    }

    event event = working_queue.submit([&](handler &cgh) {
      auto best_am_ac =
          best_am_buffer.template get_access<access::mode::read_write>(cgh);
      auto best_beta_ac =
          best_beta_buffer.template get_access<access::mode::read_write>(cgh);
      auto best_score_ac =
          best_score_buffer.template get_access<access::mode::read_write>(cgh);
      auto n_best_trees_ac =
          n_best_trees_buffer.template get_access<access::mode::read_write>(
              cgh);

      auto data_ac = data_buffer.template get_access<access::mode::read>(cgh);

      Accessors accessors{
          .best_am = best_am_ac,
          .best_beta = best_beta_ac,
          .best_score = best_score_ac,
          .n_best_trees = n_best_trees_ac,
          .mutation_data = data_ac,
      };

      float beta_jump_sd =
          parameters.get_beta_sd() / parameters.get_beta_jump_scaling_chi();
      MCMCKernel kernel(accessors, twister, parameters.get_prob_beta_change(),
                        parameters.get_prob_prune_n_reattach(),
                        parameters.get_prob_swap_nodes(), beta_jump_sd,
                        parameters.get_alpha_mean(), parameters.get_beta_mean(),
                        parameters.get_beta_sd(), parameters.get_gamma(),
                        n_chains, n_steps, n_cells, n_genes);
      cgh.single_task(kernel);
    });

    working_queue.submit([&](handler &cgh) {
      auto current_am_ac =
          current_am_buffer.template get_access<access::mode::read_write>(cgh);
      auto current_beta_ac =
          current_beta_buffer.template get_access<access::mode::read_write>(
              cgh);
      auto current_score_ac =
          current_score_buffer.template get_access<access::mode::read_write>(
              cgh);

      cgh.single_task<class IOKernel>([=]() {
        [[intel::loop_coalesce(2)]] for (uint32_t step_i = 0; step_i < n_steps;
                                         step_i++) {
          for (uint32_t chain_i = 0; chain_i < n_chains; chain_i++) {
            AncestorInPipe::write(current_am_ac[chain_i]);
            BetaInPipe::write(current_beta_ac[chain_i]);
            ScoreInPipe::write(current_score_ac[chain_i]);

            current_am_ac[chain_i] = AncestorOutPipe::read();
            current_beta_ac[chain_i] = BetaOutPipe::read();
            current_score_ac[chain_i] = ScoreOutPipe::read();
          }
        }
      });
    });

    auto best_am_ac = best_am_buffer.template get_access<access::mode::read>();
    auto best_beta_ac =
        best_beta_buffer.template get_access<access::mode::read>();
    auto n_best_trees_ac =
        n_best_trees_buffer.template get_access<access::mode::read>();

    std::vector<AncestorMatrix> best_am_vec;
    std::vector<float> best_beta_vec;
    best_am_vec.reserve(n_best_trees_ac[0]);
    best_beta_vec.reserve(n_best_trees_ac[0]);
    for (uint32_t i = 0; i < n_best_trees_ac[0]; i++) {
      best_am_vec.push_back(best_am_ac[i]);
      best_beta_vec.push_back(best_beta_ac[i]);
    }

    return {best_am_vec, best_beta_vec, event};
  }

private:
  Accessors acs;

  RNG rng;
  float prob_beta_change, prob_prune_n_reattach, prob_swap_nodes, beta_jump_sd;
  float alpha_mean, beta_mean, beta_sd;
  float gamma;
  uint32_t n_chains, n_steps;
  uint32_t n_cells, n_genes;
};
} // namespace ffSCITE