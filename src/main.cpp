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
#include "ChangeProposer.hpp"
#include "MCMCKernel.hpp"
#include "Parameters.hpp"
#include "StateScorer.hpp"
#include <CL/sycl.hpp>
#include <ext/intel/fpga_extensions.hpp>
#include <iostream>

using namespace ffSCITE;

constexpr uint64_t max_n_cells = 128;
constexpr uint64_t max_n_genes = 127;
using ChangeProposerImpl =
    ChangeProposer<max_n_genes, oneapi::dpl::minstd_rand0>;
using StateScorerImpl = StateScorer<max_n_cells, max_n_genes>;
using MCMCKernelImpl =
    MCMCKernel<max_n_cells, max_n_genes, ChangeProposerImpl, StateScorerImpl>;
using MutationDataMatrix = StateScorerImpl::MutationDataMatrix;
using ChainStateImpl = ChainState<max_n_genes>;

int main(int argc, char **argv) {
  Parameters parameters(argc, argv, max_n_cells, max_n_genes);

  oneapi::dpl::minstd_rand0 rng;
  rng.seed(parameters.get_seed());

  ChangeProposerImpl change_proposer(rng, parameters.get_prob_beta_change(),
                                     parameters.get_prob_prune_n_reattach(),
                                     parameters.get_prob_swap_nodes(),
                                     parameters.get_beta_jump_sd());

  MutationDataMatrix data;
  {
    std::ifstream input_file(parameters.get_input_path());
    for (uint64_t gene_i = 0; gene_i < parameters.get_n_genes(); gene_i++) {
      for (uint64_t cell_i = 0; cell_i < parameters.get_n_cells(); cell_i++) {
        int entry;
        input_file >> entry;
        switch (entry) {
        case 0:
          data[cell_i][gene_i] = 0;
          break;
        case 1:
        case 2:
          data[cell_i][gene_i] = 1;
          break;
        case 3:
          data[cell_i][gene_i] = 2;
          break;
        default:
          std::cerr << "Error: The input file contains the invalid entry "
                    << entry << std::endl;
          exit(1);
        }
      }
    }
  }

  StateScorerImpl state_scorer(
      parameters.get_alpha(), parameters.get_beta(), parameters.get_beta_sd(),
      parameters.get_n_cells(), parameters.get_n_genes(), data);

  cl::sycl::buffer<std::tuple<ChainStateImpl, double>, 1> best_state_buffer(
      cl::sycl::range<1>(1));
  {
    ChainStateImpl best_state = ChainStateImpl::sample_random_state(
        change_proposer.get_rng(), parameters.get_n_genes(),
        parameters.get_beta());
    double best_score = state_scorer.logscore_state(best_state);

    auto best_state_ac =
        best_state_buffer.get_access<cl::sycl::access::mode::discard_write>();
    best_state_ac[0] = {best_state, best_score};
  }

  cl::sycl::queue working_queue(
      (cl::sycl::ext::intel::fpga_emulator_selector()));

  for (uint64_t rep_i = 0; rep_i < parameters.get_n_chains(); rep_i++) {
    cl::sycl::buffer<std::tuple<ChainStateImpl, double>, 1>
        current_state_buffer(cl::sycl::range<1>(1));
    {
      ChainStateImpl current_state = ChainStateImpl::sample_random_state(
          change_proposer.get_rng(), parameters.get_n_genes(),
          parameters.get_beta());
      double current_score = state_scorer.logscore_state(current_state);

      auto current_state_ac =
          current_state_buffer
              .get_access<cl::sycl::access::mode::discard_write>();
      current_state_ac[0] = {current_state, current_score};
    }

    working_queue.submit([&](cl::sycl::handler &cgh) {
      auto best_state_ac =
          best_state_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);
      auto current_state_ac =
          current_state_buffer.get_access<cl::sycl::access::mode::read_write>(
              cgh);
      uint64_t chain_length = parameters.get_chain_length();
      double gamma = parameters.get_gamma();

      MCMCKernelImpl kernel(change_proposer, state_scorer, gamma, best_state_ac,
                            current_state_ac, chain_length);
      cgh.single_task<MCMCKernelImpl>(kernel);
    });
  }

  auto best_state_ac =
      best_state_buffer.get_access<cl::sycl::access::mode::read>();
  ChainStateImpl best_state = std::get<0>(best_state_ac[0]);

  // Output the tree in graphviz format
  {
    std::ofstream output_file(parameters.get_output_path_base() + "_ml0.gv");
    output_file << "digraph G {" << std::endl;
    output_file << "node [color=deeppink4, style=filled, fontcolor=white];"
                << std::endl;
    for (uint64_t node_i = 0; node_i < parameters.get_n_genes(); node_i++) {
      output_file << best_state.mutation_tree[node_i] << " -> " << node_i << ";"
                  << std::endl;
    }
    output_file << "}" << std::endl;
  }
}