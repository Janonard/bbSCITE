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
#include <sstream>

using namespace ffSCITE;

constexpr uint64_t max_n_cells = 128;
constexpr uint64_t max_n_genes = 127;
using ChangeProposerImpl =
    ChangeProposer<max_n_genes, oneapi::dpl::minstd_rand0>;
using StateScorerImpl = StateScorer<max_n_cells, max_n_genes>;
using MCMCKernelImpl =
    MCMCKernel<max_n_cells, max_n_genes, ChangeProposerImpl, StateScorerImpl>;
using ChainStateImpl = ChainState<max_n_genes>;

int main(int argc, char **argv) {
  Parameters parameters;
  if (parameters.load_and_verify_args(argc, argv, max_n_cells, max_n_genes)) {
    std::cerr << "Quitting due to CLI argument errors." << std::endl; 
    return 1;
  }

  cl::sycl::buffer<ac_int<2, false>, 2> data(
      cl::sycl::range<2>(parameters.get_n_cells(), parameters.get_n_genes()));
  {
    auto data_ac = data.get_access<cl::sycl::access::mode::discard_write>();
    std::ifstream input_file(parameters.get_input_path());

    for (uint64_t gene_i = 0; gene_i < parameters.get_n_genes(); gene_i++) {
      for (uint64_t cell_i = 0; cell_i < parameters.get_n_cells(); cell_i++) {
        int entry;
        input_file >> entry;
        switch (entry) {
        case 0:
          data_ac[cell_i][gene_i] = 0;
          break;
        case 1:
        case 2:
          data_ac[cell_i][gene_i] = 1;
          break;
        case 3:
          data_ac[cell_i][gene_i] = 2;
          break;
        default:
          std::cerr << "Error: The input file contains the invalid entry "
                    << entry << std::endl;
          exit(1);
        }
      }
    }
  }

  cl::sycl::queue working_queue(
      (cl::sycl::ext::intel::fpga_emulator_selector()));

  std::vector<ChainStateImpl> best_states =
      MCMCKernelImpl::run_simulation(data, working_queue, parameters);

  // Output the tree in graphviz format
  for (uint64_t state_i = 0; state_i < best_states.size(); state_i++) {
    std::stringstream output_path;
    output_path << parameters.get_output_path_base() << "_ml" << state_i
                << ".gv";

    std::ofstream output_file(output_path.str());
    output_file << "digraph G {" << std::endl;
    output_file << "node [color=deeppink4, style=filled, fontcolor=white];"
                << std::endl;
    for (uint64_t node_i = 0; node_i < parameters.get_n_genes(); node_i++) {
      output_file << best_states[state_i].mutation_tree[node_i] << " -> "
                  << node_i << ";" << std::endl;
    }
    output_file << "}" << std::endl;
  }
}