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
  // Load the CLI parameters.
  Parameters parameters;
  if (!parameters.load_and_verify_args(argc, argv, max_n_cells, max_n_genes)) {
    std::cerr << "Quitting due to CLI argument errors." << std::endl;
    return 1;
  }

  // Load the mutation input data.
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

  // Initializing the SYCL queue.
  cl::sycl::device device =
      cl::sycl::ext::intel::fpga_emulator_selector().select_device();
  cl::sycl::property_list queue_properties = {
      cl::sycl::property::queue::enable_profiling{}};
  cl::sycl::queue working_queue(device, queue_properties);

  // Running the simulation and retrieving the best states.
  auto result = MCMCKernelImpl::run_simulation(data, working_queue, parameters);
  std::vector<ChainStateImpl> best_states = std::get<0>(result);
  cl::sycl::event runtime_event = std::get<1>(result);

  static constexpr double timesteps_per_second = 1000000000.0;
  double start_of_event =
      runtime_event.get_profiling_info<
          cl::sycl::info::event_profiling::command_start>() /
      timesteps_per_second;
  double end_of_event =
      runtime_event
          .get_profiling_info<cl::sycl::info::event_profiling::command_end>() /
      timesteps_per_second;
  std::cout << "Simulation completed in " << end_of_event - start_of_event
            << "s" << std::endl;

  for (uint64_t state_i = 0; state_i < best_states.size(); state_i++) {
    // Output the tree as a graphviz file.
    {
      std::stringstream output_path;
      output_path << parameters.get_output_path_base() << "_ml" << state_i
                  << ".gv";

      std::ofstream output_file(output_path.str());
      output_file << best_states[state_i].mutation_tree.to_graphviz();
    }

    // Output the tree in newick format
    {
      std::stringstream output_path;
      output_path << parameters.get_output_path_base() << "_ml" << state_i
                  << ".newick";

      std::ofstream output_file(output_path.str());
      output_file << best_states[state_i].mutation_tree.to_newick();
    }
  }
}