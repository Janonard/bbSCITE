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
#include "Application.hpp"
#include "Parameters.hpp"
#include <CL/sycl.hpp>
#include <ext/intel/fpga_extensions.hpp>
#include <iostream>
#include <sstream>

using namespace ffSCITE;

constexpr uint32_t max_n_cells = 64;
constexpr uint32_t max_n_genes = 63;

#ifdef HARDWARE
// Assert that this design does indeed have the correct ranges set.
// I often lower the max number of cells and genes for experiments and then
// forgot to reset them. This should fix it.
static_assert(max_n_cells == 64 && max_n_genes == 63);
#endif

using URNG = oneapi::dpl::minstd_rand;

using ApplicationImpl = Application<max_n_cells, max_n_genes>;
using MutationTreeImpl = MutationTree<max_n_genes>;
using AncestorMatrix = MutationTreeImpl::AncestorMatrix;
using AncestryVector = MutationTreeImpl::AncestryVector;

int main(int argc, char **argv) {
  // Load the CLI parameters.
  Parameters parameters;
  if (!parameters.load_and_verify_args(argc, argv)) {
    std::cerr << "Quitting due to CLI argument errors." << std::endl;
    return 1;
  }

  // First pass over the input data to check validity and identify number of
  // cells and genes.
  uint32_t n_cells = 0, n_genes = 0;
  {
    std::ifstream input_file(parameters.get_input_path());
    char c;
    uint32_t cells_in_line = 0;

    while ((c = input_file.get()) != -1) {
      if (c == '0' || c == '1' || c == '2' || c == '3') {
        cells_in_line++;
      } else if (c == '\n') {
        if (n_cells == 0) {
          n_cells = cells_in_line;
        } else if (n_cells != cells_in_line) {
          std::cerr << "Error: Inconsistent rows in the input file!"
                    << std::endl;
          return 1;
        }
        n_genes++;
        cells_in_line = 0;
      } else if (c == ' ') {
        continue;
      } else {
        std::cerr << "Error: Illegal character " << c << " in the input file!"
                  << std::endl;
        return 1;
      }
    }

    if (cells_in_line == n_cells) {
      n_genes++;
    } else if (cells_in_line != 0) {
      std::cerr << "Error: Inconsistent rows in the input file!" << std::endl;
    }
  }
  if (n_cells == 0 || n_genes == 0) {
    std::cerr << "Error: Empty input file!" << std::endl;
    return 1;
  }
  if (n_cells > max_n_cells) {
    std::cerr << "Error: This build of ffSCITE only supports up " << max_n_cells
              << " cells. The input data contains " << n_cells << " cells."
              << std::endl;
    return 1;
  }
  if (n_genes > max_n_genes) {
    std::cerr << "Error: This build of ffSCITE only supports up " << max_n_genes
              << " genes. The input data contains " << n_genes << " genes."
              << std::endl;
    return 1;
  }

  // Load the mutation input data.
  cl::sycl::buffer<AncestryVector, 1> is_mutated_buffer =
      cl::sycl::range<1>(n_cells);
  cl::sycl::buffer<AncestryVector, 1> is_known_buffer =
      cl::sycl::range<1>(n_cells);
  {
    auto is_mutated =
        is_mutated_buffer.get_access<cl::sycl::access::mode::discard_write>();
    auto is_known =
        is_known_buffer.get_access<cl::sycl::access::mode::discard_write>();
    std::ifstream input_file(parameters.get_input_path());

    // Zeroing the data.
    for (uint32_t cell_i = 0; cell_i < n_cells; cell_i++) {
      is_mutated[cell_i] = is_known[cell_i] = 0;
    }

    for (uint32_t gene_i = 0; gene_i < n_genes; gene_i++) {
      for (uint32_t cell_i = 0; cell_i < n_cells; cell_i++) {
        int entry;
        input_file >> entry;
        switch (entry) {
        case 0:
          is_mutated[cell_i][gene_i] = 0;
          is_known[cell_i][gene_i] = 1;
          break;
        case 1:
        case 2:
          is_mutated[cell_i][gene_i] = 1;
          is_known[cell_i][gene_i] = 1;
          break;
        case 3:
          is_known[cell_i][gene_i] = 0;
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
#ifdef EMULATOR
  cl::sycl::device device =
      cl::sycl::ext::intel::fpga_emulator_selector().select_device();
#else
  cl::sycl::device device =
      cl::sycl::ext::intel::fpga_selector().select_device();
#endif
  cl::sycl::property_list queue_properties = {
      cl::sycl::property::queue::enable_profiling{}};
  cl::sycl::queue working_queue(device, queue_properties);

  // Running the simulation and retrieving the best trees.
  ApplicationImpl app(is_mutated_buffer, is_known_buffer, working_queue,
                      parameters, n_cells, n_genes);
  float runtime = app.run_simulation();

  std::cout << "Time elapsed: " << runtime << " ms" << std::endl;

  AncestorMatrix best_am = app.get_best_am();
  float best_beta = app.get_best_beta();

  MutationTreeImpl tree(best_am, n_genes, best_beta);

  // Output the tree as a graphviz file.
  {
    std::stringstream output_path;
    output_path << parameters.get_output_path_base() << "_ml0.gv";

    std::ofstream output_file(output_path.str());
    output_file << tree.to_graphviz();
  }

  // Output the tree in newick format
  {
    std::stringstream output_path;
    output_path << parameters.get_output_path_base() << "_ml0.newick";

    std::ofstream output_file(output_path.str());
    output_file << tree.to_newick();
  }

  // Output the found beta value for the tree
  {
    std::stringstream output_path;
    output_path << parameters.get_output_path_base() << "_ml0_beta.txt";

    std::ofstream output_file(output_path.str());
    output_file << best_beta << std::endl;
  }
}
