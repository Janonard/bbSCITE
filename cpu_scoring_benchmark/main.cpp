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
#include <CL/sycl.hpp>
#include <MutationTree.hpp>
#include <Parameters.hpp>
#include <TreeScorer.hpp>
#include <iostream>
#include <random>
#include <sstream>

using namespace ffSCITE;
using namespace cl::sycl;

constexpr uint32_t max_n_cells = 64;
constexpr uint32_t max_n_genes = 63;

#ifdef HARDWARE
// Assert that this design does indeed have the correct ranges set.
// I often lower the max number of cells and genes for experiments and then
// forgot to reset them. This should fix it.
static_assert(max_n_cells == 128 && max_n_genes == 127);
#endif

using URNG = std::minstd_rand;

using TreeScorerImpl = TreeScorer<max_n_cells, max_n_genes>;
using MutationDataMatrix = TreeScorerImpl::MutationDataMatrix;
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
  uint32_t n_nodes = n_genes + 1;

  std::cout << "Loading mutation data" << std::endl;

  // Load the mutation input data.
  buffer<AncestryVector, 1> is_mutated_buffer = range<1>(n_cells);
  buffer<AncestryVector, 1> is_known_buffer = range<1>(n_cells);
  {
    auto is_mutated =
        is_mutated_buffer.get_access<access::mode::discard_write>();
    auto is_known = is_known_buffer.get_access<access::mode::discard_write>();
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

  std::cout << "Generating random trees" << std::endl;

  buffer<AncestorMatrix, 1> am_buffer = range<1>(parameters.get_n_chains());
  buffer<AncestorMatrix, 1> dm_buffer = range<1>(parameters.get_n_chains());
  buffer<float, 2> scores_buffer =
      range<2>(parameters.get_n_chains(), parameters.get_chain_length());
  {
    auto am_ac = am_buffer.get_access<access::mode::discard_write>();
    auto dm_ac = dm_buffer.get_access<access::mode::discard_write>();
    std::minstd_rand rng;
    rng.seed(std::random_device()());

    for (uint32_t i = 0; i < parameters.get_n_chains(); i++) {
      std::vector<uint32_t> pruefer_code =
          MutationTreeImpl::sample_random_pruefer_code(rng, n_genes);
      std::vector<uint32_t> parent_vector =
          MutationTreeImpl::pruefer_code_to_parent_vector(pruefer_code);
      std::tie(am_ac[i], dm_ac[i]) =
          MutationTreeImpl::parent_vector_to_matrix(parent_vector);
    }
  }

  // Initializing the SYCL queue.
  property_list queue_properties = {property::queue::enable_profiling{}};
  queue working_queue(cpu_selector().select_device(), queue_properties);

  std::cout << "Submitting work" << std::endl;

  event work_event = working_queue.submit([&](handler &cgh) {
    auto is_mutated_ac = is_mutated_buffer.get_access<access::mode::read>(cgh);
    auto is_known_ac = is_known_buffer.get_access<access::mode::read>(cgh);
    auto am_ac = am_buffer.get_access<access::mode::read_write>(cgh);
    auto dm_ac = dm_buffer.get_access<access::mode::read_write>(cgh);
    auto scores_ac = scores_buffer.get_access<access::mode::discard_write>(cgh);

    float alpha_mean = parameters.get_alpha_mean();
    float beta_mean = parameters.get_beta_mean();
    float beta_sd = parameters.get_beta_sd();

    uint64_t n_chains = parameters.get_n_chains();
    uint64_t chain_length = parameters.get_chain_length();

    cgh.parallel_for_work_group<class TreeScoringKernel>(
        range<2>(n_chains, chain_length), [=](id<2> idx) {
          MutationDataMatrix is_mutated, is_known;
          TreeScorerImpl tree_scorer(alpha_mean, beta_mean, beta_sd, n_cells,
                                     n_genes, is_mutated_ac, is_known_ac,
                                     is_mutated, is_known);
          MutationTreeImpl tree(am_ac[idx[0]], dm_ac[idx[0]], n_genes,
                                beta_mean);
          scores_ac[idx[0]][idx[1]] = tree_scorer.logscore_tree(tree);
        });
  });

  uint64_t start =
      work_event
          .template get_profiling_info<info::event_profiling::command_start>();
  uint64_t end =
      work_event
          .template get_profiling_info<info::event_profiling::command_end>();
  double runtime = (end - start) / 1000000000.0;
  std::cout << "Work finished in " << runtime << " s" << std::endl;

  uint64_t n_chains = parameters.get_n_chains();
  uint64_t chain_length = parameters.get_chain_length();
  uint64_t n_steps = n_chains * chain_length;
  uint64_t popcounted_words = n_steps * n_nodes * n_cells * 4;
  uint64_t floating_point_operations = n_steps * n_nodes * n_cells * 8;

  double steps_per_second = n_steps / runtime;
  double popcounts_per_second = popcounted_words / runtime;
  double counted_bits_per_second =
      (popcounted_words * max_n_genes + 1) / runtime;
  double flops = floating_point_operations / runtime;

  std::cout << "Performance:" << std::endl;
  std::cout << steps_per_second * 1e-3 << " thousand steps per second."
            << std::endl;
  std::cout << popcounts_per_second * 1e-9 << " billion popcounts per second."
            << std::endl;
  std::cout << flops * 1e-9 << " GFLOPS" << std::endl;

  return 0;
}
