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
#include "CPUTreeScorer.hpp"
#include <CL/sycl.hpp>
#include <MutationTree.hpp>
#include <Parameters.hpp>
#include <iostream>
#include <random>
#include <sstream>

using namespace ffSCITE;
using namespace cl::sycl;

constexpr uint32_t n_words = CPUTreeScorer::n_words;
constexpr uint32_t n_cells = CPUTreeScorer::n_cells;
constexpr uint32_t n_genes = CPUTreeScorer::n_genes;
constexpr uint32_t n_nodes = CPUTreeScorer::n_nodes;

using MutationDataMatrix = CPUTreeScorer::MutationDataMatrix;
using AncestorMatrix = CPUTreeScorer::AncestorMatrix;
using MutationTreeImpl = MutationTree<n_genes>;

using URNG = std::minstd_rand;

AncestorMatrix
parent_vector_to_descendant_matrix(std::vector<uint32_t> const &parent_vector) {
  AncestorMatrix descendant;
  uint32_t root = n_nodes - 1;

  for (uint32_t i_word = 0; i_word < n_words; i_word++) {
    for (uint32_t j = 0; j < n_nodes; j++) {
      // Zero all vectors.
      descendant[i_word][j] = 0;
    }
  }

  for (uint32_t i = 0; i < n_nodes; i++) {
    // Then we start from the node i and walk up to the root, marking all
    // nodes on the way as ancestors.
    uint32_t anc = i;
    while (anc != root) {
      descendant[anc / 64][i] += 1 << (anc % 64);
      anc = parent_vector[anc];
      // Otherwise, there is a circle in the graph!
      assert(anc != i && anc < n_nodes);
    }

    // Lastly, also mark ourselves and the root as our ancestor.
    descendant[i / 64][i] += 1 << (i % 64);
    descendant[root / 64][i] += 1 << (i % root);
  }

  return descendant;
}

int main(int argc, char **argv) {
  // Load the CLI parameters.
  Parameters parameters;
  if (!parameters.load_and_verify_args(argc, argv)) {
    std::cerr << "Quitting due to CLI argument errors." << std::endl;
    return 1;
  }

  // First pass over the input data to check validity and identify number of
  // cells and genes.
  {
    std::ifstream input_file(parameters.get_input_path());
    char c;
    uint32_t cells_in_line = 0;
    uint32_t genes_in_file = 0;

    while ((c = input_file.get()) != -1) {
      if (c == '0' || c == '1' || c == '2' || c == '3') {
        cells_in_line++;
      } else if (c == '\n') {
        if (cells_in_line != n_cells) {
          std::cerr << "Error: Illegal number of cells! This benchmark "
                       "requires exactly "
                    << n_cells << " cells!" << std::endl;
          return 1;
        }
        genes_in_file++;
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
      genes_in_file++;
    } else if (cells_in_line != 0) {
      std::cerr
          << "Error: Illegal number of cells! This benchmark requires exactly "
          << n_cells << " cells!" << std::endl;
      return 1;
    }

    if (genes_in_file != n_genes) {
      std::cerr
          << "Error: Illegal number of genes! This benchmark requires exactly "
          << n_genes << " genes!" << std::endl;
      return 1;
    }
  }

  std::cout << "Loading mutation data" << std::endl;

  // Load the mutation input data.
  buffer<MutationDataMatrix, 1> is_mutated_buffer = range<1>(1);
  buffer<MutationDataMatrix, 1> is_known_buffer = range<1>(1);
  {
    auto is_mutated =
        is_mutated_buffer.get_access<access::mode::discard_write>();
    auto is_known = is_known_buffer.get_access<access::mode::discard_write>();
    std::ifstream input_file(parameters.get_input_path());

    // Zeroing the data.
    for (uint32_t word_i = 0; word_i < n_words; word_i++) {
      for (uint32_t cell_i = 0; cell_i < n_cells; cell_i++) {
        is_mutated[0][word_i][cell_i] = is_known[0][word_i][cell_i] = 0;
      }
    }

    for (uint32_t gene_i = 0; gene_i < n_genes; gene_i++) {
      for (uint32_t cell_i = 0; cell_i < n_cells; cell_i++) {
        int entry;
        input_file >> entry;
        switch (entry) {
        case 0:
          // gene reported as unmutated
          is_known[0][cell_i / 64][gene_i] += 1 << (cell_i % 64);
          break;
        case 1:
        case 2:
          // gene reported as mutated
          is_known[0][cell_i / 64][gene_i] += 1 << (cell_i % 64);
          is_mutated[0][cell_i / 64][gene_i] += 1 << (cell_i % 64);
          break;
        case 3:
          // gene reported as unknown
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

  buffer<AncestorMatrix, 1> dm_buffer = range<1>(parameters.get_n_chains());
  buffer<float, 2> scores_buffer =
      range<2>(parameters.get_n_chains(), parameters.get_chain_length());
  {
    auto dm_ac = dm_buffer.get_access<access::mode::discard_write>();
    std::minstd_rand rng;
    rng.seed(std::random_device()());

    for (uint32_t i = 0; i < parameters.get_n_chains(); i++) {
      std::vector<uint32_t> pruefer_code =
          MutationTreeImpl::sample_random_pruefer_code(rng, n_genes);
      std::vector<uint32_t> parent_vector =
          MutationTreeImpl::pruefer_code_to_parent_vector(pruefer_code);
      dm_ac[i] = parent_vector_to_descendant_matrix(parent_vector);
    }
  }

  // Initializing the SYCL queue.
  property_list queue_properties = {property::queue::enable_profiling{}};
  queue working_queue(cpu_selector().select_device(), queue_properties);

  std::cout << "Submitting work" << std::endl;

  event work_event = working_queue.submit([&](handler &cgh) {
    auto is_mutated_ac = is_mutated_buffer.get_access<access::mode::read>(cgh);
    auto is_known_ac = is_known_buffer.get_access<access::mode::read>(cgh);
    auto dm_ac = dm_buffer.get_access<access::mode::read_write>(cgh);
    auto scores_ac = scores_buffer.get_access<access::mode::discard_write>(cgh);

    float alpha_mean = parameters.get_alpha_mean();
    float beta_mean = parameters.get_beta_mean();
    float beta_sd = parameters.get_beta_sd();

    uint64_t n_chains = parameters.get_n_chains();
    uint64_t chain_length = parameters.get_chain_length();

    cgh.parallel_for_work_group<class TreeScoringKernel>(
        range<2>(n_chains, chain_length), [=](id<2> idx) {
          CPUTreeScorer tree_scorer(alpha_mean, beta_mean, beta_sd,
                                    is_mutated_ac[0], is_known_ac[0]);
          scores_ac[idx[0]][idx[1]] = tree_scorer.logscore_tree(dm_ac[idx[0]]);
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
  uint64_t floating_point_operations = n_steps * n_nodes * (n_cells * 9 + 1);

  double steps_per_second = n_steps / runtime;
  double popcounts_per_second = popcounted_words / runtime;
  double counted_bits_per_second = (popcounted_words * n_genes + 1) / runtime;
  double flops = floating_point_operations / runtime;

  std::cout << "Performance:" << std::endl;
  std::cout << steps_per_second * 1e-3 << " thousand steps per second."
            << std::endl;
  std::cout << popcounts_per_second * 1e-9 << " billion popcounts per second."
            << std::endl;
  std::cout << floating_point_operations * 1e-9 << " billion floating point operations."
            << std::endl;
  std::cout << flops * 1e-9 << " GFLOPS." << std::endl;

  return 0;
}
