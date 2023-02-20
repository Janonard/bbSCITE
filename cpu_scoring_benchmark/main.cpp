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
#include "TreeScorer.hpp"
#include <CL/sycl.hpp>
#include <MutationTree.hpp>
#include <Parameters.hpp>
#include <chrono>
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

  MutationDataMatrix is_mutated, is_known;

  // Filling the vectors with zeroes.
  for (uint32_t cell_i = 0; cell_i < n_cells; cell_i++) {
    is_mutated[cell_i] = is_known[cell_i] = 0;
  }

  // Load the mutation input data.
  std::ifstream input_file(parameters.get_input_path());

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

  std::cout << "Generating random trees" << std::endl;

  std::vector<AncestorMatrix> am;
  std::vector<AncestorMatrix> dm;
  am.reserve(parameters.get_n_chains());
  dm.reserve(parameters.get_n_chains());

  std::vector<std::vector<float>> scores;
  scores.reserve(parameters.get_n_chains());

  std::minstd_rand rng;
  rng.seed(std::random_device()());

  // Initialize ancestor matrix, descendant matrix, and scores.
  for (uint32_t chain_i = 0; chain_i < parameters.get_n_chains(); chain_i++) {
    std::vector<uint32_t> pruefer_code =
        MutationTreeImpl::sample_random_pruefer_code(rng, n_genes);
    std::vector<uint32_t> parent_vector =
        MutationTreeImpl::pruefer_code_to_parent_vector(pruefer_code);
    auto am_and_dm = MutationTreeImpl::parent_vector_to_matrix(parent_vector);
    am.push_back(std::get<0>(am_and_dm));
    dm.push_back(std::get<1>(am_and_dm));

    scores.push_back(std::vector<float>());
    scores[chain_i].reserve(parameters.get_chain_length());
    for (uint32_t step_i = 0; step_i < n_cells; step_i++) {
      scores[chain_i].push_back(0);
    }
  }

  // Initializing the SYCL queue.

  std::cout << "Submitting work" << std::endl;

  TreeScorerImpl tree_scorer(
      parameters.get_alpha_mean(), parameters.get_beta_mean(),
      parameters.get_beta_sd(), n_cells, n_genes, is_mutated, is_known);

  auto start = std::chrono::high_resolution_clock::now();

  for (uint32_t chain_i = 0; chain_i < parameters.get_n_chains(); chain_i++) {
    for (uint32_t step_i = 0; step_i < parameters.get_chain_length();
         step_i++) {
      MutationTreeImpl tree(am[chain_i], dm[chain_i], n_genes,
                            parameters.get_beta_mean());
      scores[chain_i][step_i] = tree_scorer.logscore_tree(tree);
    }
  }

  auto end = std::chrono::high_resolution_clock::now();

  double runtime =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  runtime /= 1000.0;
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
