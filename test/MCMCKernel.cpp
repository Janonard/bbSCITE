#include "MCMCKernel.hpp"
#include <catch2/catch_all.hpp>

constexpr uint64_t n_cells = 15;
constexpr uint64_t n_genes = 4;

constexpr double alpha = 6.04e-5, beta = 0.25, prior_sd = 0.1;
constexpr unsigned long repetitions = 2;
constexpr unsigned long chain_length = 450000;

using MCMCKernelImpl = ffSCITE::MCMCKernel<n_cells, n_genes, std::mt19937>;
using uindex_node_t = MCMCKernelImpl::uindex_node_t;
using uindex_cell_t = MCMCKernelImpl::uindex_cell_t;
using MutationDataMatrix =
    ffSCITE::StaticMatrix<ac_int<2, false>, n_cells, n_genes>;
using ChainStateImpl = MCMCKernelImpl::ChainStateImpl;

TEST_CASE("MCMCKernel::operator()", "[MCMCKernel]") {
  /*
   * This test runs the MCMC chain for an simple, constructed setup to see if it
   * finds it. This is not a statistical test, just a simple sanity check.
   *
   * Mutation Tree:
   *
   *  ┌4┐
   * ┌2┐3
   * 0 1
   *
   * There are two cells attached to every node, without errors.
   */
  MutationDataMatrix data;
  // node 0
  data[{0, 0}] = 1;
  data[{0, 1}] = 0;
  data[{0, 2}] = 1;
  data[{0, 3}] = 0;

  data[{1, 0}] = 1;
  data[{1, 1}] = 0;
  data[{1, 2}] = 1;
  data[{1, 3}] = 0;

  data[{2, 0}] = 0; // error
  data[{2, 1}] = 0;
  data[{2, 2}] = 1;
  data[{2, 3}] = 0;

  // node 1
  data[{3, 0}] = 0;
  data[{3, 1}] = 1;
  data[{3, 2}] = 1;
  data[{3, 3}] = 0;

  data[{4, 0}] = 0;
  data[{4, 1}] = 0; // error
  data[{4, 2}] = 1;
  data[{4, 3}] = 0;

  data[{5, 0}] = 0;
  data[{5, 1}] = 1;
  data[{5, 2}] = 1;
  data[{5, 3}] = 0;

  // node 2
  data[{6, 0}] = 0;
  data[{6, 1}] = 0;
  data[{6, 2}] = 1;
  data[{6, 3}] = 0;

  data[{7, 0}] = 0;
  data[{7, 1}] = 0;
  data[{7, 2}] = 0; // error
  data[{7, 3}] = 0;

  data[{8, 0}] = 1;
  data[{8, 1}] = 0;
  data[{8, 2}] = 1;
  data[{8, 3}] = 0;

  // node 3
  data[{9, 0}] = 0;
  data[{9, 1}] = 0;
  data[{9, 2}] = 0;
  data[{9, 3}] = 1;

  data[{10, 0}] = 0;
  data[{10, 1}] = 0;
  data[{10, 2}] = 0;
  data[{10, 3}] = 0; // error

  data[{11, 0}] = 0;
  data[{11, 1}] = 0;
  data[{11, 2}] = 0;
  data[{11, 3}] = 1;

  // node 4
  data[{13, 0}] = 0;
  data[{13, 1}] = 0;
  data[{13, 2}] = 0;
  data[{13, 3}] = 0;

  data[{14, 0}] = 0;
  data[{14, 1}] = 0;
  data[{14, 2}] = 0;
  data[{14, 3}] = 0;

  data[{15, 0}] = 0;
  data[{15, 1}] = 0;
  data[{15, 2}] = 0;
  data[{15, 3}] = 0;

  std::random_device seeder;
  std::mt19937 twister;
  twister.seed(seeder());

  MCMCKernelImpl kernel(twister, alpha, beta, prior_sd, 1, n_cells, n_genes,
                        data);

  ChainStateImpl best_state =
      ChainStateImpl::sample_random_state(kernel.get_rng(), n_genes, beta);
  double best_score = kernel.get_state_scorer().logscore_state(best_state);

  for (unsigned long rep_i = 0; rep_i < repetitions; rep_i++) {
    ChainStateImpl current_state =
        ChainStateImpl::sample_random_state(kernel.get_rng(), n_genes, beta);
    double current_score =
        kernel.get_state_scorer().logscore_state(current_state);

    for (unsigned long i = 0; i < chain_length; i++) {
      auto next_state = kernel(current_state, current_score);
      current_state = std::get<0>(next_state);
      current_score = std::get<1>(next_state);

      bool is_not_nan = std::numeric_limits<double>::is_iec559
                            ? (current_score == current_score)
                            : !std::isnan(current_score);
      REQUIRE(is_not_nan); // NaN test

      if (current_score > best_score) {
        best_state = current_state;
        best_score = current_score;

        std::cout << "Beta: " << best_state.beta << ", tree:";
        for (uindex_node_t i = 0; i < n_genes; i++) {
          std::cout << int(best_state.mutation_tree[i]) << ", ";
        }
        std::cout << "score: " << best_score << std::endl;
      }
    }
  }

  REQUIRE(int(best_state.mutation_tree[0]) == 2);
  REQUIRE(int(best_state.mutation_tree[1]) == 2);
  REQUIRE(int(best_state.mutation_tree[2]) == 4);
  REQUIRE(int(best_state.mutation_tree[3]) == 4);
  REQUIRE(int(best_state.mutation_tree[4]) == 4);
}