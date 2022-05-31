#include "MCMCKernel.hpp"
#include <catch2/catch_all.hpp>
#include <ext/intel/fpga_extensions.hpp>

using namespace ffSCITE;

constexpr uint64_t n_cells = 15;
constexpr uint64_t n_genes = 4;

constexpr double alpha = 6.04e-5, beta = 0.25, prior_sd = 0.1;
constexpr unsigned long repetitions = 10;
constexpr unsigned long chain_length = 1000000;

using ChangeProposerImpl = ChangeProposer<n_genes, oneapi::dpl::minstd_rand0>;
using StateScorerImpl = StateScorer<n_cells, n_genes>;
using MCMCKernelImpl =
    MCMCKernel<n_cells, n_genes, ChangeProposerImpl, StateScorerImpl>;
using MutationDataMatrix = StateScorerImpl::MutationDataMatrix;
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
   * There are three cells attached to every node and there are some errors in
   * the data to make it interesting.
   */
  MutationDataMatrix data;
  // node 0
  data[0][0] = 1;
  data[0][1] = 0;
  data[0][2] = 1;
  data[0][3] = 0;

  data[1][0] = 1;
  data[1][1] = 0;
  data[1][2] = 1;
  data[1][3] = 0;

  data[2][0] = 1;
  data[2][1] = 0;
  data[2][2] = 0; // error
  data[2][3] = 0;

  // node 1
  data[3][0] = 0;
  data[3][1] = 1;
  data[3][2] = 1;
  data[3][3] = 0;

  data[4][0] = 0;
  data[4][1] = 1;
  data[4][2] = 0; // error
  data[4][3] = 0;

  data[5][0] = 0;
  data[5][1] = 1;
  data[5][2] = 1;
  data[5][3] = 0;

  // node 2
  data[6][0] = 0;
  data[6][1] = 0;
  data[6][2] = 1;
  data[6][3] = 0;

  data[7][0] = 0;
  data[7][1] = 0;
  data[7][2] = 1;
  data[7][3] = 0;

  data[8][0] = 0;
  data[8][1] = 0;
  data[8][2] = 1;
  data[8][3] = 0;

  // node 3
  data[9][0] = 0;
  data[9][1] = 0;
  data[9][2] = 0;
  data[9][3] = 1;

  data[10][0] = 0;
  data[10][1] = 0;
  data[10][2] = 0;
  data[10][3] = 1;

  data[11][0] = 0;
  data[11][1] = 0;
  data[11][2] = 0;
  data[11][3] = 1;

  // node 4
  data[12][0] = 0;
  data[12][1] = 0;
  data[12][2] = 0;
  data[12][3] = 0;

  data[13][0] = 0;
  data[13][1] = 0;
  data[13][2] = 0;
  data[13][3] = 0;

  data[14][0] = 0;
  data[14][1] = 0;
  data[14][2] = 0;
  data[14][3] = 0;

  std::random_device seeder;
  oneapi::dpl::minstd_rand0 twister;
  twister.seed(seeder());

  ChangeProposerImpl change_proposer(twister);
  StateScorerImpl state_scorer(alpha, beta, prior_sd, n_cells, n_genes, data);

  cl::sycl::buffer<std::tuple<ChainStateImpl, double>, 1> best_state_buffer(
      cl::sycl::range<1>(1));
  {
    ChainStateImpl best_state = ChainStateImpl::sample_random_state(
        change_proposer.get_rng(), n_genes, beta);
    double best_score = state_scorer.logscore_state(best_state);

    auto best_state_ac =
        best_state_buffer.get_access<cl::sycl::access::mode::discard_write>();
    best_state_ac[0] = {best_state, best_score};
  }

  cl::sycl::queue working_queue(
      (cl::sycl::ext::intel::fpga_emulator_selector()));

  for (unsigned long rep_i = 0; rep_i < repetitions; rep_i++) {
    cl::sycl::buffer<std::tuple<ChainStateImpl, double>, 1>
        current_state_buffer(cl::sycl::range<1>(1));
    {
      ChainStateImpl current_state = ChainStateImpl::sample_random_state(
          change_proposer.get_rng(), n_genes, beta);
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

      MCMCKernelImpl kernel(change_proposer, state_scorer, 1.0, best_state_ac,
                            current_state_ac, chain_length);
      cgh.single_task(kernel);
    });
  }

  auto best_state_ac =
      best_state_buffer.get_access<cl::sycl::access::mode::read>();
  ChainStateImpl best_state = std::get<0>(best_state_ac[0]);

  REQUIRE(int(best_state.mutation_tree[0]) == 2);
  REQUIRE(int(best_state.mutation_tree[1]) == 2);
  REQUIRE(int(best_state.mutation_tree[2]) == 4);
  REQUIRE(int(best_state.mutation_tree[3]) == 4);
  REQUIRE(int(best_state.mutation_tree[4]) == 4);
}