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
#include "MCMCKernel.hpp"
#include <catch2/catch_all.hpp>
#include <ext/intel/fpga_extensions.hpp>

using namespace ffSCITE;

constexpr uint32_t n_cells = 15;
constexpr uint32_t n_genes = 4;

constexpr float alpha = 6.04e-5, beta = 0.25, beta_sd = 0.1;
constexpr unsigned long n_chains = 10;
constexpr unsigned long chain_length = 1000000;

using MCMCKernelImpl = MCMCKernel<32, 31, oneapi::dpl::minstd_rand>;
using AncestorMatrix = MCMCKernelImpl::AncestorMatrix;
using MutationTreeImpl = MCMCKernelImpl::MutationTreeImpl;
using MutationDataWord = MCMCKernelImpl::MutationDataWord;

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
  AncestorMatrix am =
      MutationTreeImpl::parent_vector_to_ancestor_matrix({2, 2, 4, 4, 4});
  MutationTreeImpl correct_tree(am, 4, beta);

  cl::sycl::buffer<MutationDataWord, 1> data_buffer(
      (cl::sycl::range<1>(n_cells)));
  {
    auto data = data_buffer.get_access<cl::sycl::access::mode::discard_write>();

    // node 0
    data[0] = 0b00010001;
    data[1] = 0b00010001;
    data[2] = 0b00000001; // error

    // node 1
    data[3] = 0b00010100;
    data[4] = 0b00000100; // error
    data[5] = 0b00010100;

    // node 2
    data[6] = 0b00010000;
    data[7] = 0b00010000;
    data[8] = 0b00010000;

    // node 3
    data[9] = 0b01000000;
    data[10] = 0b01000000;
    data[11] = 0b01000000;

    // node 4
    data[12] = 0b00000000;
    data[13] = 0b00000000;
    data[14] = 0b00000000;
  }

  cl::sycl::queue working_queue(
      (cl::sycl::ext::intel::fpga_emulator_selector()));

  Parameters parameters;
  parameters.set_alpha_mean(alpha);
  parameters.set_beta_mean(beta);
  parameters.set_beta_sd(beta_sd);
  parameters.set_n_chains(n_chains);
  parameters.set_chain_length(chain_length);

  auto result = MCMCKernelImpl::run_simulation(data_buffer, working_queue,
                                               parameters, n_cells, n_genes);
  std::vector<AncestorMatrix> best_am = std::get<0>(result);
  std::vector<float> best_beta = std::get<1>(result);
  MutationTreeImpl found_tree(best_am[0], n_genes, best_beta[0]);

  REQUIRE(best_am.size() >= 1);
  REQUIRE(best_am.size() == best_beta.size());

  MCMCKernelImpl::MutationDataMatrix data;
  MCMCKernelImpl::HostTreeScorerImpl host_scorer(
      alpha, beta, beta_sd, n_cells, n_genes,
      data_buffer.get_access<cl::sycl::access::mode::read>(), data);
  float correct_score = host_scorer.logscore_tree(correct_tree);
  float found_score = host_scorer.logscore_tree(found_tree);

  REQUIRE(found_score == Catch::Approx(correct_score));
}