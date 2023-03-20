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
#include <catch2/catch_all.hpp>
#include <ext/intel/fpga_extensions.hpp>

using namespace ffSCITE;

constexpr uint32_t n_cells = 15;
constexpr uint32_t n_genes = 4;
constexpr uint32_t pipeline_capacity = 1;

constexpr float alpha = 6.04e-5, beta = 0.25, beta_sd = 0.1;
constexpr unsigned long n_chains = 3;
constexpr unsigned long chain_length = 50000;

using ApplicationImpl = Application<32, 31, pipeline_capacity>;
using AncestryVector = ApplicationImpl::AncestryVector;
using AncestorMatrix = ApplicationImpl::AncestorMatrix;
using MutationTreeImpl = ApplicationImpl::MutationTreeImpl;
using MutationDataMatrix = ApplicationImpl::MutationDataMatrix;
using HostTreeScorerImpl = ApplicationImpl::HostTreeScorerImpl;

TEST_CASE("Application::run_simulation()", "[Application]") {
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
  auto matrix_tuple =
      MutationTreeImpl::parent_vector_to_matrix({2, 2, 4, 4, 4});
  AncestorMatrix am = std::get<0>(matrix_tuple);
  AncestorMatrix dm = std::get<1>(matrix_tuple);
  MutationTreeImpl correct_tree(am, dm, 4, beta);

  cl::sycl::buffer<AncestryVector, 1> is_mutated_buffer =
      cl::sycl::range<1>(n_cells);
  cl::sycl::buffer<AncestryVector, 1> is_known_buffer =
      cl::sycl::range<1>(n_cells);
  {
    auto is_mutated =
        is_mutated_buffer.get_access<cl::sycl::access::mode::discard_write>();
    auto is_known =
        is_known_buffer.get_access<cl::sycl::access::mode::discard_write>();

    // node 0
    is_mutated[0] = 0b0101;
    is_mutated[1] = 0b0101;
    is_mutated[2] = 0b0001; // error
    is_known[0] = 0b1111;
    is_known[1] = 0b1111;
    is_known[2] = 0b1111;

    // node 1
    is_mutated[3] = 0b0110;
    is_mutated[4] = 0b0010; // error
    is_mutated[5] = 0b0110;
    is_known[3] = 0b1111;
    is_known[4] = 0b1111;
    is_known[5] = 0b1111;

    // node 2
    is_mutated[6] = 0b0100;
    is_mutated[7] = 0b0100;
    is_mutated[8] = 0b0100;
    is_known[6] = 0b1111;
    is_known[7] = 0b1111;
    is_known[8] = 0b1111;

    // node 3
    is_mutated[9] = 0b1000;
    is_mutated[10] = 0b1000;
    is_mutated[11] = 0b1000;
    is_known[9] = 0b1111;
    is_known[10] = 0b1111;
    is_known[11] = 0b1111;

    // node 4
    is_mutated[12] = 0b0000;
    is_mutated[13] = 0b0000;
    is_mutated[14] = 0b0000;
    is_known[12] = 0b1111;
    is_known[13] = 0b1111;
    is_known[14] = 0b1111;
  }

  cl::sycl::device device =
      cl::sycl::ext::intel::fpga_emulator_selector().select_device();
  cl::sycl::property_list queue_properties = {
      cl::sycl::property::queue::enable_profiling{}};
  cl::sycl::queue working_queue(device, queue_properties);

  Parameters parameters;
  parameters.set_alpha_mean(alpha);
  parameters.set_beta_mean(beta);
  parameters.set_beta_sd(beta_sd);
  parameters.set_n_chains(n_chains);
  parameters.set_chain_length(chain_length);

  ApplicationImpl app(is_mutated_buffer, is_known_buffer, working_queue,
                      parameters, n_cells, n_genes);
  app.run_simulation();

  AncestorMatrix best_am = app.get_best_am();
  AncestorMatrix best_dm = app.get_best_dm();
  float best_beta = app.get_best_beta();
  float best_score = app.get_best_score();
  MutationTreeImpl best_tree(best_am, best_dm, n_genes, best_beta);

  typename HostTreeScorerImpl::Parameters scorer_params(alpha, beta, beta_sd, n_cells, n_genes);
  MutationDataMatrix is_mutated, is_known;
  HostTreeScorerImpl host_scorer(
      scorer_params,
      is_mutated_buffer.get_access<cl::sycl::access::mode::read>(),
      is_known_buffer.get_access<cl::sycl::access::mode::read>(), is_mutated,
      is_known);
  float correct_score = host_scorer.logscore_tree(correct_tree);
  float found_score = host_scorer.logscore_tree(best_tree);

  REQUIRE(found_score == Catch::Approx(correct_score));
  REQUIRE(found_score == Catch::Approx(best_score));
}