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
#include <catch2/catch_all.hpp>

using namespace Catch;

constexpr uint32_t n_cells = 6;
constexpr uint32_t n_genes = 4;
using ScorerImpl =
    ffSCITE::TreeScorer<32, 31, cl::sycl::access::target::host_buffer>;
using MutationTreeImpl = ScorerImpl::MutationTreeImpl;
using AncestryVector = MutationTreeImpl::AncestryVector;
using AncestorMatrix = MutationTreeImpl::AncestorMatrix;
using MutationDataMatrix = ScorerImpl::MutationDataMatrix;

constexpr float alpha = 0.01, beta = 0.5, prior_sd = 0.1;

TEST_CASE("TreeScorer::logscore_tree", "[TreeScorer]") {
  // Mutation tree:
  //
  //  ┌4┐
  // ┌2┐3
  // 0 1
  AncestorMatrix am =
      MutationTreeImpl::parent_vector_to_ancestor_matrix({2, 2, 4, 4, 4});
  MutationTreeImpl tree(am, 4, beta);

  cl::sycl::buffer<AncestryVector, 1> is_mutated_buffer(
      (cl::sycl::range<1>(n_cells)));
  cl::sycl::buffer<AncestryVector, 1> is_known_buffer(
      (cl::sycl::range<1>(n_cells)));
  {
    auto is_mutated =
        is_mutated_buffer.get_access<cl::sycl::access::mode::discard_write>();
    auto is_known =
        is_known_buffer.get_access<cl::sycl::access::mode::discard_write>();

    // cell 0, attached to node 4 (root)
    is_mutated[0] = 0b00000;
    is_known[0] = 0b01111;

    // cell 1, attached to node 1
    is_mutated[1] = 0b00110;
    is_known[1] = 0b01111;

    // cell 2, attached to node 1, with missing data
    is_mutated[2] = 0b00110;
    is_known[2] = 0b00011;

    // cell 3, attached to node 4, with missing data
    is_mutated[3] = 0b00000;
    is_known[3] = 0b01100;

    // cell 4, attached to node 0, with false negative for gene 2
    is_mutated[4] = 0b00001;
    is_known[4] = 0b01111;

    // cell 5, attached to node 3, with false positive for gene 1
    is_mutated[5] = 0b01001;
    is_known[5] = 0b01111;
  }

  auto is_mutated_ac =
      is_mutated_buffer.get_access<cl::sycl::access::mode::read>();
  auto is_known_ac = is_known_buffer.get_access<cl::sycl::access::mode::read>();
  MutationDataMatrix is_mutated, is_known;
  ScorerImpl scorer(alpha, beta, prior_sd, n_cells, n_genes, is_mutated_ac,
                    is_known_ac, is_mutated, is_known);

  float score = scorer.logscore_tree(tree);
  float beta_score = scorer.logscore_beta(tree.get_beta());

  float true_score = 13 * std::log(1 - alpha) + 5 * std::log(1 - beta) +
                     std::log(beta) + std::log(alpha);
  REQUIRE(score - beta_score == Approx(true_score));
}