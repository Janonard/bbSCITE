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
using ScorerImpl = ffSCITE::TreeScorer<n_cells, n_genes,
                                       cl::sycl::access::target::host_buffer>;
using MutationTreeImpl = ScorerImpl::MutationTreeImpl;
using AncestorMatrix = MutationTreeImpl::AncestorMatrix;
using DataEntry = ScorerImpl::DataEntry;
using DataMatrix = ScorerImpl::DataMatrix;
using OccurrenceMatrix = ScorerImpl::OccurrenceMatrix;

constexpr float alpha = 0.01, beta = 0.5, prior_sd = 0.1;

void run_with_scorer(
    std::function<void(MutationTreeImpl, ScorerImpl, OccurrenceMatrix)>
        function) {
  // Mutation tree:
  //
  //  ┌4┐
  // ┌2┐3
  // 0 1
  AncestorMatrix am =
      MutationTreeImpl::parent_vector_to_ancestor_matrix({2, 2, 4, 4, 4});
  MutationTreeImpl tree(am, 4, beta);

  OccurrenceMatrix occurrences(0);

  cl::sycl::buffer<DataEntry, 2> data_buffer(
      cl::sycl::range<2>(n_cells, n_genes));
  {
    auto data = data_buffer.get_access<cl::sycl::access::mode::discard_write>();

    // cell 0, attached to node 4 (root)
    data[0][0] = 0;
    occurrences[{0, 0}]++;
    data[0][1] = 0;
    occurrences[{0, 0}]++;
    data[0][2] = 0;
    occurrences[{0, 0}]++;
    data[0][3] = 0;
    occurrences[{0, 0}]++;

    // cell 1, attached to node 1
    data[1][0] = 0;
    occurrences[{0, 0}]++;
    data[1][1] = 1;
    occurrences[{1, 1}]++;
    data[1][2] = 1;
    occurrences[{1, 1}]++;
    data[1][3] = 0;
    occurrences[{0, 0}]++;

    // cell 2, attached to node 1, with missing data
    data[2][0] = 0;
    occurrences[{0, 0}]++;
    data[2][1] = 1;
    occurrences[{1, 1}]++;
    data[2][2] = 2;
    occurrences[{2, 1}]++;
    data[2][3] = 2;
    occurrences[{2, 0}]++;

    // cell 3, attached to node 4, with missing data
    data[3][0] = 2;
    occurrences[{2, 0}]++;
    data[3][1] = 2;
    occurrences[{2, 0}]++;
    data[3][2] = 0;
    occurrences[{0, 0}]++;
    data[3][3] = 0;
    occurrences[{0, 0}]++;

    // cell 4, attached to node 0, with false negatives
    data[4][0] = 1;
    occurrences[{1, 1}]++;
    data[4][1] = 0;
    occurrences[{0, 0}]++;
    data[4][2] = 0; // Error in this position
    occurrences[{0, 1}]++;
    data[4][3] = 0;
    occurrences[{0, 0}]++;

    // cell 5, attached to node 3, with false positive
    data[5][0] = 1; // Error in this position
    occurrences[{1, 0}]++;
    data[5][1] = 0;
    occurrences[{0, 0}]++;
    data[5][2] = 0;
    occurrences[{0, 0}]++;
    data[5][3] = 1;
    occurrences[{1, 1}]++;
  }

  auto data_ac = data_buffer.get_access<cl::sycl::access::mode::read>();
  DataMatrix data;
  ScorerImpl scorer(alpha, beta, prior_sd, data_ac, data);

  function(tree, scorer, occurrences);
}

TEST_CASE("TreeScorer::get_logscore_of_occurrences", "[TreeScorer]") {
  run_with_scorer([](MutationTreeImpl tree, ScorerImpl scorer,
                     OccurrenceMatrix true_occurrences) {
    OccurrenceMatrix occurrences(0);

    occurrences[{0, 0}] = 2;
    REQUIRE(scorer.get_logscore_of_occurrences(occurrences) ==
            Approx(2 * std::log(1.0 - alpha)));

    occurrences[{0, 0}] = 0;
    occurrences[{1, 0}] = 2;
    REQUIRE(scorer.get_logscore_of_occurrences(occurrences) ==
            Approx(2 * std::log(alpha)));

    occurrences[{1, 0}] = 0;
    occurrences[{2, 0}] = 2;
    REQUIRE(scorer.get_logscore_of_occurrences(occurrences) == 0.0);

    occurrences[{2, 0}] = 0;
    occurrences[{0, 1}] = 2;
    REQUIRE(scorer.get_logscore_of_occurrences(occurrences) ==
            Approx(2 * std::log(beta)));

    occurrences[{0, 1}] = 0;
    occurrences[{1, 1}] = 2;
    REQUIRE(scorer.get_logscore_of_occurrences(occurrences) ==
            Approx(2 * std::log(1 - beta)));

    occurrences[{1, 1}] = 0;
    occurrences[{2, 1}] = 2;
    REQUIRE(scorer.get_logscore_of_occurrences(occurrences) == 0.0);
  });
}

TEST_CASE("TreeScorer::logscore_tree", "[TreeScorer]") {
  run_with_scorer([](MutationTreeImpl tree, ScorerImpl scorer,
                     OccurrenceMatrix true_occurrences) {
    float score = scorer.logscore_tree(tree);
    float beta_score = scorer.logscore_beta(tree.get_beta());

    float true_score = 13 * std::log(1 - alpha) + 5 * std::log(1 - beta) +
                       std::log(beta) + std::log(alpha);
    REQUIRE(score - beta_score == Approx(true_score));
  });
}