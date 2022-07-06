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
#include "StateScorer.hpp"
#include <catch2/catch_all.hpp>

constexpr uint32_t n_cells = 6;
constexpr uint32_t n_genes = 4;
using ScorerImpl = ffSCITE::StateScorer<n_cells, n_genes,
                                        cl::sycl::access::target::host_buffer>;
using ParentVectorImpl = ScorerImpl::ParentVectorImpl;
using AncestorMatrixImpl = ScorerImpl::AncestorMatrixImpl;
using ChainStateImpl = ScorerImpl::ChainStateImpl;
using DataEntry = ScorerImpl::DataEntry;
using OccurrenceMatrix = ScorerImpl::OccurrenceMatrix;

constexpr double alpha = 0.01, beta = 0.5, prior_sd = 0.1;

void run_with_scorer(std::function<void(ChainStateImpl, ScorerImpl)> function) {
  // Mutation tree:
  //
  //  ┌4┐
  // ┌2┐3
  // 0 1
  ParentVectorImpl pv = ParentVectorImpl::from_pruefer_code({2, 2, 4});

  ChainStateImpl state(pv, beta);

  cl::sycl::buffer<DataEntry, 2> data_buffer(
      cl::sycl::range<2>(n_cells, n_genes));
  {
    auto data = data_buffer.get_access<cl::sycl::access::mode::discard_write>();

    // cell 0, attached to node 4 (root)
    data[0][0] = 0;
    data[0][1] = 0;
    data[0][2] = 0;
    data[0][3] = 0;

    // cell 1, attached to node 1
    data[1][0] = 0;
    data[1][1] = 1;
    data[1][2] = 1;
    data[1][3] = 0;

    // cell 2, attached to node 1, with missing data
    data[2][0] = 0;
    data[2][1] = 1;
    data[2][2] = 2;
    data[2][3] = 2;

    // cell 3, attached to node 4, with missing data
    data[3][0] = 2;
    data[3][1] = 2;
    data[3][2] = 0;
    data[3][3] = 0;

    // cell 4, attached to node 0, with false negatives
    data[4][0] = 1;
    data[4][1] = 0;
    data[4][2] = 0; // Error in this position
    data[4][3] = 0;

    // cell 5, attached to node 3, with false positive
    data[5][0] = 1; // Error in this position
    data[5][1] = 0;
    data[5][2] = 0;
    data[5][3] = 1;
  }

  auto data = data_buffer.get_access<cl::sycl::access::mode::read>();
  ScorerImpl scorer(alpha, beta, prior_sd, data);

  function(state, scorer);
}

TEST_CASE("StateScorer::get_logscore_of_occurrences", "[StateScorer]") {
  run_with_scorer([](ChainStateImpl state, ScorerImpl scorer) {
    OccurrenceMatrix occurrences(0);

    occurrences[{0, 0}] = 2;
    REQUIRE(scorer.get_logscore_of_occurrences(occurrences) == 2 * std::log(1.0 - alpha));

    occurrences[{0, 0}] = 0;
    occurrences[{1, 0}] = 2;
    REQUIRE(scorer.get_logscore_of_occurrences(occurrences) == 2 * std::log(alpha));

    occurrences[{1, 0}] = 0;
    occurrences[{2, 0}] = 2;
    REQUIRE(scorer.get_logscore_of_occurrences(occurrences) == 0.0);

    occurrences[{2, 0}] = 0;
    occurrences[{0, 1}] = 2;
    REQUIRE(scorer.get_logscore_of_occurrences(occurrences) == 2 * std::log(beta));

    occurrences[{0, 1}] = 0;
    occurrences[{1, 1}] = 2;
    REQUIRE(scorer.get_logscore_of_occurrences(occurrences) == 2 * std::log(1 - beta));

    occurrences[{1, 1}] = 0;
    occurrences[{2, 1}] = 2;
    REQUIRE(scorer.get_logscore_of_occurrences(occurrences) == 0.0);
  });
}

TEST_CASE("StateScorer::get_best_attachment", "[StateScorer]") {
  run_with_scorer([](ChainStateImpl state, ScorerImpl scorer) {
    AncestorMatrixImpl am(state.mutation_tree);

    // Node 0

    ScorerImpl::Attachment attachment = scorer.get_best_attachment(0, am);
    REQUIRE(int(attachment.node_i) == 4);

    OccurrenceMatrix occurrences = attachment.occurrences;
    REQUIRE(int(occurrences[{0, 0}]) == 4);
    REQUIRE(int(occurrences[{1, 0}]) == 0);
    REQUIRE(int(occurrences[{2, 0}]) == 0);
    REQUIRE(int(occurrences[{0, 1}]) == 0);
    REQUIRE(int(occurrences[{1, 1}]) == 0);
    REQUIRE(int(occurrences[{2, 1}]) == 0);

    REQUIRE(attachment.logscore == 4 * std::log(1 - alpha));

    // Node 1

    attachment = scorer.get_best_attachment(1, am);
    REQUIRE(int(attachment.node_i) == 1);

    occurrences = attachment.occurrences;
    REQUIRE(int(occurrences[{0, 0}]) == 2);
    REQUIRE(int(occurrences[{1, 0}]) == 0);
    REQUIRE(int(occurrences[{2, 0}]) == 0);
    REQUIRE(int(occurrences[{0, 1}]) == 0);
    REQUIRE(int(occurrences[{1, 1}]) == 2);
    REQUIRE(int(occurrences[{2, 1}]) == 0);

    REQUIRE(attachment.logscore ==
            2 * std::log(1 - alpha) + 2 * std::log(1 - beta));

    // Node 2

    attachment = scorer.get_best_attachment(2, am);
    REQUIRE(int(attachment.node_i) == 1);

    occurrences = attachment.occurrences;
    REQUIRE(int(occurrences[{0, 0}]) == 1);
    REQUIRE(int(occurrences[{1, 0}]) == 0);
    REQUIRE(int(occurrences[{2, 0}]) == 1);
    REQUIRE(int(occurrences[{0, 1}]) == 0);
    REQUIRE(int(occurrences[{1, 1}]) == 1);
    REQUIRE(int(occurrences[{2, 1}]) == 1);

    REQUIRE(attachment.logscore == std::log(1 - alpha) + std::log(1 - beta));

    // Node 3

    attachment = scorer.get_best_attachment(3, am);
    REQUIRE(int(attachment.node_i) == 4);

    occurrences = attachment.occurrences;
    REQUIRE(int(occurrences[{0, 0}]) == 2);
    REQUIRE(int(occurrences[{1, 0}]) == 0);
    REQUIRE(int(occurrences[{2, 0}]) == 2);
    REQUIRE(int(occurrences[{0, 1}]) == 0);
    REQUIRE(int(occurrences[{1, 1}]) == 0);
    REQUIRE(int(occurrences[{2, 1}]) == 0);

    REQUIRE(attachment.logscore == 2 * std::log(1 - alpha));

    // Node 4

    attachment = scorer.get_best_attachment(4, am);
    REQUIRE(int(attachment.node_i) == 0);

    occurrences = attachment.occurrences;
    REQUIRE(int(occurrences[{0, 0}]) == 2);
    REQUIRE(int(occurrences[{1, 0}]) == 0);
    REQUIRE(int(occurrences[{2, 0}]) == 0);
    REQUIRE(int(occurrences[{0, 1}]) == 1);
    REQUIRE(int(occurrences[{1, 1}]) == 1);
    REQUIRE(int(occurrences[{2, 1}]) == 0);

    REQUIRE(attachment.logscore ==
            2 * std::log(1 - alpha) + std::log(beta) + std::log(1 - beta));

    // Node 5

    attachment = scorer.get_best_attachment(5, am);
    REQUIRE(int(attachment.node_i) == 3);

    occurrences = attachment.occurrences;
    REQUIRE(int(occurrences[{0, 0}]) == 2);
    REQUIRE(int(occurrences[{1, 0}]) == 1);
    REQUIRE(int(occurrences[{2, 0}]) == 0);
    REQUIRE(int(occurrences[{0, 1}]) == 0);
    REQUIRE(int(occurrences[{1, 1}]) == 1);
    REQUIRE(int(occurrences[{2, 1}]) == 0);

    REQUIRE(attachment.logscore == 2 * std::log(1 - alpha) + std::log(alpha) + std::log(1 - beta));
  });
}

TEST_CASE("StateScorer::score_state", "[StateScorer]") {
  run_with_scorer([](ChainStateImpl state, ScorerImpl scorer) {
    double score = scorer.logscore_state(state);
    double beta_score = scorer.logscore_beta(state.beta);

    double true_score = 13 * std::log(1 - alpha) + 5 * std::log(1 - beta) +
                        std::log(beta) + std::log(alpha);
    REQUIRE(score - beta_score == true_score);
  });
}