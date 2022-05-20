#include "StateScorer.hpp"
#include <catch2/catch_all.hpp>

constexpr uint64_t n_cells = 4;
constexpr uint64_t n_genes = 4;
using ScorerImpl = ffSCITE::StateScorer<n_cells, n_genes>;
using uindex_node_t = ScorerImpl::uindex_node_t;
using ParentVectorImpl = ScorerImpl::ParentVectorImpl;
using AncestorMatrixImpl = ScorerImpl::AncestorMatrixImpl;
using ChainStateImpl = ScorerImpl::ChainStateImpl;
using MutationDataMatrix = ScorerImpl::MutationDataMatrix;
using OccurrenceMatrix = ScorerImpl::OccurrenceMatrix;

constexpr double alpha = 6.04e-5, beta = 0.4309, prior_sd = 0.1;

struct StateScorerScenario {
  static StateScorerScenario build_scenario() {
    // Mutation tree:
    //
    //  ┌4┐
    // ┌2┐3
    // 0 1
    ParentVectorImpl pv = ParentVectorImpl::from_pruefer_code({2, 2, 4});

    ChainStateImpl state(pv, beta);

    MutationDataMatrix data;
    // cell 0, attached to node 4 (root)
    data[{0, 0}] = 0;
    data[{0, 1}] = 0;
    data[{0, 2}] = 0;
    data[{0, 3}] = 0;

    // cell 1, attached to node 1
    data[{1, 0}] = 0;
    data[{1, 1}] = 1;
    data[{1, 2}] = 1;
    data[{1, 3}] = 0;

    // cell 2, attached to node 1, with missing data
    data[{2, 0}] = 0;
    data[{2, 1}] = 1;
    data[{2, 2}] = 2;
    data[{2, 3}] = 2;

    // cell 3, attached to node 4, with missing data
    data[{3, 0}] = 2;
    data[{3, 1}] = 2;
    data[{3, 2}] = 0;
    data[{3, 3}] = 0;

    ScorerImpl scorer(alpha, beta, prior_sd, n_cells, n_genes, data);

    return StateScorerScenario{scorer, state};
  }

  ScorerImpl scorer;
  ScorerImpl::ChainStateImpl state;
};

TEST_CASE("StateScorer::get_logscore_of_occurrences", "[StateScorer]") {
  auto scenario = StateScorerScenario::build_scenario();

  OccurrenceMatrix occurrences(0);

  occurrences[{0, 0}] = 2;
  REQUIRE(scenario.scorer.get_logscore_of_occurrences(occurrences) ==
          2 * std::log(1.0 - alpha));

  occurrences[{0, 0}] = 0;
  occurrences[{1, 0}] = 2;
  REQUIRE(scenario.scorer.get_logscore_of_occurrences(occurrences) ==
          2 * std::log(alpha));

  occurrences[{1, 0}] = 0;
  occurrences[{2, 0}] = 2;
  REQUIRE(scenario.scorer.get_logscore_of_occurrences(occurrences) == 0.0);

  occurrences[{2, 0}] = 0;
  occurrences[{0, 1}] = 2;
  REQUIRE(scenario.scorer.get_logscore_of_occurrences(occurrences) ==
          2 * std::log(beta));

  occurrences[{0, 1}] = 0;
  occurrences[{1, 1}] = 2;
  REQUIRE(scenario.scorer.get_logscore_of_occurrences(occurrences) ==
          2 * std::log(1 - beta));

  occurrences[{1, 1}] = 0;
  occurrences[{2, 1}] = 2;
  REQUIRE(scenario.scorer.get_logscore_of_occurrences(occurrences) == 0.0);
}

TEST_CASE("StateScorer::get_best_attachment", "[StateScorer]") {
  auto scenario = StateScorerScenario::build_scenario();

  AncestorMatrixImpl am(scenario.state.mutation_tree);

  // Node 0

  std::tuple<uindex_node_t, OccurrenceMatrix, double> attachment =
      scenario.scorer.get_best_attachment(0, am);
  REQUIRE(std::get<0>(attachment) == 4);

  OccurrenceMatrix occurrences = std::get<1>(attachment);
  REQUIRE(int(occurrences[{0, 0}]) == 4);
  REQUIRE(int(occurrences[{1, 0}]) == 0);
  REQUIRE(int(occurrences[{2, 0}]) == 0);
  REQUIRE(int(occurrences[{0, 1}]) == 0);
  REQUIRE(int(occurrences[{1, 1}]) == 0);
  REQUIRE(int(occurrences[{2, 1}]) == 0);

  REQUIRE(std::get<2>(attachment) == 4 * std::log(1 - alpha));

  // Node 1

  attachment = scenario.scorer.get_best_attachment(1, am);
  REQUIRE(std::get<0>(attachment) == 1);

  occurrences = std::get<1>(attachment);
  REQUIRE(int(occurrences[{0, 0}]) == 2);
  REQUIRE(int(occurrences[{1, 0}]) == 0);
  REQUIRE(int(occurrences[{2, 0}]) == 0);
  REQUIRE(int(occurrences[{0, 1}]) == 0);
  REQUIRE(int(occurrences[{1, 1}]) == 2);
  REQUIRE(int(occurrences[{2, 1}]) == 0);

  REQUIRE(std::get<2>(attachment) ==
          2 * std::log(1 - alpha) + 2 * std::log(1 - beta));

  // Node 2

  attachment = scenario.scorer.get_best_attachment(2, am);
  REQUIRE(std::get<0>(attachment) == 1);

  occurrences = std::get<1>(attachment);
  REQUIRE(int(occurrences[{0, 0}]) == 1);
  REQUIRE(int(occurrences[{1, 0}]) == 0);
  REQUIRE(int(occurrences[{2, 0}]) == 1);
  REQUIRE(int(occurrences[{0, 1}]) == 0);
  REQUIRE(int(occurrences[{1, 1}]) == 1);
  REQUIRE(int(occurrences[{2, 1}]) == 1);

  REQUIRE(std::get<2>(attachment) == std::log(1 - alpha) + std::log(1 - beta));

  // Node 3

  attachment = scenario.scorer.get_best_attachment(3, am);
  REQUIRE(std::get<0>(attachment) == 4);

  occurrences = std::get<1>(attachment);
  REQUIRE(int(occurrences[{0, 0}]) == 2);
  REQUIRE(int(occurrences[{1, 0}]) == 0);
  REQUIRE(int(occurrences[{2, 0}]) == 2);
  REQUIRE(int(occurrences[{0, 1}]) == 0);
  REQUIRE(int(occurrences[{1, 1}]) == 0);
  REQUIRE(int(occurrences[{2, 1}]) == 0);

  REQUIRE(std::get<2>(attachment) == 2 * std::log(1 - alpha));
}

TEST_CASE("StateScorer::score_state", "[StateScorer]") {
  auto scenario = StateScorerScenario::build_scenario();

  double score = scenario.scorer.logscore_state(scenario.state);
  REQUIRE(std::isfinite(score));
  bool is_not_nan = std::numeric_limits<double>::is_iec559 ? (score == score) : !std::isnan(score);
  REQUIRE(is_not_nan); // NaN test
}