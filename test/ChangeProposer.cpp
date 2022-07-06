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
#include "ChangeProposer.hpp"
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <catch2/catch_all.hpp>
#include <map>
#include <set>

using namespace ffSCITE;

constexpr unsigned int n_iterations = 1024;
constexpr double alpha = 0.05;
constexpr uint64_t n_genes = 7;
constexpr uint64_t n_nodes = 8;

using ProposerImpl = ChangeProposer<n_genes, oneapi::dpl::minstd_rand>;

ProposerImpl init_proposer() {
  std::random_device seeder;

  oneapi::dpl::minstd_rand twister;
  twister.seed(seeder());

  ProposerImpl proposer(twister);

  return proposer;
}

/*
 * These unit tests do not prove that the tested methods are correct. Instead,
 * they only check whether their behavior is plausible. We chose to not correct
 * this since we needed to prioritize the optimization of the application. For
 * more information, check out Issue 15
 * (https://git.uni-paderborn.de/joo/ffscite/-/issues/15).
 */

TEST_CASE("ChangeProposer::sample_nonroot_nodepair", "[ChangeProposer]") {
  auto proposer = init_proposer();

  std::map<std::array<uint64_t, 2>, unsigned int> sampled_nodes;

  for (uint64_t i = 0; i < n_iterations; i++) {
    auto pair = proposer.sample_nonroot_nodepair(n_nodes);
    REQUIRE(pair[0] < n_nodes - 1);
    REQUIRE(pair[1] < n_nodes - 1);
    REQUIRE(pair[0] != pair[1]);

    if (pair[0] > pair[1]) {
      std::swap(pair[0], pair[1]);
    }

    if (sampled_nodes.contains(pair)) {
      sampled_nodes[pair] += 1;
    } else {
      sampled_nodes[pair] = 1;
    }
  }

  // We want to test that the samples are uniformly distributed among all
  // possible node pairs, i.e. subsets of the nodesets with a cardinality of
  // two. We therefore view `sample_nonroot_nodepair` as a random variable that
  // samples from all those subsets. Our null-hypothesis is that this random
  // variable is uniformly distributed and we test this hypothesis with a
  // chi-squared-test.
  double t = 0;
  double n_pairs = boost::math::binomial_coefficient<double>(n_nodes - 1, 2);
  for (uint64_t i = 0; i < n_nodes - 1; i++) {
    for (uint64_t j = i + 1; j < n_nodes - 1; j++) {
      double n_occurrences;
      if (sampled_nodes.contains({i, j})) {
        n_occurrences = sampled_nodes[{i, j}];
      } else {
        n_occurrences = 0.0;
      }

      double numerator = n_occurrences - n_iterations / n_pairs;
      numerator *= numerator;
      double denominator = n_iterations / n_pairs;
      t += numerator / denominator;
    }
  }

  boost::math::chi_squared chi_squared_dist(n_pairs - 1);
  REQUIRE(t < quantile(chi_squared_dist, 1 - alpha));
}

TEST_CASE("ChangeProposer::sample_descendant_or_nondescendant",
          "[ChangeProposer]") {
  auto proposer = init_proposer();

  /*
   * Original tree:
   *
   *   ┌-7-┐
   *  ┌5┐ ┌6
   * ┌2┐3 4
   * 0 1
   */
  auto pv = ParentVector<n_nodes>::from_pruefer_code({2, 2, 5, 5, 6, 7});
  AncestorMatrix<n_nodes> am(pv);

  // We count how often each descendant of five is picked to run a hypthesis
  // test.
  std::map<unsigned int, unsigned int> sampled_nodes_for_five;
  sampled_nodes_for_five[0] = 0;
  sampled_nodes_for_five[1] = 0;
  sampled_nodes_for_five[2] = 0;
  sampled_nodes_for_five[3] = 0;
  sampled_nodes_for_five[5] = 0;

  for (uint64_t i = 0; i < n_iterations; i++) {
    auto sampled_node =
        proposer.sample_descendant_or_nondescendant(am, 5, true, false);
    REQUIRE(am.is_ancestor(5, sampled_node));
    sampled_nodes_for_five[sampled_node]++;

    sampled_node =
        proposer.sample_descendant_or_nondescendant(am, 5, false, true);
    REQUIRE(!am.is_ancestor(5, sampled_node));

    sampled_node =
        proposer.sample_descendant_or_nondescendant(am, 6, true, false);
    REQUIRE(am.is_ancestor(6, sampled_node));

    sampled_node =
        proposer.sample_descendant_or_nondescendant(am, 6, false, false);
    REQUIRE(!am.is_ancestor(6, sampled_node));
    REQUIRE(sampled_node != n_nodes - 1);
  }

  // We want to test that the samples are uniformly distributed among all
  // descendants of five. We therefore view `sample_descendant_or_nondescendant`
  // as a random variable that samples from the descendants of a node. Our
  // null-hypothesis is that X_v is uniformly distributed, i.e. \forall w \in
  // Desc(v): P(X_v = w) = (|Desc(v)|)^{-1} and we test this hypothesis with a
  // chi-squared-test.
  double t = 0;
  const double n_descendants = am.get_n_descendants(5);
  for (std::pair<unsigned int, unsigned int> pair : sampled_nodes_for_five) {
    double numerator = (pair.second - n_iterations / n_descendants);
    numerator *= numerator;
    double denominator = n_iterations / n_descendants;
    t += numerator / denominator;
  }

  boost::math::chi_squared chi_squared_dist(5 - 1);
  REQUIRE(t < quantile(chi_squared_dist, 1 - alpha));
}

TEST_CASE("ChangeProposer::change_beta", "[ChangeProposer]") {
  auto proposer = init_proposer();

  for (uint64_t i = 0; i < n_iterations; i++) {
    double sampled_beta = proposer.change_beta(0.0);
    REQUIRE(sampled_beta >= 0.0);
    REQUIRE(sampled_beta <= 1.0);

    sampled_beta = proposer.change_beta(0.5);
    REQUIRE(sampled_beta >= 0.0);
    REQUIRE(sampled_beta <= 1.0);

    sampled_beta = proposer.change_beta(1.0);
    REQUIRE(sampled_beta >= 0.0);
    REQUIRE(sampled_beta <= 1.0);
  }
}

TEST_CASE("ChangeProposer::prune_and_reattach", "[ChangeProposer]") {
  auto proposer = init_proposer();

  /*
   * Original tree:
   *
   *   ┌-7-┐
   *  ┌5┐ ┌6
   * ┌2┐3 4
   * 0 1
   */
  auto pv = ParentVector<n_nodes>::from_pruefer_code({2, 2, 5, 5, 6, 7});
  AncestorMatrix<n_nodes> am(pv);

  for (uint64_t i = 0; i < n_iterations; i++) {
    ParentVector<n_nodes> pv_copy(pv);
    unsigned int moved_node_i = proposer.prune_and_reattach(pv_copy, am);

    // Simple sanity checks for the move.
    REQUIRE(moved_node_i < n_nodes);
    REQUIRE(pv_copy[moved_node_i] != moved_node_i);

    // Check that the node has not been attached to one of it's previous
    // descendants.
    REQUIRE(!am.is_ancestor(moved_node_i, pv_copy[moved_node_i]));

    // Check that nothing else was changed.
    for (uint64_t node_i = 0; node_i < n_nodes; node_i++) {
      if (node_i != moved_node_i) {
        REQUIRE(pv_copy[node_i] == pv[node_i]);
      }
    }
  }
}

TEST_CASE("ChangeProposer::swap_subtrees", "[ChangeProposer]") {
  auto proposer = init_proposer();

  /*
   * Original tree:
   *
   *   ┌-7-┐
   *  ┌5┐ ┌6
   * ┌2┐3 4
   * 0 1
   */
  auto pv = ParentVector<n_nodes>::from_pruefer_code({2, 2, 5, 5, 6, 7});
  AncestorMatrix<n_nodes> am(pv);

  for (uint64_t i = 0; i < n_iterations; i++) {
    ParentVector<n_nodes> pv_copy(pv);
    double neighborhood_correction = 1.0;
    auto swapped_subtrees =
        proposer.swap_subtrees(pv_copy, am, neighborhood_correction);

    unsigned int node_a_i = swapped_subtrees[0];
    unsigned int node_b_i = swapped_subtrees[1];

    bool common_lineage = am.is_ancestor(node_a_i, node_b_i) ||
                          am.is_ancestor(node_b_i, node_a_i);

    // Check soundness of the change.
    if (common_lineage) {
      // Ensure that node_a_i is the lower node.
      if (am.is_ancestor(node_a_i, node_b_i)) {
        std::swap(node_a_i, node_b_i);
      }

      REQUIRE(pv_copy[node_a_i] == pv[node_b_i]);
      REQUIRE(am.is_ancestor(node_a_i, pv_copy[node_b_i]));
      REQUIRE(neighborhood_correction ==
              double(am.get_n_descendants(node_a_i)) /
                  double(am.get_n_descendants(node_b_i)));
    } else {
      REQUIRE(pv_copy[node_a_i] == pv[node_b_i]);
      REQUIRE(pv_copy[node_b_i] == pv[node_a_i]);
      REQUIRE(neighborhood_correction == 1.0);
    }

    // Check that nothing else was changed.
    for (uint64_t i = 0; i < n_nodes; i++) {
      if (i != node_a_i && i != node_b_i) {
        REQUIRE(pv_copy[i] == pv[i]);
      }
    }
  }
}