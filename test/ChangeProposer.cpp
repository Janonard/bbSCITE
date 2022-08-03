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
constexpr float alpha = 0.05;
constexpr uint32_t n_genes = 7;
constexpr uint32_t n_nodes = 8;
constexpr unsigned int ulps = 4;

using ProposerImpl = ChangeProposer<31, oneapi::dpl::minstd_rand>;
using MutationTreeImpl = MutationTree<31>;
using AncestorMatrix = MutationTreeImpl::AncestorMatrix;

ProposerImpl init_proposer(float prob_beta_change, float prob_prune_n_reattach,
                           float prob_swap_nodes) {
  std::random_device seeder;

  oneapi::dpl::minstd_rand twister;
  twister.seed(seeder());

  ProposerImpl proposer(twister, prob_beta_change, prob_prune_n_reattach,
                        prob_swap_nodes, 0.1);

  return proposer;
}

ProposerImpl init_proposer() { return init_proposer(0.0, 0.5, 0.45); }

/*
 * These unit tests do not prove that the tested methods are correct. Instead,
 * they only check whether their behavior is plausible. We chose to not correct
 * this since we needed to prioritize the optimization of the application. For
 * more information, check out Issue 15
 * (https://git.uni-paderborn.de/joo/ffscite/-/issues/15).
 */

TEST_CASE("ChangeProposer::sample_nonroot_nodepair", "[ChangeProposer]") {
  auto proposer = init_proposer();

  /*
   * Original tree:
   *
   *   ┌-7-┐
   *  ┌5┐ ┌6
   * ┌2┐3 4
   * 0 1
   */
  AncestorMatrix am = MutationTreeImpl::parent_vector_to_ancestor_matrix(
      {2, 2, 5, 5, 6, 7, 7, 7});
  MutationTreeImpl tree(am, 7, 0.42);

  std::map<std::array<uint32_t, 2>, unsigned int> sampled_nodes;

  for (uint32_t i = 0; i < n_iterations; i++) {
    auto pair = proposer.sample_nonroot_nodepair(tree);
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
  float t = 0;
  float n_pairs = boost::math::binomial_coefficient<float>(n_nodes - 1, 2);
  for (uint32_t i = 0; i < n_nodes - 1; i++) {
    for (uint32_t j = i + 1; j < n_nodes - 1; j++) {
      float n_occurrences;
      if (sampled_nodes.contains({i, j})) {
        n_occurrences = sampled_nodes[{i, j}];
      } else {
        n_occurrences = 0.0;
      }

      float numerator = n_occurrences - n_iterations / n_pairs;
      numerator *= numerator;
      float denominator = n_iterations / n_pairs;
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
  AncestorMatrix am = MutationTreeImpl::parent_vector_to_ancestor_matrix(
      {2, 2, 5, 5, 6, 7, 7, 7});
  MutationTreeImpl tree(am, 7, 0.42);

  // We count how often each descendant of five is picked to run a hypthesis
  // test.
  std::map<unsigned int, unsigned int> sampled_nodes_for_five;
  sampled_nodes_for_five[0] = 0;
  sampled_nodes_for_five[1] = 0;
  sampled_nodes_for_five[2] = 0;
  sampled_nodes_for_five[3] = 0;
  sampled_nodes_for_five[5] = 0;

  for (uint32_t i = 0; i < n_iterations; i++) {
    auto sampled_node =
        proposer.sample_descendant_or_nondescendant(tree, 5, true, false);
    REQUIRE(tree.is_ancestor(5, sampled_node));
    sampled_nodes_for_five[sampled_node]++;

    sampled_node =
        proposer.sample_descendant_or_nondescendant(tree, 5, false, true);
    REQUIRE(!tree.is_ancestor(5, sampled_node));

    sampled_node =
        proposer.sample_descendant_or_nondescendant(tree, 6, true, false);
    REQUIRE(tree.is_ancestor(6, sampled_node));

    sampled_node =
        proposer.sample_descendant_or_nondescendant(tree, 6, false, false);
    REQUIRE(!tree.is_ancestor(6, sampled_node));
    REQUIRE(sampled_node != n_nodes - 1);
  }

  // We want to test that the samples are uniformly distributed among all
  // descendants of five. We therefore view `sample_descendant_or_nondescendant`
  // as a random variable that samples from the descendants of a node. Our
  // null-hypothesis is that X_v is uniformly distributed, i.e. \forall w \in
  // Desc(v): P(X_v = w) = (|Desc(v)|)^{-1} and we test this hypothesis with a
  // chi-squared-test.
  float t = 0;
  const float n_descendants = tree.get_n_descendants(5);
  for (std::pair<unsigned int, unsigned int> pair : sampled_nodes_for_five) {
    float numerator = (pair.second - n_iterations / n_descendants);
    numerator *= numerator;
    float denominator = n_iterations / n_descendants;
    t += numerator / denominator;
  }

  boost::math::chi_squared chi_squared_dist(5 - 1);
  REQUIRE(t < quantile(chi_squared_dist, 1 - alpha));
}

TEST_CASE("ChangeProposer::change_beta", "[ChangeProposer]") {
  auto proposer = init_proposer();

  for (uint32_t i = 0; i < n_iterations; i++) {
    float sampled_beta = proposer.change_beta(0.0);
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
