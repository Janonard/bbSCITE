#include "ChangeProposer.hpp"
#include <catch2/catch_all.hpp>
#include <set>

using namespace ffSCITE;

TEST_CASE("ChangeProposer::sample_nodepair (correctness)", "[ChangeProposer]") {
  ChangeProposer<7, std::mt19937> proposer((std::mt19937()));

  for (uint64_t i = 0; i < 128; i++) {
    auto pair = proposer.sample_nodepair(7);
    REQUIRE(pair[0] < 7);
    REQUIRE(pair[1] < 7);
    REQUIRE(pair[0] != pair[1]);
  }
}

TEST_CASE("ChangeProposer::sample_descendant_or_nondescendant (correctness)",
          "[ChangeProposer]") {
  ChangeProposer<7, std::mt19937> proposer((std::mt19937()));

  /*
   * Original tree:
   *
   *   ┌-7-┐
   *  ┌5┐ ┌6
   * ┌2┐3 4
   * 0 1
   */
  ParentVector<7> pv(7);
  pv.from_pruefer_code({2, 2, 5, 5, 6, 7});
  AncestorMatrix<7> am(pv);

  std::set<unsigned int> descendants_of_five = {0, 1, 2, 3, 5};
  std::set<unsigned int> descendants_of_six = {4, 6};

  for (uint64_t i = 0; i < 128; i++) {
    auto sampled_node =
        proposer.sample_descendant_or_nondescendant(am, 5, true);
    REQUIRE(descendants_of_five.contains(sampled_node));

    sampled_node = proposer.sample_descendant_or_nondescendant(am, 5, false);
    REQUIRE(!descendants_of_five.contains(sampled_node));

    sampled_node = proposer.sample_descendant_or_nondescendant(am, 6, true);
    REQUIRE(descendants_of_six.contains(sampled_node));

    sampled_node = proposer.sample_descendant_or_nondescendant(am, 6, false);
    REQUIRE(!descendants_of_six.contains(sampled_node));
  }
}

TEST_CASE("ChangeProposer::change_beta (correctness)", "[ChangeProposer]") {
  ChangeProposer<7, std::mt19937> proposer((std::mt19937()));

  for (uint64_t i = 0; i < 128; i++) {
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

TEST_CASE("ChangeProposer::prune_and_reattach (correctness)",
          "[ChangeProposer]") {
  ChangeProposer<7, std::mt19937> proposer((std::mt19937()));

  /*
   * Original tree:
   *
   *   ┌-7-┐
   *  ┌5┐ ┌6
   * ┌2┐3 4
   * 0 1
   */
  ParentVector<7> pv(7);
  pv.from_pruefer_code({2, 2, 5, 5, 6, 7});
  AncestorMatrix<7> am(pv);

  for (uint64_t i = 0; i < 128; i++) {
    ParentVector<7> pv_copy(pv);
    unsigned int moved_node_i = proposer.prune_and_reattach(pv_copy, am);

    // Simple sanity checks for the move.
    REQUIRE(moved_node_i < 7);
    REQUIRE(pv_copy[moved_node_i] != moved_node_i);

    // Check that the node has not been attached to one of it's previous
    // descendants.
    REQUIRE(!am.is_ancestor(moved_node_i, pv_copy[moved_node_i]));

    // Check that nothing else was changed.
    for (uint64_t i = 0; i < 7; i++) {
      if (i != moved_node_i) {
        REQUIRE(pv_copy[i] == pv[i]);
      }
    }
  }
}

TEST_CASE("ChangeProposer::swap_subtrees (correctness)", "[ChangeProposer]") {
  ChangeProposer<7, std::mt19937> proposer((std::mt19937()));

  /*
   * Original tree:
   *
   *   ┌-7-┐
   *  ┌5┐ ┌6
   * ┌2┐3 4
   * 0 1
   */
  ParentVector<7> pv(7);
  pv.from_pruefer_code({2, 2, 5, 5, 6, 7});
  AncestorMatrix<7> am(pv);

  for (uint64_t i = 0; i < 128; i++) {
    ParentVector<7> pv_copy(pv);
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
      REQUIRE(neighborhood_correction == double(am.get_n_descendants(node_a_i)) /
          double(am.get_n_descendants(node_b_i)));
    } else {
      REQUIRE(pv_copy[node_a_i] == pv[node_b_i]);
      REQUIRE(pv_copy[node_b_i] == pv[node_a_i]);
      REQUIRE(neighborhood_correction == 1.0);
    }

    // Check that nothing else was changed.
    for (uint64_t i = 0; i < 7; i++) {
      if (i != node_a_i && i != node_b_i) {
        REQUIRE(pv_copy[i] == pv[i]);
      }
    }
  }
}