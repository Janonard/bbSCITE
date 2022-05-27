#pragma once
#include "ChangeProposer.hpp"
#include "StateScorer.hpp"

namespace ffSCITE {
template <uint64_t max_n_cells, uint64_t max_n_genes, typename RNG>
class MCMCKernel {
public:
  using ChangeProposerImpl = ChangeProposer<max_n_genes, RNG>;
  using StateScorerImpl = StateScorer<max_n_cells, max_n_genes>;
  using ChainStateImpl = ChainState<max_n_genes>;

  using MutationDataMatrix = typename StateScorerImpl::MutationDataMatrix;

  MCMCKernel(RNG rng, double prior_alpha, double prior_beta,
             double prior_beta_sd, double gamma, uint64_t n_cells,
             uint64_t n_genes, MutationDataMatrix data)
      : rng(rng), change_proposer(rng),
        state_scorer(prior_alpha, prior_beta, prior_beta_sd, n_cells, n_genes,
                     data),
        gamma(gamma) {}

  std::tuple<ChainStateImpl, double> execute_step(ChainStateImpl current_state,
                                                  double current_score) {
    double neighborhood_correction = 1.0;

    ChainStateImpl proposed_state = current_state;
    change_proposer.propose_change(proposed_state, neighborhood_correction);
    double proposed_score = state_scorer.logscore_state(proposed_state);

    double acceptance_probability =
        neighborhood_correction *
        std::exp((proposed_score - current_score) * gamma);
    bool accept_move = std::bernoulli_distribution(acceptance_probability)(rng);
    if (accept_move) {
      return {proposed_state, proposed_score};
    } else {
      return {current_state, current_score};
    }
  }

  RNG &get_rng() { return rng; }

  ChangeProposerImpl &get_change_proposer() { return change_proposer; }

  StateScorerImpl &get_state_scorer() { return state_scorer; }

private:
  RNG rng;
  ChangeProposerImpl change_proposer;
  StateScorerImpl state_scorer;

  double gamma;
};
} // namespace ffSCITE