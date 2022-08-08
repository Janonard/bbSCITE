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
#pragma once
#include "Parameters.hpp"
#include "TreeScorer.hpp"
#include <CL/sycl.hpp>

namespace ffSCITE {
template <uint32_t max_n_cells, uint32_t max_n_genes,
          uint32_t pipeline_capacity>
class Application {
public:
  using MutationTreeImpl = MutationTree<max_n_genes>;
  using AncestorMatrix = typename MutationTreeImpl::AncestorMatrix;
  using TreeScorerImpl = TreeScorer<max_n_cells, max_n_genes>;
  using HostTreeScorerImpl = TreeScorer<max_n_cells, max_n_genes,
                                        cl::sycl::access::target::host_buffer>;
  using MutationDataWord = typename TreeScorerImpl::MutationDataWord;
  using MutationDataMatrix = typename TreeScorerImpl::MutationDataMatrix;
  using OccurrenceMatrix = typename TreeScorerImpl::OccurrenceMatrix;

  Application(cl::sycl::buffer<MutationDataWord, 1> data_buffer,
              cl::sycl::queue working_queue, Parameters const &parameters,
              uint32_t n_cells, uint32_t n_genes)
      : current_am_buffer(cl::sycl::range<1>(1)),
        current_beta_buffer(cl::sycl::range<1>(1)),
        current_score_buffer(cl::sycl::range<1>(1)),
        best_am_buffer(cl::sycl::range<1>(1)),
        best_beta_buffer(cl::sycl::range<1>(1)),
        best_score_buffer(cl::sycl::range<1>(1)), data_buffer(data_buffer),
        working_queue(working_queue), parameters(parameters), n_cells(n_cells),
        n_genes(n_genes) {
    initialize_state();
  }

  void initialize_state() {
    using namespace cl::sycl;

    oneapi::dpl::minstd_rand rng;
    rng.seed(std::random_device()());

    current_am_buffer = range<1>(parameters.get_n_chains());
    current_beta_buffer = range<1>(parameters.get_n_chains());
    current_score_buffer = range<1>(parameters.get_n_chains());

    auto current_am_ac =
        current_am_buffer.template get_access<access::mode::discard_write>();
    auto current_beta_ac =
        current_beta_buffer.template get_access<access::mode::discard_write>();
    auto current_score_ac =
        current_score_buffer.template get_access<access::mode::discard_write>();
    auto data_ac = data_buffer.template get_access<access::mode::read>();
    MutationDataMatrix data;

    HostTreeScorerImpl host_scorer(
        parameters.get_alpha_mean(), parameters.get_beta_mean(),
        parameters.get_beta_sd(), n_cells, n_genes, data_ac, data);

    for (uint32_t rep_i = 0; rep_i < parameters.get_n_chains(); rep_i++) {
      std::vector<uint32_t> pruefer_code =
          MutationTreeImpl::sample_random_pruefer_code(rng, n_genes);
      std::vector<uint32_t> parent_vector =
          MutationTreeImpl::pruefer_code_to_parent_vector(pruefer_code);
      current_am_ac[rep_i] =
          MutationTreeImpl::parent_vector_to_ancestor_matrix(parent_vector);

      current_beta_ac[rep_i] = parameters.get_beta_mean();

      MutationTreeImpl tree(current_am_ac[rep_i], n_genes,
                            current_beta_ac[rep_i]);
      current_score_ac[rep_i] = host_scorer.logscore_tree(tree);
    }
  }

  struct ChainState {
    AncestorMatrix ancestor_matrix;
    float beta;
    float score;
  };

  using InputPipe = cl::sycl::pipe<class InputPipeID, ChainState>;
  using OutputPipe = cl::sycl::pipe<class OutputPipeID, ChainState>;

  struct ProposedChangeState {
    ChainState current_state;

    MoveType move_type;
    uint32_t v, w, descendant_of_v, nondescendant_of_v;
    float new_beta;
    float acceptance_level;
  };

  using ProposedChangePipe =
      cl::sycl::pipe<class ChangedStatePipeID, ProposedChangeState>;

  float run_simulation() {
    using namespace cl::sycl;

    std::vector<event> events;
    events.push_back(enqueue_io());
    events.push_back(enqueue_change_proposer());
    events.push_back(enqueue_tree_scorer());

    std::vector<uint64_t> starts, ends;

    for (event kernel_event : events) {
      uint64_t start = kernel_event.template get_profiling_info<
          cl::sycl::info::event_profiling::command_start>();
      uint64_t end = kernel_event.template get_profiling_info<
          cl::sycl::info::event_profiling::command_end>();
      starts.push_back(start);
      ends.push_back(end);
    }

    uint64_t execution_start = *std::min_element(starts.begin(), starts.end());
    uint64_t execution_end = *std::max_element(ends.begin(), ends.end());

    return (execution_end - execution_start) / 1000000.0;
  }

  AncestorMatrix get_best_am() {
    auto best_am_ac =
        best_am_buffer.template get_access<cl::sycl::access::mode::read>();
    return best_am_ac[0];
  }

  float get_best_beta() {
    auto best_beta_ac =
        best_beta_buffer.template get_access<cl::sycl::access::mode::read>();
    return best_beta_ac[0];
  }

  float get_best_score() {
    auto best_score_ac =
        best_score_buffer.template get_access<cl::sycl::access::mode::read>();
    return best_score_ac[0];
  }

private:
  cl::sycl::event enqueue_io() {
    using namespace cl::sycl;

    return working_queue.submit([&](handler &cgh) {
      auto current_am_ac =
          current_am_buffer.template get_access<access::mode::read>(cgh);
      auto current_beta_ac =
          current_beta_buffer.template get_access<access::mode::read>(cgh);
      auto current_score_ac =
          current_score_buffer.template get_access<access::mode::read>(cgh);

      uint32_t n_steps = parameters.get_chain_length();
      uint32_t n_chains = parameters.get_n_chains();

      cgh.single_task<class IOKernel>([=]() {
        uint32_t i_initial_state = 0;

        for (uint32_t i = 0; i < n_steps * n_chains + pipeline_capacity; i++) {
          // Always read the result from the output pipe if we can expect an
          // output, even if the result is not written back.
          ChainState feedback_state;
          if (i > pipeline_capacity) {
            feedback_state = OutputPipe::read();
          }

          // We only provide inputs for the first `n_steps * n_chains`
          // iterations. The rest is just the tail to empty the pipes.
          if (i < n_steps * n_chains) {

            ChainState input_state;
            if (i % n_steps < pipeline_capacity) {
              // For the first couple of iterations, we read the inputs from our
              // initial state buffer.
              input_state = ChainState{
                  .ancestor_matrix = current_am_ac[i_initial_state],
                  .beta = current_beta_ac[i_initial_state],
                  .score = current_score_ac[i_initial_state],
              };
              i_initial_state++;
            } else {
              // After that, we simply use the previous output.
              input_state = feedback_state;
            }

            InputPipe::write(input_state);
          }
        }
      });
    });
  }

  cl::sycl::event enqueue_change_proposer() {
    using namespace cl::sycl;

    std::random_device seeder;
    std::array<uint64_t, 6> seed;
    seed[0] = seeder();
    seed[1] = seeder();
    seed[2] = seeder();
    seed[3] = seeder();
    seed[4] = seeder();
    seed[5] = seeder();

    return working_queue.submit([&](handler &cgh) {
      float prob_beta_change = parameters.get_prob_beta_change();
      float prob_prune_n_reattach = parameters.get_prob_prune_n_reattach();
      float prob_swap_nodes = parameters.get_prob_swap_nodes();
      float beta_jump_sd =
          parameters.get_beta_sd() / parameters.get_beta_jump_scaling_chi();
      uint32_t n_steps = parameters.get_chain_length();
      uint32_t n_chains = parameters.get_n_chains();
      uint32_t n_genes = this->n_genes;

      cgh.single_task<class ChangeProposerKernel>([=]() {
        oneapi::dpl::minstd_rand nodepair_rng, descendant_rng,
            nondescendant_rng, move_rng, beta_rng, acceptance_rng;
        nodepair_rng.seed(seed[0]);
        descendant_rng.seed(seed[1]);
        nondescendant_rng.seed(seed[2]);
        move_rng.seed(seed[3]);
        beta_rng.seed(seed[4]);
        acceptance_rng.seed(seed[5]);

        for (uint32_t i = 0; i < n_chains * n_steps; i++) {
          ChainState input_state = InputPipe::read();

          AncestorMatrix current_am = input_state.ancestor_matrix;
          float current_beta = input_state.beta;

          MutationTreeImpl current_tree(current_am, n_genes, current_beta);

          std::array<uint32_t, 2> v_and_w =
              current_tree.sample_nonroot_nodepair(nodepair_rng);
          uint32_t v = v_and_w[0];
          uint32_t w = v_and_w[1];

          uint32_t descendant_of_v =
              current_tree.sample_descendant(descendant_rng, v);
          uint32_t nondescendant_of_v =
              current_tree.sample_nondescendant(nondescendant_rng, v);

          MoveType move_type =
              current_tree.sample_move(move_rng, prob_beta_change,
                                       prob_prune_n_reattach, prob_swap_nodes);

          float new_beta = current_tree.sample_new_beta(beta_rng, beta_jump_sd);

          float acceptance_level =
              oneapi::dpl::uniform_real_distribution(0.0, 1.0)(acceptance_rng);

          ProposedChangeState proposed_change_state{
              .current_state = input_state,
              .move_type = move_type,
              .v = v,
              .w = w,
              .descendant_of_v = descendant_of_v,
              .nondescendant_of_v = nondescendant_of_v,
              .new_beta = new_beta,
              .acceptance_level = acceptance_level};

          ProposedChangePipe::write(proposed_change_state);
        }
      });
    });
  }

  cl::sycl::event enqueue_tree_scorer() {
    using namespace cl::sycl;

    return working_queue.submit([&](handler &cgh) {
      auto data_ac = data_buffer.template get_access<access::mode::read>(cgh);

      auto best_am_ac =
          best_am_buffer.template get_access<access::mode::discard_write>(cgh);
      auto best_beta_ac =
          best_beta_buffer.template get_access<access::mode::discard_write>(
              cgh);
      auto best_score_ac =
          best_score_buffer.template get_access<access::mode::discard_write>(
              cgh);

      uint32_t n_cells = this->n_cells;
      uint32_t n_genes = this->n_genes;
      float alpha_mean = parameters.get_alpha_mean();
      float beta_mean = parameters.get_beta_mean();
      float beta_sd = parameters.get_beta_sd();
      float gamma = parameters.get_gamma();
      uint32_t n_steps = parameters.get_chain_length();
      uint32_t n_chains = parameters.get_n_chains();

      cgh.single_task<class TreeScorerKernel>([=]() {
        MutationDataMatrix data;
        TreeScorerImpl tree_scorer(alpha_mean, beta_mean, beta_sd, n_cells,
                                   n_genes, data_ac, data);

        AncestorMatrix best_am;
        float best_beta = 1.0;
        float best_score = -std::numeric_limits<float>::infinity();

        for (uint32_t i = 0; i < n_chains * n_steps; i++) {
          ProposedChangeState proposed_change_state =
              ProposedChangePipe::read();

          AncestorMatrix current_am =
              proposed_change_state.current_state.ancestor_matrix;
          float current_beta = proposed_change_state.current_state.beta;
          float current_score = proposed_change_state.current_state.score;
          MutationTreeImpl current_tree(current_am, n_genes, current_beta);

          uint32_t v = proposed_change_state.v;
          uint32_t w = proposed_change_state.w;

          uint32_t parent_of_v, parent_of_w;
          for (uint32_t node_i = 0; node_i < max_n_genes + 1; node_i++) {
            if (node_i >= current_tree.get_n_nodes()) {
              continue;
            }
            if (current_tree.is_parent(node_i, v)) {
              parent_of_v = node_i;
            }
            if (current_tree.is_parent(node_i, w)) {
              parent_of_w = node_i;
            }
          }

          typename MutationTreeImpl::ModificationParameters mod_params{
              .move_type = proposed_change_state.move_type,
              .v = proposed_change_state.v,
              .w = proposed_change_state.w,
              .parent_of_v = parent_of_v,
              .parent_of_w = parent_of_w,
              .descendant_of_v = proposed_change_state.descendant_of_v,
              .nondescendant_of_v = proposed_change_state.nondescendant_of_v,
              .new_beta = proposed_change_state.new_beta,
          };

          AncestorMatrix proposed_am;
          MutationTreeImpl proposed_tree(proposed_am, current_tree, mod_params);
          float proposed_beta = proposed_tree.get_beta();
          float proposed_score = tree_scorer.logscore_tree(proposed_tree);

          float neighborhood_correction;
          if (proposed_change_state.move_type == MoveType::SwapSubtrees &&
              current_tree.is_ancestor(w, v)) {
            neighborhood_correction = float(current_tree.get_n_descendants(v)) /
                                      float(current_tree.get_n_descendants(w));
          } else {
            neighborhood_correction = 1.0;
          }

          float acceptance_probability =
              neighborhood_correction *
              std::exp((proposed_score - current_score) * gamma);
          bool accept_move =
              acceptance_probability > proposed_change_state.acceptance_level;

          ChainState output_state{
              .ancestor_matrix = accept_move ? proposed_am : current_am,
              .beta = accept_move ? proposed_beta : current_beta,
              .score = accept_move ? proposed_score : current_score,
          };

          OutputPipe::write(output_state);

          if (i == 0 || proposed_score > best_score) {
            best_am = proposed_am;
            best_beta = proposed_beta;
            best_score = proposed_score;
          }
        }

        best_am_ac[0] = best_am;
        best_beta_ac[0] = best_beta;
        best_score_ac[0] = best_score;
      });
    });
  }

  cl::sycl::buffer<AncestorMatrix, 1> current_am_buffer;
  cl::sycl::buffer<float, 1> current_beta_buffer;
  cl::sycl::buffer<float, 1> current_score_buffer;

  cl::sycl::buffer<AncestorMatrix, 1> best_am_buffer;
  cl::sycl::buffer<float, 1> best_beta_buffer;
  cl::sycl::buffer<float, 1> best_score_buffer;

  cl::sycl::buffer<MutationDataWord, 1> data_buffer;
  cl::sycl::queue working_queue;
  Parameters const &parameters;
  uint32_t n_cells, n_genes;
};
} // namespace ffSCITE