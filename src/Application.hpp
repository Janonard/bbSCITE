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
#include "ChangeProposer.hpp"
#include "Parameters.hpp"
#include "TreeScorer.hpp"
#include <CL/sycl.hpp>

namespace ffSCITE {
template <typename RNG, typename NumberPipe, typename FeedbackPipe>
class PipeRNG {
public:
  using result_type = typename RNG::result_type;

  static result_type min() { return RNG::min(); }

  static result_type max() { return RNG::max(); }

  result_type operator()() { return NumberPipe::read(); }

  void close() {
    // signal the kernel to halt.
    FeedbackPipe::write(true);
    // Read a number to unlock the kernel.
    NumberPipe::read();
  }
};

template <uint32_t max_n_cells, uint32_t max_n_genes> class Application {
public:
  using RNG = oneapi::dpl::minstd_rand;
  using NumberPipe =
      cl::sycl::pipe<class NumberPipeID, typename RNG::result_type>;
  using FeedbackPipe = cl::sycl::pipe<class FeedbackPipeID, bool>;
  using PipeRNGImpl = PipeRNG<RNG, NumberPipe, FeedbackPipe>;

  using MutationTreeImpl = MutationTree<max_n_genes>;
  using AncestorMatrix = typename MutationTreeImpl::AncestorMatrix;
  using ChangeProposerImpl = ChangeProposer<max_n_genes, PipeRNGImpl>;
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
    rng.seed(parameters.get_seed());

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

  struct InputState {
    AncestorMatrix current_am;
    float current_beta;
    float current_score;
  };

  using InputPipe = cl::sycl::pipe<class InputPipeID, InputState>;

  struct ProposedChangeState {
    AncestorMatrix current_am;
    float current_beta;
    float current_score;

    typename ChangeProposerImpl::ChainStepSample change_sample;
  };

  using ProposedChangePipe =
      cl::sycl::pipe<class ChangedStatePipeID, ProposedChangeState>;

  struct OutputState {
    AncestorMatrix new_am;
    float new_beta;
    float new_score;
  };

  using OutputPipe = cl::sycl::pipe<class OutputPipeID, OutputState>;

  float run_simulation() {
    using namespace cl::sycl;

    std::vector<event> events;
    events.push_back(enqueue_io());
    events.push_back(enqueue_rng_kernel());
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
          current_am_buffer.template get_access<access::mode::read_write>(cgh);
      auto current_beta_ac =
          current_beta_buffer.template get_access<access::mode::read_write>(
              cgh);
      auto current_score_ac =
          current_score_buffer.template get_access<access::mode::read_write>(
              cgh);

      uint32_t n_steps = parameters.get_chain_length();
      uint32_t n_chains = parameters.get_n_chains();

      cgh.single_task<class IOKernel>([=]() {
        [[intel::loop_coalesce(2)]] for (uint32_t step_i = 0; step_i < n_steps;
                                         step_i++) {
          for (uint32_t chain_i = 0; chain_i < n_chains; chain_i++) {
            InputState input_state{
                .current_am = current_am_ac[chain_i],
                .current_beta = current_beta_ac[chain_i],
                .current_score = current_score_ac[chain_i],
            };
            InputPipe::write(input_state);

            OutputState output_state = OutputPipe::read();
            current_am_ac[chain_i] = output_state.new_am;
            current_beta_ac[chain_i] = output_state.new_beta;
            current_score_ac[chain_i] = output_state.new_score;
          }
        }
      });
    });
  }

  cl::sycl::event enqueue_rng_kernel() {
    using namespace cl::sycl;

    return working_queue.submit([&](handler &cgh) {
      uint64_t seed = parameters.get_seed();

      cgh.single_task<class RNGKernel>([=]() {
        RNG rng;
        rng.seed(seed);

        bool halt = false;
        while (!halt) {
          NumberPipe::write(rng());
          bool read_successful;
          bool halt_execution = FeedbackPipe::read(read_successful);
          halt = read_successful && halt_execution;
        }
      });
    });
  }

  cl::sycl::event enqueue_change_proposer() {
    using namespace cl::sycl;

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
        PipeRNGImpl rng;
        [[intel::fpga_register]] ChangeProposerImpl change_proposer(
            rng, prob_beta_change, prob_prune_n_reattach, prob_swap_nodes,
            beta_jump_sd);

        for (uint32_t i = 0; i < n_chains * n_steps; i++) {
          InputState input_state = InputPipe::read();

          AncestorMatrix current_am = input_state.current_am;
          float current_beta = input_state.current_beta;
          float current_score = input_state.current_score;

          MutationTreeImpl current_tree(current_am, n_genes, current_beta);

          auto change_sample =
              change_proposer.sample_step_parameters(current_tree);

          ProposedChangeState proposed_change_state{
              .current_am = current_am,
              .current_beta = current_beta,
              .current_score = current_score,
              .change_sample = change_sample};

          ProposedChangePipe::write(proposed_change_state);
        }

        rng.close();
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

          AncestorMatrix current_am = proposed_change_state.current_am;
          float current_beta = proposed_change_state.current_beta;
          float current_score = proposed_change_state.current_score;
          MutationTreeImpl current_tree(current_am, n_genes, current_beta);

          auto change_sample = proposed_change_state.change_sample;

          uint32_t v = change_sample.v;
          uint32_t w = change_sample.w;

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
              .move_type = change_sample.move_type,
              .v = change_sample.v,
              .w = change_sample.w,
              .parent_of_v = parent_of_v,
              .parent_of_w = parent_of_w,
              .descendant_of_v = change_sample.descendant_of_v,
              .nondescendant_of_v = change_sample.nondescendant_of_v,
              .new_beta = change_sample.new_beta,
          };

          AncestorMatrix proposed_am;
          MutationTreeImpl proposed_tree(proposed_am, current_tree, mod_params);
          float proposed_beta = proposed_tree.get_beta();
          float proposed_score = tree_scorer.logscore_tree(proposed_tree);

          float neighborhood_correction;
          if (change_sample.move_type == MoveType::SwapSubtrees &&
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
              acceptance_probability > change_sample.acceptance_level;

          OutputState output_state{
              .new_am = accept_move ? proposed_am : current_am,
              .new_beta = accept_move ? proposed_beta : current_beta,
              .new_score = accept_move ? proposed_score : current_score,
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