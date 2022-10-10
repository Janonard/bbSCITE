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
/**
 * @brief Controller class for the SCITE algorithm.
 *
 * @tparam max_n_cells The maximum number of cells processable by the design.
 * @tparam max_n_genes The maximum number of genes processable by the design.
 * @tparam pipeline_capacity The (assumed) capacity of the computation pipeline.
 * The feedback system assumes that this number is correct, but it has to be
 * manually verified using design space exploration.
 */
template <uint32_t max_n_cells, uint32_t max_n_genes,
          uint32_t pipeline_capacity>
class Application {
public:
  /**
   * @brief The mutation tree class used by the design.
   */
  using MutationTreeImpl = MutationTree<max_n_genes>;

  /**
   * @brief The ancestor matrix class used by the design.
   */
  using AncestorMatrix = typename MutationTreeImpl::AncestorMatrix;

  /**
   * @brief The type of the ancestor matrix rows.
   */
  using AncestryVector = typename MutationTreeImpl::AncestryVector;

  /**
   * @brief The tree scorer class used by the design on the FPGA.
   */
  using TreeScorerImpl = TreeScorer<max_n_cells, max_n_genes>;

  /**
   * @brief The tree scorer class used to compute the scores of the initial
   * states on the host.
   */
  using HostTreeScorerImpl = TreeScorer<max_n_cells, max_n_genes,
                                        cl::sycl::access::target::host_buffer>;

  /**
   * @brief Type of the mutation data matrix rows.
   */
  using MutationDataWord = typename TreeScorerImpl::MutationDataWord;

  /**
   * @brief Type of the locally stored mutation data matrix.
   */
  using MutationDataMatrix = typename TreeScorerImpl::MutationDataMatrix;

  /**
   * @brief Type of the matrix used to count error occurrences.
   */
  using OccurrenceMatrix = typename TreeScorerImpl::OccurrenceMatrix;

  /**
   * @brief The maximum number of tree nodes supported by the design.
   */
  static constexpr uint32_t max_n_nodes = MutationTreeImpl::max_n_nodes;

  /**
   * @brief Create a new application object.
   *
   * @param data_buffer A buffer with mutation data, used to judge the
   * likelihood of a mutation tree.
   * @param working_queue A SYCL queue, configured with the correct FPGA or FPGA
   * emulator device.
   * @param parameters The CLI parameters of the application.
   * @param n_cells The total number of cells in the input data.
   * @param n_genes The total number of genes in the input data.
   */
  Application(cl::sycl::buffer<MutationDataWord, 1> data_buffer,
              cl::sycl::queue working_queue, Parameters const &parameters,
              uint32_t n_cells, uint32_t n_genes)
      : raw_moves(cl::sycl::range<1>(1)),
        current_am_buffer(cl::sycl::range<1>(1)),
        current_beta_buffer(cl::sycl::range<1>(1)),
        current_score_buffer(cl::sycl::range<1>(1)),
        best_am_buffer(cl::sycl::range<1>(1)),
        best_beta_buffer(cl::sycl::range<1>(1)),
        best_score_buffer(cl::sycl::range<1>(1)), data_buffer(data_buffer),
        working_queue(working_queue), parameters(parameters), n_cells(n_cells),
        n_genes(n_genes) {
    using namespace cl::sycl;

    // Check that the required number of chains is correct.
    if (this->parameters.get_n_chains() % pipeline_capacity != 0) {
      uint32_t old_n_chains = this->parameters.get_n_chains();
      uint32_t new_n_chains =
          old_n_chains +
          (pipeline_capacity - (old_n_chains % pipeline_capacity));
      std::cerr << "Warning: Increasing the number of chains to "
                << new_n_chains << "." << std::endl;
      std::cerr << "This is the next multiple of the pipeline capacity and "
                   "doing so improves the performance."
                << std::endl;
      this->parameters.set_n_chains(new_n_chains);
    }

    /*
     * Generate the initial chain states.
     */
    current_am_buffer = range<1>(this->parameters.get_n_chains());
    current_beta_buffer = range<1>(this->parameters.get_n_chains());
    current_score_buffer = range<1>(this->parameters.get_n_chains());

    auto current_am_ac =
        current_am_buffer.template get_access<access::mode::discard_write>();
    auto current_beta_ac =
        current_beta_buffer.template get_access<access::mode::discard_write>();
    auto current_score_ac =
        current_score_buffer.template get_access<access::mode::discard_write>();
    auto data_ac = data_buffer.template get_access<access::mode::read>();
    MutationDataMatrix data;

    HostTreeScorerImpl host_scorer(
        this->parameters.get_alpha_mean(), this->parameters.get_beta_mean(),
        this->parameters.get_beta_sd(), n_cells, n_genes, data_ac, data);

    oneapi::dpl::minstd_rand rng;
    rng.seed(std::random_device()());

    for (uint32_t rep_i = 0; rep_i < this->parameters.get_n_chains(); rep_i++) {
      std::vector<uint32_t> pruefer_code =
          MutationTreeImpl::sample_random_pruefer_code(rng, n_genes);
      std::vector<uint32_t> parent_vector =
          MutationTreeImpl::pruefer_code_to_parent_vector(pruefer_code);
      current_am_ac[rep_i] =
          MutationTreeImpl::parent_vector_to_ancestor_matrix(parent_vector);

      current_beta_ac[rep_i] = this->parameters.get_beta_mean();

      MutationTreeImpl tree(current_am_ac[rep_i], n_genes,
                            current_beta_ac[rep_i]);
      current_score_ac[rep_i] = host_scorer.logscore_tree(tree);
    }

    /*
     * Generate the raw move samples.
     */
    raw_moves = cl::sycl::range<1>(this->parameters.get_n_chains() *
                                   this->parameters.get_chain_length());
    auto raw_moves_ac =
        raw_moves.template get_access<cl::sycl::access::mode::discard_write>();

    RawMoveDistribution distribution(n_genes + 1, this->parameters);
    for (uint32_t i = 0; i < raw_moves.get_range()[0]; i++) {
      raw_moves_ac[i] = distribution(rng);
    }
  }

  /**
   * @brief Execute the requested number of chains and chain steps on the
   * configured input, and store the most-likely solution.
   *
   * This method blocks on the FPGA execution and overrides the previous
   * best-known solution.
   *
   * @return float The makespan of the design, in seconds.
   */
  float run_simulation() {
    using namespace cl::sycl;

    std::vector<event> events;
    events.push_back(enqueue_io());
    events.push_back(enqueue_work_kernel());

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

  /**
   * @brief Get the ancestor matrix of the most-likely chain state.
   */
  AncestorMatrix get_best_am() {
    auto best_am_ac =
        best_am_buffer.template get_access<cl::sycl::access::mode::read>();
    return best_am_ac[0];
  }

  /**
   * @brief Get the beta value (probability of false negatives) of the
   * most-likely chain state.
   */
  float get_best_beta() {
    auto best_beta_ac =
        best_beta_buffer.template get_access<cl::sycl::access::mode::read>();
    return best_beta_ac[0];
  }

  /**
   * @brief Get the log-likelihood of the most-likely chain state.
   */
  float get_best_score() {
    auto best_score_ac =
        best_score_buffer.template get_access<cl::sycl::access::mode::read>();
    return best_score_ac[0];
  }

private:
  /**
   * @brief The scalar meta-information of a chain step.
   */
  struct ChainMeta {
    float beta;
    float score;
  };

  using InputMetaPipe = cl::sycl::pipe<class InputMetaPipeID, ChainMeta>;
  using InputTreePipe = cl::sycl::pipe<class InputTreePipeID, AncestryVector>;
  using OutputMetaPipe = cl::sycl::pipe<class OutputMetaPipeID, ChainMeta>;
  using OutputTreePipe = cl::sycl::pipe<class OutputTreePipeID, AncestryVector>;

  /**
   * @brief Enqueue the IO kernel which controls the introduction of initial
   * states and the pipeline feedback.
   *
   * @return cl::sycl::event The event object of the kernel invocation.
   */
  cl::sycl::event enqueue_io() {
    using namespace cl::sycl;

    assert(parameters.get_n_chains() % pipeline_capacity == 0);

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
          bool read_output = i >= pipeline_capacity;
          bool write_input = i < n_steps * n_chains;
          ac_int<1, false> input_source =
              (i % (n_steps * pipeline_capacity) < pipeline_capacity) ? 1 : 0;

          ChainMeta initial_meta;
          AncestorMatrix initial_am;
          if (input_source == 1) {
            initial_meta = ChainMeta{
                .beta = current_beta_ac[i_initial_state],
                .score = current_score_ac[i_initial_state],
            };
            initial_am = current_am_ac[i_initial_state];
            i_initial_state++;
          }

          ChainMeta output_meta;
          AncestorMatrix output_am;
          if (read_output) {
            output_meta = OutputMetaPipe::read();
          }
          for (uint32_t word_i = 0; word_i < max_n_nodes; word_i++) {
            if (read_output) {
              output_am[word_i] = OutputTreePipe::read();
            }
          }

          if (write_input) {
            ChainMeta input_meta =
                (input_source == 1) ? initial_meta : output_meta;
            InputMetaPipe::write(input_meta);
          }
          for (uint32_t word_i = 0; word_i < max_n_nodes; word_i++) {
            if (write_input) {
              AncestryVector input_vector;
              if (input_source == 1) {
                input_vector = initial_am[word_i];
              } else {
                input_vector = output_am[word_i];
              }
              InputTreePipe::write(input_vector);
            }
          }
        }
      });
    });
  }

  /**
   * @brief Enqueue the work kernel, which loads and realizes the move
   * parameters, computes the resulting state of a move, computes its likelihood
   * and decides whether it is the new current state of the chain.
   *
   * @return cl::sycl::event The event object of the kernel invocation.
   */
  cl::sycl::event enqueue_work_kernel() {
    using namespace cl::sycl;

    return working_queue.submit([&](handler &cgh) {
      auto data_ac = data_buffer.template get_access<access::mode::read>(cgh);
      auto raw_moves_ac =
          raw_moves.template get_access<access::mode::read>(cgh);

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

      cgh.single_task<class WorkKernel>([=]() {
        MutationDataMatrix data;
        TreeScorerImpl tree_scorer(alpha_mean, beta_mean, beta_sd, n_cells,
                                   n_genes, data_ac, data);

        AncestorMatrix best_am;
        float best_beta = 1.0;
        float best_score = -std::numeric_limits<float>::infinity();

        for (uint32_t i = 0; i < n_chains * n_steps; i++) {
          ChainMeta tree_meta = InputMetaPipe::read();

          AncestorMatrix current_am;
          for (uint32_t word_i = 0; word_i < max_n_nodes; word_i++) {
            current_am[word_i] = InputTreePipe::read();
          }

          float current_beta = tree_meta.beta;
          float current_score = tree_meta.score;
          MutationTreeImpl current_tree(current_am, n_genes, current_beta);

          RawMoveSample raw_move = raw_moves_ac[i];
          typename MutationTreeImpl::ModificationParameters mod_params =
              current_tree.realize_raw_move_sample(raw_move);

          uint32_t v = mod_params.v;
          uint32_t w = mod_params.w;

          AncestorMatrix proposed_am;
          MutationTreeImpl proposed_tree(proposed_am, current_tree, mod_params);
          float proposed_beta = proposed_tree.get_beta();
          float proposed_score = tree_scorer.logscore_tree(proposed_tree);

          float neighborhood_correction;
          if (raw_move.move_type == MoveType::SwapSubtrees &&
              current_tree.is_ancestor(w, v)) {
            neighborhood_correction = float(current_tree.get_n_descendants(v)) /
                                      float(current_tree.get_n_descendants(w));
          } else {
            neighborhood_correction = 1.0;
          }

          float acceptance_probability =
              neighborhood_correction *
              std::exp((proposed_score - current_score) * gamma);
          bool accept_move = acceptance_probability > raw_move.acceptance_level;

          ChainMeta output_meta{
              .beta = accept_move ? proposed_beta : current_beta,
              .score = accept_move ? proposed_score : current_score,
          };

          OutputMetaPipe::write(output_meta);

          for (uint32_t word_i = 0; word_i < max_n_nodes; word_i++) {
            AncestryVector output_vector;
            if (accept_move) {
              output_vector = proposed_am[word_i];
            } else {
              output_vector = current_am[word_i];
            }
            OutputTreePipe::write(output_vector);
          }

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

  cl::sycl::buffer<RawMoveSample, 1> raw_moves;

  cl::sycl::buffer<AncestorMatrix, 1> current_am_buffer;
  cl::sycl::buffer<float, 1> current_beta_buffer;
  cl::sycl::buffer<float, 1> current_score_buffer;

  cl::sycl::buffer<AncestorMatrix, 1> best_am_buffer;
  cl::sycl::buffer<float, 1> best_beta_buffer;
  cl::sycl::buffer<float, 1> best_score_buffer;

  cl::sycl::buffer<MutationDataWord, 1> data_buffer;
  cl::sycl::queue working_queue;
  Parameters parameters;
  uint32_t n_cells, n_genes;
};
} // namespace ffSCITE