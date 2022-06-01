/*
 * Copyright © 2022 Jan-Oliver Opdenhövel
 * Copyright © 2015 Katharina Jahn
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
#include <cstring>
#include <getopt.h>
#include <iostream>
#include <optional>
#include <random>
#include <vector>

namespace ffSCITE {
/**
 * @brief Container with all experiment parameters and a method to load them
 * from CLI arguments.
 *
 */
class Parameters {
public:
  /**
   * @brief Initialize the parameters class with default values.
   *
   * Not that for some fields like @ref get_input_path, the internal default
   * value is std::nullopt. Therefore, calling the respective getter without
   * setting it before will through an exception.
   */
  Parameters()
      : input_path(std::nullopt), output_path_base(std::nullopt),
        n_chains(std::nullopt), chain_length(std::nullopt),
        max_n_best_states(128), alpha_mean(6.04e-5), beta_mean(0.4309),
        beta_sd(0.1), prob_beta_change(0.0), prob_prune_n_reattach(0.55),
        prob_swap_nodes(0.4), prob_swap_subtrees(0.05),
        beta_jump_scaling_chi(10.0), gamma(1.0), seed(std::nullopt) {}

  /**
   * @brief Verify the CLI arguments and store them in the object's fields.
   *
   * The way this is implemented is WET and errorprone, but necessary to
   * stay CLI-compatible with the original SCITE application. I have copied it
   * from their implementation, and added some errors and warnings.
   *
   * @param argc The number of arguments passed to the main function.
   * @param argv The list of arguments passed to the main function.
   * @return true Arguments were free of errors and parameters were successfully
   * loaded.
   * @return false There are errors in the CLI arguments. Some parameters may
   * have been loaded successfully.
   */
  bool load_and_verify_args(int argc, char **argv) {
    bool error = false;
    for (int i = 1; i < argc; ++i) {
      if (argv[i][0] != '-') {
        continue;
      }

      if (strcmp(argv[i], "-i") == 0) {
        if (i + 1 < argc) {
          input_path = argv[++i];
        } else {
          std::cerr << "Error: Missing argument to parameter -i." << std::endl;
          error = true;
        }
      } else if (strcmp(argv[i], "-t") == 0) {
        std::cerr << "Warning: The -t parameter is not supported by ffSCITE "
                     "(yet). Ignoring the -t parameter."
                  << std::endl;
      } else if (strcmp(argv[i], "-o") == 0) {
        if (i + 1 < argc) {
          output_path_base = argv[++i];
        } else {
          std::cerr << "Error: Missing argument to parameter -o." << std::endl;
          error = true;
        }
      } else if (strcmp(argv[i], "-n") == 0) {
        std::cerr << "Warning: The number of genes is inferred from the input "
                     "file. Ignoring parameter -n."
                  << std::endl;
      } else if (strcmp(argv[i], "-m") == 0) {
        std::cerr << "Warning: The number of cells is inferred from the input "
                     "file. Ignoring parameter -n."
                  << std::endl;
      } else if (strcmp(argv[i], "-r") == 0) {
        if (i + 1 < argc) {
          n_chains = atoi(argv[++i]);
        } else {
          std::cerr << "Error: Missing argument to parameter -r." << std::endl;
          error = true;
        }
      } else if (strcmp(argv[i], "-l") == 0) {
        if (i + 1 < argc) {
          chain_length = atoi(argv[++i]);
        } else {
          std::cerr << "Error: Missing argument to parameter -l." << std::endl;
          error = true;
        }
      } else if (strcmp(argv[i], "-g") == 0) {
        if (i + 1 < argc) {
          gamma = atof(argv[++i]);
        } else {
          std::cerr << "Error: Missing argument to parameter -g." << std::endl;
          error = true;
        }
      } else if (strcmp(argv[i], "-fd") == 0) {
        if (i + 1 < argc) {
          alpha_mean = atof(argv[++i]);
        } else {
          std::cerr << "Error: Missing argument to parameter -fd." << std::endl;
          error = true;
        }
      } else if (strcmp(argv[i], "-ad") == 0) {
        if (i + 1 < argc) {
          beta_mean = atof(argv[++i]);
        } else {
          std::cerr << "Error: Missing argument to parameter -ad." << std::endl;
          error = true;
        }
        if (i + 1 < argc) {
          std::string next = argv[i + 1];
          if (next.compare(0, 1, "-") != 0) {
            beta_mean += atof(argv[++i]);
            std::cerr << "Warning: homo- and heterocygous mutations are not "
                         "supported by "
                         "ffSCITE (yet). Summing false negative probabilites."
                      << std::endl;
          }
        }
      } else if (strcmp(argv[i], "-cc") == 0) {
        std::cerr
            << "Warning: homo- and heterocygous mutations are not supported by "
               "ffSCITE (yet). -cc argument ignored."
            << std::endl;
      } else if (strcmp(argv[i], "-e") == 0) {
        if (i + 1 < argc) {
          prob_beta_change = atof(argv[++i]);
          if (prob_beta_change < 0.0) {
            std::cerr << "Error: Move probabilities may not be negative."
                      << std::endl;
            error = true;
          }
        } else {
          std::cerr << "Error: Missing argument to parameter -e." << std::endl;
          error = true;
        }
      } else if (strcmp(argv[i], "-x") == 0) {
        if (i + 1 < argc) {
          beta_jump_scaling_chi = atof(argv[++i]);
        } else {
          std::cerr << "Error: Missing argument to parameter -x." << std::endl;
          error = true;
        }
      } else if (strcmp(argv[i], "-sd") == 0) {
        if (i + 1 < argc) {
          beta_sd = atof(argv[++i]);
        } else {
          std::cerr << "Error: Missing argument to parameter -sd." << std::endl;
          error = true;
        }
      } else if (strcmp(argv[i], "-a") == 0) {
        std::cerr << "Warning: The -a parameter is not supported by ffSCITE "
                     "(yet). Ignoring the -a parameter."
                  << std::endl;
      } else if (strcmp(argv[i], "-p") == 0) {
        std::cerr << "Warning: The -p parameter is not supported by ffSCITE "
                     "(yet). Ignoring the -p parameter."
                  << std::endl;
      } else if (strcmp(argv[i], "-names") == 0) {
        std::cerr << "The -names parameter is not supported by ffSCITE (yet). "
                     "Ignoring the -names parameter."
                  << std::endl;
      } else if (strcmp(argv[i], "-move_probs") == 0) {
        if (i + 1 < argc) {
          prob_prune_n_reattach = atof(argv[++i]);
          if (prob_prune_n_reattach < 0.0) {
            std::cerr << "Error: Move probabilities may not be negative."
                      << std::endl;
            error = true;
          }
        } else {
          std::cerr << "Error: Missing argument to parameter -move_probs."
                    << std::endl;
          error = true;
        }
        if (i + 1 < argc) {
          prob_swap_nodes = atof(argv[++i]);
          if (prob_swap_nodes < 0.0) {
            std::cerr << "Error: Move probabilities may not be negative."
                      << std::endl;
            error = true;
          }
        } else {
          std::cerr << "Error: Missing argument to parameter -move_probs."
                    << std::endl;
          error = true;
        }
        if (i + 1 < argc) {
          prob_swap_subtrees = atof(argv[++i]);
          if (prob_swap_subtrees < 0.0) {
            std::cerr << "Error: Move probabilities may not be negative."
                      << std::endl;
            error = true;
          }
        } else {
          std::cerr << "Error: Missing argument to parameter -move_probs."
                    << std::endl;
          error = true;
        }
      } else if (strcmp(argv[i], "-seed") == 0) {
        if (i + 1 < argc) {
          seed = atoi(argv[++i]);
        } else {
          std::cerr << "Error: Missing argument to parameter -seed."
                    << std::endl;
          error = true;
        }
      } else if (strcmp(argv[i], "-max_treelist_size") == 0) {
        if (i + 1 < argc) {
          max_n_best_states = atoi(argv[++i]);
        } else {
          std::cerr
              << "Error: Missing argument to parameter -max_treelist_size."
              << std::endl;
          error = true;
        }
      } else if (strcmp(argv[i], "-no_tree_list") == 0) {
        std::cerr << "Warning: The -no_tree_list parameter is not supported by "
                     "ffSCITE (yet). Ignoring the -no_tree_list parameter."
                  << std::endl;
      } else if (strcmp(argv[i], "-s") == 0) {
        std::cerr << "Warning: The -s parameter is not supported by ffSCITE "
                     "(yet). Ignoring the -s parameter."
                  << std::endl;
      } else if (strcmp(argv[i], "-transpose") == 0) {
        std::cerr << "Warning: The -transpose parameter is not supported by "
                     "ffSCITE (yet). Ignoring the -transpose parameter."
                  << std::endl;
      } else {
        std::cerr << "Error: Unknown parameter " << argv[i] << "." << std::endl;
        error = true;
      }
    }

    // Checking that the required parameters are set.
    if (!input_path.has_value()) {
      std::cerr << "Error: Missing input file path. Did you forget to set the "
                   "-i paramater?"
                << std::endl;
      error = true;
    }

    if (!n_chains.has_value()) {
      std::cerr << "Error: Missing number of markov chains to simulate. Did "
                   "you forget to set the -r parameter?"
                << std::endl;
      error = true;
    }

    if (!chain_length.has_value()) {
      std::cerr << "Error: Missing markov chain length. Did you forget to set "
                   "the -l parameter?"
                << std::endl;
      error = true;
    }

    if (prob_beta_change + prob_prune_n_reattach + prob_swap_nodes +
            prob_swap_subtrees ==
        0) {
      std::cerr << "Error: The sum of the move type probabilities is zero."
                << std::endl;
      error = true;
    }

    return !error;
  }

  /**
   * @brief Get the path to the input file.
   *
   * @return std::string The path to the input file.
   */
  std::string get_input_path() const { return input_path.value(); }

  /**
   * @brief Set the path to the input file.
   *
   * @param input_path The new path to the input file.
   */
  void set_input_path(std::string input_path) { this->input_path = input_path; }

  /**
   * @brief Get the path base for output files.
   *
   * If the path base has not been explicitly set, it will be the input path
   * without the last suffix. Users should append additional identifiers and
   * file name suffixes to the path base for their output files.
   *
   * @return std::string The path base for output files.
   */
  std::string get_output_path_base() const {
    if (output_path_base.has_value()) {
      return output_path_base.value();
    } else {
      int lastIndex = input_path->find_last_of(".");
      return input_path->substr(0, lastIndex);
    }
  }

  /**
   * @brief Set the path base for output files.
   *
   * @param output_path_base The new path base for output files.
   */
  void set_output_path_base(std::string output_path_base) {
    this->output_path_base = output_path_base;
  }

  /**
   * @brief Get the number of chains to execute.
   *
   * For every chain, a new random starting state is initialized and @ref
   * get_chain_length steps are executed from this state.
   *
   * @return uint64_t The number of chains to execute.
   */
  uint64_t get_n_chains() const { return n_chains.value(); }

  /**
   * @brief Set the number of chains to execute.
   *
   * For every chain, a new random starting state is initialized and @ref
   * get_chain_length steps are executed from this state.
   *
   * @param n_chains The new number of chains to execute.
   */
  void set_n_chains(uint64_t n_chains) { this->n_chains = n_chains; }

  /**
   * @brief Get the number of steps to execute on every chain.
   *
   * A single step involves proposing a random modification to the current
   * state, computing its likelihood and deciding whether the new state should
   * be the new state of the chain.
   *
   * @return uint64_t The number of steps to execute on every chain.
   */
  uint64_t get_chain_length() const { return chain_length.value(); }

  /**
   * @brief Set the number of steps to execute on every chain.
   *
   * A single step involves proposing a random modification to the current
   * state, computing its likelihood and deciding whether the new state should
   * be the new state of the chain.
   *
   * @param chain_length The new number of steps to execute on every chain.
   */
  void set_chain_length(uint64_t chain_length) {
    this->chain_length = chain_length;
  }

  /**
   * @brief Get the maximal number of optimal states to return.
   *
   * The buffer for the optimal states list has to be allocated in advance and
   * can not be reallocated. Therefore, an upper bound for the maximal number of
   * optimal states is needed.
   *
   * @return uint64_t The maximal number of optimal states to return.
   */
  uint64_t get_max_n_best_states() const { return max_n_best_states; }

  /**
   * @brief Set the maximal number of optimal states to return.
   *
   * The buffer for the optimal states list has to be allocated in advance and
   * can not be reallocated. Therefore, an upper bound for the maximal number of
   * optimal states is needed.
   *
   * @param max_n_best_states The new maximal number of optimal states to
   * return.
   */
  void set_max_n_best_states(uint64_t max_n_best_states) {
    this->max_n_best_states = max_n_best_states;
  }

  /**
   * @brief Get the mean value of the probability for false positives (alpha).
   *
   * @return double The mean value of the probability for false positives
   * (alpha).
   */
  double get_alpha_mean() const { return alpha_mean; }

  /**
   * @brief Set the mean value of the probability for false positives (alpha).
   *
   * @param alpha_mean The new mean value of the probability for false positives
   * (alpha).
   */
  void set_alpha_mean(double alpha_mean) { this->alpha_mean = alpha_mean; }

  /**
   * @brief Get the mean value for the probability for false negatives (beta).
   *
   * @return double The mean value for the probability for false negatives
   * (beta).
   */
  double get_beta_mean() const { return beta_mean; }

  /**
   * @brief Set the mean value for the probability for false negatives (beta).
   *
   * @param beta_mean The new mean value for the probability for false negatives
   * (beta).
   */
  void set_beta_mean(double beta_mean) { this->beta_mean = beta_mean; }

  /**
   * @brief Get the standard derivation for the probability for false negatives
   * (beta).
   *
   * @return double The standard derivation for the probability for false
   * negatives (beta).
   */
  double get_beta_sd() const { return beta_sd; }

  /**
   * @brief Set the standard derivation for the probability for false negatives
   * (beta).
   *
   * @param beta_sd The new standard derivation for the probability for false
   * negatives (beta).
   */
  void set_beta_sd(double beta_sd) { this->beta_sd = beta_sd; }

  /**
   * @brief Get the probability that the beta value is modified in a chain step.
   *
   * The returned value of this method is bound to the interval [0,1].
   *
   * @return double The probability that the beta value is modified in a chain
   * step.
   */
  double get_prob_beta_change() const {
    return prob_beta_change / get_sum_of_move_probs();
  }

  /**
   * @brief Set the probability that the beta value is modified in a chain step.
   *
   * The passed new probability may be any non-negative number since the
   * probabilities of the different moves are normalized.
   *
   * @param prob_beta_change The new probability that the beta value is modified
   * in a chain step.
   */
  void set_prob_beta_change(double prob_beta_change) {
    if (prob_beta_change < 0.0) {
      throw std::out_of_range("Move probabilities may not be negative.");
    }
    this->prob_beta_change = prob_beta_change;
  }

  /**
   * @brief Get the probability that a node and its subtree is moved to another
   * node in a chain step.
   *
   * The returned value of this method is bound to the interval [0,1].
   *
   * @return double The probability that a node and its subtree is moved to
   * another node in a chain step.
   */
  double get_prob_prune_n_reattach() const {
    return prob_prune_n_reattach / get_sum_of_move_probs();
  }

  /**
   * @brief Set the probability that a node and its subtree is moved to another
   * node in a chain step.
   *
   * The passed new probability may be any non-negative number since the
   * probabilities of the different moves are normalized.
   *
   * @param prob_prune_n_reattach The new probability that a node and its
   * subtree is moved to another node in a chain step.
   */
  void set_prob_prune_n_reattach(double prob_prune_n_reattach) {
    if (prob_prune_n_reattach < 0.0) {
      throw std::out_of_range("Move probabilities may not be negative.");
    }
    this->prob_prune_n_reattach = prob_prune_n_reattach;
  }

  /**
   * @brief Get the probability that two node labels are swapped in a chain
   * step.
   *
   * The returned value of this method is bound to the interval [0,1].
   *
   * @return double The probability that two node labels are swapped in a chain
   * step.
   */
  double get_prob_swap_nodes() const {
    return prob_swap_nodes / get_sum_of_move_probs();
  }

  /**
   * @brief Set the probability that two node labels are swapped in a chain
   * step.
   *
   * The passed new probability may be any non-negative number since the
   * probabilities of the different moves are normalized.
   *
   * @param prob_swap_nodes The new probability that two node labels are swapped
   * in a chain step.
   */
  void set_prob_swap_nodes(double prob_swap_nodes) {
    if (prob_swap_nodes < 0.0) {
      throw std::out_of_range("Move probabilities may not be negative.");
    }
    this->prob_swap_nodes = prob_swap_nodes;
  }

  /**
   * @brief Get the probability that two subtrees are swapped in a chain step.
   *
   * The returned value of this method is bound to the interval [0,1].
   *
   * @return double The probability that two subtrees are swapped in a chain
   * step.
   */
  double get_prob_swap_subtrees() const {
    return prob_swap_subtrees / get_sum_of_move_probs();
  }

  /**
   * @brief Set the probability that two subtrees are swapped in a chain step.
   *
   * The passed new probability may be any non-negative number since the
   * probabilities of the different moves are normalized.
   *
   * @param prob_swap_subtrees The new probability that two subtrees are swapped
   * in a chain step.
   */
  void set_prob_swap_subtrees(double prob_swap_subtrees) {
    if (prob_swap_subtrees < 0.0) {
      throw std::out_of_range("Move probabilities may not be negative.");
    }
    this->prob_swap_subtrees = prob_swap_subtrees;
  }

  /**
   * @brief Get the scaling factor to the standard derivation for beta changes.
   *
   * @return double The scaling factor to the standard derivation for beta
   * changes.
   */
  double get_beta_jump_scaling_chi() const { return beta_jump_scaling_chi; }

  /**
   * @brief Set the scaling factor to the standard derivation for beta changes.
   *
   * @param beta_jump_scaling_chi The new scaling factor to the standard
   * derivation for beta changes.
   */
  void set_beta_jump_scaling_chi(double beta_jump_scaling_chi) {
    this->beta_jump_scaling_chi = beta_jump_scaling_chi;
  }

  /**
   * @brief Get the gamma factor for the exploration behavior.
   *
   * For gamma = 1.0, the chains converge on the posterior distribution. For
   * gamma > 1.0, the chain focuses on local exploration and for gamma < 1.0,
   * the chain focuses on global exploration.
   *
   * @return double The gamma factor for the exploration behavior.
   */
  double get_gamma() const { return gamma; }

  /**
   * @brief Set the gamma factor for the exploration behavior.
   *
   * For gamma = 1.0, the chains converge on the posterior distribution. For
   * gamma > 1.0, the chain focuses on local exploration and for gamma < 1.0,
   * the chain focuses on global exploration.
   *
   * @param gamma The new gamma factor for the exploration behavior.
   */
  void set_gamma(double gamma) { this->gamma = gamma; }

  /**
   * @brief Get the seed for the URNGs.
   *
   * If the seed has not been explicitly set before, it is sampled from the
   * random device.
   *
   * @return uint32_t The seed for the URNGs.
   */
  uint32_t get_seed() const {
    if (seed.has_value()) {
      return seed.value();
    } else {
      std::random_device seeder;
      return seeder();
    }
  }

  /**
   * @brief Set the seed for the URNGs.
   *
   * @param seed The new seed for the URNGs.
   */
  void set_seed(uint32_t seed) { this->seed = seed; }

private:
  double get_sum_of_move_probs() const {
    return prob_beta_change + prob_prune_n_reattach + prob_swap_nodes +
           prob_swap_subtrees;
  }

  std::optional<std::string> input_path, output_path_base;

  std::optional<uint64_t> n_chains, chain_length;
  uint64_t max_n_best_states;

  double alpha_mean, beta_mean, beta_sd;
  double prob_beta_change, prob_prune_n_reattach, prob_swap_nodes,
      prob_swap_subtrees;
  double beta_jump_scaling_chi;
  double gamma;

  std::optional<uint32_t> seed;
};
} // namespace ffSCITE
