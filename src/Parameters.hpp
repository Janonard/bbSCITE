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
class Parameters {
public:
  Parameters(int argc, char **argv, uint64_t max_n_cells, uint64_t max_n_genes)
      : input_path(std::nullopt), output_path_base(std::nullopt),
        n_cells(std::nullopt), n_genes(std::nullopt), n_chains(std::nullopt),
        chain_length(std::nullopt), max_n_best_states(128), alpha(6.04e-5),
        beta(0.4309), prob_beta_change(0.0), prob_prune_n_reattach(0.55),
        prob_swap_nodes(0.4), prob_swap_subtrees(0.05), beta_sd(0.1),
        beta_jump_scaling_chi(10.0), gamma(1.0), seed(std::nullopt) {

    bool error = false;
    /*
     * This style of argument parsing is WET and errorprone, but necessary to
     * stay CLI-compatible with the original SCITE application. I have copied it
     * from their implementation, and added some errors and warnings.
     */
    for (int i = 1; i < argc; ++i) {

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
        if (i + 1 < argc) {
          n_genes = atoi(argv[++i]);
        } else {
          std::cerr << "Error: Missing argument to parameter -n." << std::endl;
          error = true;
        }
      } else if (strcmp(argv[i], "-m") == 0) {
        if (i + 1 < argc) {
          n_cells = atoi(argv[++i]);
        } else {
          std::cerr << "Error: Missing argument to parameter -m." << std::endl;
          error = true;
        }
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
          alpha = atof(argv[++i]);
        } else {
          std::cerr << "Error: Missing argument to parameter -fd." << std::endl;
          error = true;
        }
      } else if (strcmp(argv[i], "-ad") == 0) {
        if (i + 1 < argc) {
          beta = atof(argv[++i]);
        } else {
          std::cerr << "Error: Missing argument to parameter -ad." << std::endl;
          error = true;
        }
        if (i + 1 < argc) {
          std::string next = argv[i + 1];
          if (next.compare(0, 1, "-") != 0) {
            beta += atof(argv[++i]);
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
        } else {
          std::cerr << "Error: Missing argument to parameter -move_probs."
                    << std::endl;
          error = true;
        }
        if (i + 1 < argc) {
          prob_swap_nodes = atof(argv[++i]);
        } else {
          std::cerr << "Error: Missing argument to parameter -move_probs."
                    << std::endl;
          error = true;
        }
        if (i + 1 < argc) {
          prob_swap_subtrees = atof(argv[++i]);
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

    if (!n_cells.has_value()) {
      std::cerr << "Error: Missing number of cells. Did you forget to set the "
                   "-m parameter?"
                << std::endl;
      error = true;
    }

    if (!n_genes.has_value()) {
      std::cerr << "Error: Missing number of genes. Did you forget to set the "
                   "-n parameter?"
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

    if (n_cells > max_n_cells) {
      std::cerr << "Error: The number of cells is too big. This build of "
                   "ffSCITE only supports up to "
                << max_n_cells << " cells." << std::endl;
      error = true;
    }

    if (n_genes > max_n_genes) {
      std::cerr << "Error: The number of genes is too big. This build of "
                   "ffSCITE only supports up to "
                << max_n_genes << " genes." << std::endl;
      error = true;
    }

    if (error) {
      exit(1);
    }
  }

  std::string get_input_path() const { return input_path.value(); }

  std::string get_output_path_base() const {
    if (output_path_base.has_value()) {
      return output_path_base.value();
    } else {
      int lastIndex = input_path->find_last_of(".");
      return input_path->substr(0, lastIndex);
    }
  }

  uint64_t get_n_cells() const { return n_cells.value(); }

  uint64_t get_n_genes() const { return n_genes.value(); }

  uint64_t get_n_chains() const { return n_chains.value(); }

  uint64_t get_chain_length() const { return chain_length.value(); }

  uint64_t get_max_n_best_states() const { return max_n_best_states; }

  double get_alpha() const { return alpha; }

  double get_beta() const { return beta; }

  double get_beta_sd() const { return beta_sd; }

  double get_sum_of_move_probs() const {
    return prob_beta_change + prob_prune_n_reattach + prob_swap_nodes +
           prob_swap_subtrees;
  }

  double get_prob_beta_change() const {
    return prob_beta_change / get_sum_of_move_probs();
  }

  double get_prob_prune_n_reattach() const {
    return prob_prune_n_reattach / get_sum_of_move_probs();
  }

  double get_prob_swap_nodes() const {
    return prob_swap_nodes / get_sum_of_move_probs();
  }

  double get_prob_swap_subtrees() const {
    return prob_swap_subtrees / get_sum_of_move_probs();
  }

  double get_beta_jump_sd() const { return beta_sd / beta_jump_scaling_chi; }

  double get_gamma() const { return gamma; }

  uint32_t get_seed() const {
    if (seed.has_value()) {
      return seed.value();
    } else {
      std::random_device seeder;
      return seeder();
    }
  }

private:
  std::optional<std::string> input_path, output_path_base;

  std::optional<uint64_t> n_cells, n_genes, n_chains, chain_length;
  uint64_t max_n_best_states;

  double alpha, beta;
  double prob_beta_change, prob_prune_n_reattach, prob_swap_nodes,
      prob_swap_subtrees;
  double beta_sd;
  double beta_jump_scaling_chi;
  double gamma;

  std::optional<uint32_t> seed;
};
} // namespace ffSCITE
