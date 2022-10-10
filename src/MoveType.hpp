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

namespace ffSCITE {
/**
 * @brief The different move types that the proposer may propose.
 */
enum class MoveType {
  /**
   * @brief Add a normally distributed value to the current beta value.
   */
  ChangeBeta,
  /**
   * @brief Sample one node, remove the edge from it's old parent and attach it
   * to another node.
   */
  PruneReattach,
  /**
   * @brief Sample two nodes and swap their labels.
   */
  SwapNodes,
  /**
   * @brief Sample two nodes, remove the edges to their old parents and attach
   * them to the opposite, previous parents.
   */
  SwapSubtrees,
};
} // namespace ffSCITE
