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
#include "MoveType.hpp"
#include <cinttypes>

namespace ffSCITE {

struct ChainStepParameters {
  uint32_t v, w;
  uint32_t parent_of_v, parent_of_w;
  uint32_t descendant_of_v, nondescendant_of_v;
  MoveType move_type;

  float new_beta;

  float tree_swap_neighborhood_correction;
  float acceptance_level;
};

} // namespace ffSCITE