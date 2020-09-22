# -*- coding: utf-8 -*-
#
# Copyright 2019-2020 Marco Favorito, Luca Iocchi
#
# ------------------------------
#
# This file is part of gym-sapientino.
#
# gym-sapientino is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gym-sapientino is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gym-sapientino.  If not, see <https://www.gnu.org/licenses/>.
#

"""Classes for the environment configurations."""
from typing import Optional

import gym
from gym.spaces import Discrete, MultiDiscrete

from gym_sapientino.core.types import (
    COMMAND_TYPES,
    DifferentialCommand,
    Direction,
    NormalCommand,
    color2int,
)


class SapientinoConfiguration:
    """A class to represent Sapientino configurations."""

    def __init__(
        self,
        rows: int = 5,
        columns: int = 7,
        nb_robots: int = 1,
        differential: bool = False,
        horizon: Optional[int] = None,
        reward_outside_grid: float = -1.0,
        reward_duplicate_beep: float = -1.0,
        reward_per_step: float = -0.01,
    ):
        """Initialize the configurations."""
        self.rows = rows
        self.columns = columns
        self.nb_robots = nb_robots
        self.differential = differential
        self._horizon = horizon if horizon else (self.columns * self.rows) * 10
        self.reward_outside_grid = reward_outside_grid
        self.reward_duplicate_beep = reward_duplicate_beep
        self.reward_per_step = reward_per_step

        self.offx = 40
        self.offy = 100
        self.radius = 5
        self.size_square = 40

    @property
    def win_width(self) -> int:
        """Get the window width."""
        if self.columns > 10:
            return self.size_square * (self.columns - 10)
        else:
            return 480

    @property
    def win_height(self) -> int:
        """Get the window height."""
        if self.rows > 10:
            return self.size_square * (self.rows - 10)
        else:
            return 520

    @property
    def action_space(self) -> gym.Space:
        """Get the action space.."""
        if self.differential:
            return Discrete(len(DifferentialCommand))
        else:
            return Discrete(len(NormalCommand))

    @property
    def observation_space(self) -> gym.Space:
        """Get the observation space."""
        if self.differential:
            # 4 is the number of possible direction - nord, sud, west, east
            return MultiDiscrete((self.columns, self.rows, Direction.NB_DIRECTIONS))
        else:
            return MultiDiscrete((self.columns, self.rows))

    def get_action(self, action: int) -> COMMAND_TYPES:
        """Get the action."""
        if self.differential:
            return DifferentialCommand(action)
        else:
            return NormalCommand(action)

    @property
    def horizon(self) -> int:
        """Get the horizon."""
        return self._horizon

    @property
    def nb_theta(self):
        """Get the number of orientations."""
        return Direction.NB_DIRECTIONS

    @property
    def nb_colors(self):
        """Get the number of colors."""
        return len(color2int)
