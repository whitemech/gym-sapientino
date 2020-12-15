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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import gym
import numpy as np
from gym.spaces import Discrete, MultiDiscrete
from gym.spaces import Tuple as GymTuple

from gym_sapientino.core.constants import ASSETS_DIR, DEFAULT_MAP_FILENAME
from gym_sapientino.core.grid import SapientinoGrid, from_map
from gym_sapientino.core.types import (
    ACTION_TYPE,
    COMMAND_ENUM_TYPES,
    COMMAND_TYPES,
    ContinuousCommand,
    DifferentialCommand,
    NormalCommand,
    color2int,
)


@dataclass(frozen=True)
class SapientinoAgentConfiguration:
    """
    Configuration for a single agent.

    By default, the agent moves on the grid cell by cell,
    with the action space as LEFT-UP-RIGHT-DOWN.

    If differential is true, the agent can move forward and
    backward, and can turn left and right (but on the same cell).
    (continuous must be false).

    If continuous is True, the agent can speed up, slow down,
    turn left and turn right. "differential" is ignored.
    """

    differential: bool = False
    continuous: bool = False
    initial_position: Optional[Tuple[float, float]] = None

    @property
    def action_type(self) -> COMMAND_ENUM_TYPES:
        """Get the action enumeration."""
        if self.continuous:
            return ContinuousCommand
        if self.differential:
            return DifferentialCommand
        return NormalCommand

    @property
    def action_space(self) -> gym.spaces.Discrete:
        """Get the action space.."""
        return Discrete(len(self.action_type))

    def get_action(self, action: int) -> COMMAND_TYPES:
        """Get the action."""
        return self.action_type(action)  # type: ignore


@dataclass(frozen=True)
class SapientinoConfiguration:
    """A class to represent Sapientino configurations."""

    # game configurations
    agent_configs: Tuple[SapientinoAgentConfiguration, ...] = (
        SapientinoAgentConfiguration(),
    )
    path_to_map: Path = ASSETS_DIR / DEFAULT_MAP_FILENAME
    reward_outside_grid: float = -1.0
    reward_duplicate_beep: float = -1.0
    reward_per_step: float = -0.01
    angular_speed: float = 20.0
    acceleration: float = 0.02
    max_velocity: float = 0.20

    def __post_init__(self):
        """
        Post init.

        Load the map.
        """
        # accept string for path_to_map
        object.__setattr__(self, "path_to_map", Path(self.path_to_map))
        grid = from_map(self.path_to_map)
        object.__setattr__(self, "_grid", grid)

    @property
    def grid(self) -> SapientinoGrid:
        """Return the grid."""
        return self._grid  # type: ignore

    @property
    def rows(self) -> int:
        """Get the number of rows."""
        return self.grid.rows

    @property
    def columns(self) -> int:
        """Get the number of columns."""
        return self.grid.columns

    @property
    def nb_robots(self) -> int:
        """Get the number of robots."""
        return len(self.agent_configs)

    @property
    def agent_config(self) -> "SapientinoAgentConfiguration":
        """Get the agent configuration."""
        assert self.nb_robots == 1, "Can be called only in single-agent mode."
        return self.agent_configs[0]

    @property
    def observation_space(self) -> gym.spaces.Tuple:
        """Get the observation space."""

        def get_observation_space(agent_config):
            postfix = 2, self.nb_colors
            if agent_config.differential:
                return MultiDiscrete((self.columns, self.rows, self.nb_theta) + postfix)
            return MultiDiscrete((self.columns, self.rows) + postfix)

        return GymTuple(tuple(map(get_observation_space, self.agent_configs)))

    @property
    def action_space(self) -> gym.spaces.Tuple:
        """Get the action space of the robots."""
        spaces = tuple(Discrete(ac.action_space.n) for ac in self.agent_configs)
        return gym.spaces.Tuple(spaces)

    @property
    def nb_theta(self):
        """Get the number of orientations."""
        return 4

    @property
    def nb_colors(self):
        """Get the number of colors."""
        return len(color2int)

    def get_action(self, action) -> ACTION_TYPE:
        """Get the action."""
        return [ac.get_action(a) for a, ac in zip(action, self.agent_configs)]

    def clip_velocity(self, velocity: float) -> float:
        """Clip velocity."""
        return float(np.clip(velocity, -self.max_velocity, self.max_velocity))
