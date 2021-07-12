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
from typing import Sequence, Tuple, Type

import numpy as np
from gym.spaces import Discrete, MultiDiscrete
from gym.spaces import Tuple as GymTuple

import gym_sapientino.assets as assets
from importlib import resources
from gym_sapientino.core.actions import Command, GridCommand
from gym_sapientino.core.grid import SapientinoGrid, from_map
from gym_sapientino.core.types import color2int
from gym_sapientino.core.constants import DEFAULT_MAP_NAME


@dataclass(frozen=True)
class SapientinoAgentConfiguration:
    """Configuration for a single agent.

    We can set its initial position and the type of commands (action space)
    it accepts. The default set of action is GridCommand. But we can use any
    other class of the core.actions module or subclasses of Command.
    """

    initial_position: Tuple[float, float]
    commands: Type[Command] = GridCommand

    @property
    def action_space(self) -> Discrete:
        """Get the action space.."""
        return Discrete(len(self.commands))

    def get_action(self, action: int) -> Command:
        """Get the action."""
        return self.commands(action)


@dataclass(frozen=True)
class SapientinoConfiguration:
    """A class to represent Sapientino configurations."""

    # game configurations
    agent_configs: Tuple[SapientinoAgentConfiguration, ...]
    grid_map: str = resources.read_text(assets, DEFAULT_MAP_NAME)
    reward_outside_grid: float = -1.0
    reward_duplicate_beep: float = -1.0
    reward_per_step: float = -0.01
    angular_speed: float = 20.0
    acceleration: float = 0.02
    max_velocity: float = 0.20
    # TODO: Velocity is maybe specific to the single agent

    def __post_init__(self):
        """
        Post init.

        Load the map.
        """
        grid = from_map(self.grid_map)
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
    def observation_space(self) -> GymTuple:
        """Get the observation space."""
        # TODO: We could push action spaces to be wrappers or subclasses of dictspace

        def get_observation_space(agent_config):
            postfix = 2, self.nb_colors
            if agent_config.differential:
                return MultiDiscrete((self.columns, self.rows, self.nb_theta) + postfix)
            return MultiDiscrete((self.columns, self.rows) + postfix)

        return GymTuple(tuple(map(get_observation_space, self.agent_configs)))

    @property
    def action_space(self) -> GymTuple:
        """Get the action space of the robots."""
        spaces = tuple(Discrete(ac.action_space.n) for ac in self.agent_configs)
        return GymTuple(spaces)

    @property
    def nb_theta(self):
        """Get the number of orientations."""
        return 4

    @property
    def nb_colors(self):
        """Get the number of colors."""
        return len(color2int)

    def get_action(self, actions: Sequence[int]) -> Sequence[Command]:
        """Get the action."""
        return [ac.get_action(a) for a, ac in zip(actions, self.agent_configs)]

    def clip_velocity(self, velocity: float) -> float:
        """Clip velocity."""
        # TODO: is this the right place
        return float(np.clip(velocity, -self.max_velocity, self.max_velocity))
