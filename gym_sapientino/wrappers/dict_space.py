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

"""Sapientino environments using a "dict" state space."""

import gym
from gym.spaces import Dict, Discrete

from gym_sapientino.core.states import SapientinoState
from gym_sapientino.sapientino_env import Sapientino


class SapientinoDictSpace(Sapientino):
    """
    A Sapientino environment with a dictionary state space.

    The components of the space are:
    - Robot x coordinate (Discrete)
    - Robot y coordinate (Discrete)
    - The orientation (Discrete)
    - A boolean to check whether the last action was a beep (Discrete)
    - The color of the current cell (Discrete)
    """

    def __init__(self, *args, **kwargs):
        """Initialize the dictionary space."""
        super().__init__(*args, **kwargs)

        self._x_space = Discrete(self.configuration.columns)
        self._y_space = Discrete(self.configuration.rows)
        self._theta_space = Discrete(self.configuration.nb_theta)
        self._beep_space = Discrete(2)
        self._color_space = Discrete(self.configuration.nb_colors)

    @property
    def observation_space(self) -> gym.spaces.Tuple:
        """Get the observation space."""

        def get_agent_dict_space(i: int):
            agent_config = self.configuration.agent_configs[i]
            d = {
                "x": self._x_space,
                "y": self._y_space,
                "beep": self._beep_space,
                "color": self._color_space,
            }
            if agent_config.differential:
                d["theta"] = self._theta_space
            return Dict(d)

        return gym.spaces.Tuple(
            tuple(map(get_agent_dict_space, range(self.configuration.nb_robots)))
        )

    def observe(self, state: SapientinoState):
        """Observe the state."""
        return state.to_dict()
