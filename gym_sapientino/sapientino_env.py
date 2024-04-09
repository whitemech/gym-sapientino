# -*- coding: utf-8 -*-
#
# Copyright 2019-2023 Marco Favorito, Roberto Cipollone, Luca Iocchi
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

"""Sapientino environment with OpenAI Gym interface."""

import random
import sys
from abc import ABC, abstractmethod
from typing import Optional

from gymnasium import Env, Space
from gymnasium.spaces import Box, Dict, Discrete, Tuple

from gym_sapientino.core.configurations import SapientinoConfiguration
from gym_sapientino.core.states import SapientinoState, make_state
from gym_sapientino.rendering.pygame import PygameRenderer


class SapientinoBase(Env, ABC):
    """The base class for the sapientino Gym environment.

    Subclasses must define an observation_space.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        configuration: Optional[SapientinoConfiguration] = None,
        render_mode: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the environment.

        :param configuration: the configuration object.
        :param render_mode: see metadata["render_modes"]. If None, rendering is off (default).
        """
        self.configuration = (
            configuration
            if configuration is not None
            else SapientinoConfiguration(*args, **kwargs)
        )
        self.render_mode = render_mode
        self.state = make_state(self.configuration)
        self.viewer = PygameRenderer(self.state) if render_mode else None
        self.action_space = self.configuration.action_space

    def step(self, action):
        """Execute an action."""
        command = self.configuration.get_action(action)
        reward = self.state.step(command)
        obs = self.observe(self.state)
        is_finished = self.state.is_finished
        if self.render_mode == "human":
            self.render()
        return obs, reward, is_finished, False, {}

    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        if seed:
            self.rng = random.Random(seed)  # nosec
        self.state = make_state(self.configuration)
        if self.viewer is not None:
            self.viewer.reset(self.state)
            self.render()
        return self.observe(self.state), {}

    def render(self):
        """Render the environment."""
        if not self.render_mode:
            return
        if self.viewer is None:
            self.viewer = PygameRenderer(self.state)
        return self.viewer.render(mode=self.render_mode)

    def close(self) -> None:
        """Close the environment."""
        if self.viewer is not None:
            self.viewer.close()

    @abstractmethod
    def observe(self, state: SapientinoState) -> Space:
        """
        Extract observation from the state of the game.

        :param state: the state of the game
        :return: an instance of a gym.Space
        """

    def __getstate__(self):
        """Get the state."""
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        state.pop("viewer")
        return state

    def __setstate__(self, state):
        """Set the state."""
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)
        self.viewer = None


class Sapientino(SapientinoBase):
    """A Sapientino environment with a dictionary state space.

    The components of the space are:
    - Robot x coordinate (Discrete)
    - Robot y coordinate (Discrete)
    - The orientation (Discrete)
    - A boolean to check whether the last action was a beep (Discrete)
    - The color of the current cell (Discrete)
    """

    def __init__(
        self,
        configuration: Optional[SapientinoConfiguration] = None,
        **kwargs,
    ):
        """Initialize the dictionary space."""
        super().__init__(configuration=configuration, **kwargs)  # type: ignore

        self._discrete_x_space = Discrete(self.configuration.columns)
        self._discrete_y_space = Discrete(self.configuration.rows)
        self._x_space = Box(0.0, self.configuration.columns, shape=[1])
        self._y_space = Box(0.0, self.configuration.rows, shape=[1])
        self._velocity_space = lambda m, M: Box(m, M, shape=[1])
        self._theta_space = lambda n: Discrete(n)
        self._angle_space = Box(0.0, 360.0 - sys.float_info.epsilon, shape=[1])
        self._beep_space = Discrete(2)
        self._color_space = Discrete(self.configuration.nb_colors)

        self.observation_space = Tuple(
            [
                Dict(
                    {
                        "discrete_x": self._discrete_x_space,
                        "discrete_y": self._discrete_y_space,
                        "x": self._x_space,
                        "y": self._y_space,
                        "velocity": self._velocity_space(
                            self.configuration.agent_configs[i].min_velocity,
                            self.configuration.agent_configs[i].max_velocity,
                        ),
                        "theta": self._theta_space(
                            self.configuration.agent_configs[i].angle_parts,
                        ),
                        "angle": self._angle_space,
                        "beep": self._beep_space,
                        "color": self._color_space,
                    }
                )
                for i in range(self.configuration.nb_robots)
            ]
        )

    def observe(self, state: SapientinoState):
        """Observe the state."""
        return state.to_dict()
