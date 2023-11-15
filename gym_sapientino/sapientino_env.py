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

from abc import ABC, abstractmethod
from typing import Optional

from gymnasium import Env, Space

from gym_sapientino.core.configurations import SapientinoConfiguration
from gym_sapientino.core.states import SapientinoState, make_state
from gym_sapientino.rendering.pygame import PygameRenderer


class Sapientino(Env, ABC):
    """The Sapientino Gym environment.

    Subclasses must define an observation_space.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        configuration: Optional[SapientinoConfiguration] = None,
        render_mode: Optional[str] = None,
        *args,
        **kwargs
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

    def reset(self):
        """Reset the environment."""
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
