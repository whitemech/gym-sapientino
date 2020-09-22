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

"""Sapientino environment with OpenAI Gym interface."""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import gym as gym
import pygame

from gym_sapientino.core.configurations import SapientinoConfiguration
from gym_sapientino.core.states import (
    SapientinoState,
    SapientinoStateSingleRobot,
    make_state,
)
from gym_sapientino.rendering.base import Renderer
from gym_sapientino.rendering.pygame import PygameRenderer


class Sapientino(gym.Env, ABC):
    """The Sapientino Gym environment."""

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, configuration: Optional[SapientinoConfiguration] = None):
        """Initialize the environment."""
        self.configuration = (
            configuration if configuration is not None else SapientinoConfiguration()
        )
        self.state = make_state(self.configuration)
        self.viewer: Optional[Renderer] = None

    @property
    def action_space(self) -> gym.Space:
        """Get the action space."""
        return self.configuration.action_space

    @property
    def observation_space(self) -> gym.Space:
        """Get the observation space."""
        return self.configuration.action_space

    def step(self, action: int):
        """Execute an action."""
        command = self.configuration.get_action(action)
        reward = self.state.step(command)
        obs = self.observe(self.state)
        is_finished = self.state.is_finished
        info: Dict = {}
        return obs, reward, is_finished, info

    def reset(self):
        """Reset the environment."""
        self.state = SapientinoStateSingleRobot(self.configuration)
        if self.viewer is not None:
            self.viewer.reset(self.state)
        return self.observe(self.state)

    def render(self, mode="human") -> None:
        """Render the environment."""
        if self.viewer is None:
            self.viewer = PygameRenderer(self.state)

        return self.viewer.render(mode=mode)

    def close(self) -> None:
        """Close the environment."""
        if self.viewer is not None:
            self.viewer.close()

    @abstractmethod
    def observe(self, state: SapientinoState) -> gym.Space:
        """
        Extract observation from the state of the game.

        :param state: the state of the game
        :return: an instance of a gym.Space
        """

    def play(self) -> None:
        """Play interactively with the environment."""
        print("Press 'Q' to quit.")
        assert self.configuration.nb_robots == 1, "Can only play with one robot."
        self.reset()
        self.render()
        quitted = False
        while not quitted:
            event = pygame.event.wait()
            cmd = 5
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    quitted = True
                elif event.key == pygame.K_LEFT:
                    cmd = 0
                elif event.key == pygame.K_UP:
                    cmd = 1
                elif event.key == pygame.K_RIGHT:
                    cmd = 2
                elif event.key == pygame.K_DOWN:
                    cmd = 3
                elif event.key == pygame.K_SPACE:
                    cmd = 4

                self.step(cmd)
                self.render()
        self.close()
