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
"""Useful gym wrappers."""

import gym
from gym.spaces import Tuple as GymTuple


class SingleAgentWrapper(gym.Wrapper):
    """Wrapper for multi-agent OpenAI Gym environment to make it single-agent.

    It adapts a multi-agent OpenAI Gym environment with just one agent
    to be used as a single agent environment.
    In particular, this means that if the observation space and the
    action space are tuples of one space, the new
    spaces will remove the tuples and return the unique space.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the wrapper."""
        super().__init__(*args, **kwargs)

        self.observation_space = self._transform_tuple_space(self.observation_space)
        self.action_space = self._transform_tuple_space(self.action_space)

    def _transform_tuple_space(self, space: GymTuple):
        """Transform a Tuple space with one element into that element."""
        assert isinstance(
            space, GymTuple
        ), "The space is not an instance of gym.spaces.tuples.Tuple."
        assert len(space.spaces) == 1, "The tuple space has more than one subspaces."
        return space.spaces[0]

    def step(self, action):
        """Do a step."""
        state, reward, done, info = super().step([action])
        new_state = state[0]
        return new_state, reward, done, info

    def reset(self, **kwargs):
        """Do a step."""
        state = super().reset(**kwargs)
        new_state = state[0]
        return new_state
