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

"""Tests for the Sapientino Gym environment."""
import logging
from importlib import resources
from typing import Tuple, cast

import gym

from gym_sapientino import SapientinoDictSpace, __version__
from gym_sapientino.core.configurations import (
    SapientinoAgentConfiguration,
    SapientinoConfiguration,
)
import gym_sapientino.assets

NB_ROLLOUT_STEPS = 20

map_str = resources.read_text(gym_sapientino.assets, "map1.txt")


def test_version():
    """Test version."""
    assert __version__ == "0.2.1"


def sapientino_dict(
        agents_conf: Tuple[SapientinoAgentConfiguration, ...]
    ) -> SapientinoDictSpace:
    """Create a sapientino instance from agents configurations."""
    conf = SapientinoConfiguration(
        agents_conf,
        grid_map=map_str,
        reward_per_step=0.0,
        reward_outside_grid=0.0,
        reward_duplicate_beep=0.0,
        acceleration=0.1,
    )
    env = SapientinoDictSpace(conf)
    return env


def rollout(env: gym.Env):
    """Perform rollout."""
    observation_space = cast(gym.Space, env.observation_space)
    logging.debug(observation_space)
    env.reset()
    for _ in range(NB_ROLLOUT_STEPS):
        action = cast(gym.Space, env.action_space).sample()
        ret = env.step(action)
        logging.debug(ret)
        assert observation_space.contains(ret[0])


def test_one():
    """Tests the initialization with one agent."""
    agent_conf = SapientinoAgentConfiguration(initial_position=(3,3))
    env = sapientino_dict((agent_conf,))
    rollout(env)


def test_two():
    """Test with two agents."""
    agents_conf = (
        SapientinoAgentConfiguration(initial_position=(3,3)),
        SapientinoAgentConfiguration(initial_position=(3,4)),
    )
    env = sapientino_dict(agents_conf)
    rollout(env)



# TODO


# Test with single agent wrapper


# Test observation spaces


# Test action spaces
