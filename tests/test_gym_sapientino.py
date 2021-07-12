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
import pytest

from gym_sapientino import SapientinoDictSpace, __version__
from gym_sapientino.core.configurations import (
    SapientinoAgentConfiguration,
    SapientinoConfiguration,
)

NB_ROLLOUT_STEPS = 20

map_resource = resources.path("gym_sapientino.assets", "map1.txt")


def test_version():
    """Test version."""
    assert __version__ == "0.2.1"


@pytest.fixture(autouse=True, scope="session")
def with_rendering(request):
    """Return true if not on CI - Pygame rendering not supported."""
    result = not request.config.getoption("--ci")
    if not result:
        logging.info("Skipping rendering, because executing the test on CI.")
    return result


def test_rollout(with_rendering):
    """Test instantiation of the environment."""
    agent_config = SapientinoAgentConfiguration(initial_position=(4,4))
    env = SapientinoDictSpace(agent_configs=(agent_config,))

    env.reset()
    for _ in range(NB_ROLLOUT_STEPS):
        env.step(env.action_space.sample())
        if with_rendering:
            env.render(mode="rgb_array")

    env.close()


def sapientino_dict(
        agents_conf: Tuple[SapientinoAgentConfiguration]
    ) -> SapientinoDictSpace:
    """Create a sapientino instance from agents configurations."""
    with map_resource as map_path:
        conf = SapientinoConfiguration(
            agents_conf,
            path_to_map=map_path,
            reward_per_step=0.0,
            reward_outside_grid=0.0,
            reward_duplicate_beep=0.0,
            acceleration=0.1,
        )
        env = SapientinoDictSpace(conf)
    return env


def rollout(env: gym.Env):
    """Perform rollout."""
    env.reset()
    for _ in range(NB_ROLLOUT_STEPS):
        action = cast(gym.Space, env.action_space).sample()
        ret = env.step(action)
        logging.debug(ret)


def test_one():
    """Tests the initializtion with one agent."""
    agent_conf = SapientinoAgentConfiguration(initial_position=(3,3))
    env = sapientino_dict((agent_conf,))
    rollout(env)


# TODO

# Test with two agents


# Test with single agent wrapper


# Test observation spaces


# Test action spaces
