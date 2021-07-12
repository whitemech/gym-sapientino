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
import time
from importlib import resources
from typing import Tuple, cast

import gym
from gym import spaces

import gym_sapientino.assets
from gym_sapientino import SapientinoDictSpace, __version__
from gym_sapientino.core import actions
from gym_sapientino.core.actions import Command
from gym_sapientino.core.configurations import (
    SapientinoAgentConfiguration,
    SapientinoConfiguration,
)
from gym_sapientino.core.objects import Robot
from gym_sapientino.wrappers import observations
from gym_sapientino.wrappers.gym import SingleAgentWrapper

NB_ROLLOUT_STEPS = 20

map_str = resources.read_text(gym_sapientino.assets, "map1.txt")
rendering = False  # NOTE: enable this to see agents move


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
        if rendering:
            time.sleep(0.2)
            env.render()
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


def test_single_agent():
    """Test a simplified observation space for one agent only."""
    env = sapientino_dict(
        agents_conf=(SapientinoAgentConfiguration(initial_position=(3,3)),)
    )
    env = SingleAgentWrapper(env)
    assert isinstance(env.action_space, spaces.Discrete)
    assert isinstance(env.observation_space, spaces.Dict)
    rollout(env)


def test_discrete_default():
    """Test the discrete action space (default)."""
    agent_conf = SapientinoAgentConfiguration(initial_position=(3,3))
    env = sapientino_dict(
        agents_conf=(agent_conf,)
    )
    assert agent_conf.commands == actions.GridCommand, "Wrong default"
    rollout(env)


def test_discrete_differential():
    """Test the differential command action space."""
    agent_conf = SapientinoAgentConfiguration(
        initial_position=(3,3),
        commands=actions.DifferentialGridCommand,
    )
    env = sapientino_dict(
        agents_conf=(agent_conf,)
    )
    rollout(env)


def test_continuous():
    """Test continuous motions."""
    agent_conf = SapientinoAgentConfiguration(
        initial_position=(3,3),
        commands=actions.ContinuousCommand,
    )
    env = sapientino_dict(
        agents_conf=(agent_conf,)
    )
    rollout(env)


def test_different_action_spaces():
    """Test agents with different action spaces."""
    agents_conf = (
        SapientinoAgentConfiguration(
            initial_position=(3,3),
            commands=actions.GridCommand,
        ),
        SapientinoAgentConfiguration(
            initial_position=(3,4),
            commands=actions.DifferentialGridCommand,
        ),
        SapientinoAgentConfiguration(
            initial_position=(3,2),
            commands=actions.ContinuousCommand,
        ),
    )
    env = sapientino_dict(agents_conf)
    rollout(env)


def test_discrete_features():
    """Test discrete features."""
    agent_conf = SapientinoAgentConfiguration(
        initial_position=(3,3),
        commands=actions.ContinuousCommand,
    )
    env = sapientino_dict(
        agents_conf=(agent_conf,)
    )
    env = observations.UseFeatures(
        env=env,
        features=[observations.DiscreteFeatures],
    )
    assert isinstance(env.observation_space, spaces.Tuple)
    assert len(env.observation_space) == 1
    assert isinstance(env.observation_space[0], spaces.MultiDiscrete)
    rollout(env)


def test_differential_features():
    """Test differential features."""
    agent_conf = SapientinoAgentConfiguration(
        initial_position=(3,3),
        commands=actions.ContinuousCommand,
    )
    env = sapientino_dict(
        agents_conf=(agent_conf,)
    )
    env = observations.UseFeatures(
        env=env,
        features=[observations.DiscreteAngleFeatures],
    )
    assert isinstance(env.observation_space, spaces.Tuple)
    assert len(env.observation_space) == 1
    assert isinstance(env.observation_space[0], spaces.MultiDiscrete)
    rollout(env)


def test_continuous_features():
    """Test continuous features."""
    agent_conf = SapientinoAgentConfiguration(
        initial_position=(3,3),
        commands=actions.ContinuousCommand,
    )
    env = sapientino_dict(
        agents_conf=(agent_conf,)
    )
    env = observations.UseFeatures(
        env=env,
        features=[observations.ContinuousFeatures],
    )
    assert isinstance(env.observation_space, spaces.Tuple)
    assert len(env.observation_space) == 1
    assert isinstance(env.observation_space[0], spaces.Box)
    rollout(env)


def test_all_features():
    """Test all features."""
    agents_conf = (
        SapientinoAgentConfiguration(
            initial_position=(3,3),
            commands=actions.GridCommand,
        ),
        SapientinoAgentConfiguration(
            initial_position=(3,4),
            commands=actions.DifferentialGridCommand,
        ),
        SapientinoAgentConfiguration(
            initial_position=(3,2),
            commands=actions.ContinuousCommand,
        ),
    )
    env = sapientino_dict(agents_conf)
    env = observations.UseFeatures(
        env=env,
        features=[
            observations.DiscreteFeatures,
            observations.DiscreteAngleFeatures,
            observations.ContinuousFeatures,
        ],
    )
    assert isinstance(env.observation_space, spaces.Tuple)
    assert len(env.observation_space) == 3
    assert isinstance(env.observation_space[0], spaces.MultiDiscrete)
    assert isinstance(env.observation_space[1], spaces.MultiDiscrete)
    assert isinstance(env.observation_space[2], spaces.Box)
    rollout(env)


class Differential45Command(Command):
    """Command with rotations fo 45 degrees."""

    LEFT = 0
    FORWARD = 1
    RIGHT = 2
    BEEP = 3
    NOP = 4

    def step(self, robot: Robot) -> Robot:
        """Move a robot according to the command."""
        sin, cos = robot.direction.sincos()
        dx = (
            1
            if cos > 0.1
            else -1
            if cos < -0.1
            else 0
        )
        dy = (
            -1
            if sin > 0.1
            else 1
            if sin < -0.1
            else 0
        )
        x, y = robot.x, robot.y
        direction = robot.direction
        if self == self.LEFT:
            direction = direction.rotate(45.0)
        elif self == self.RIGHT:
            direction = direction.rotate(-45.0)
        elif self == self.FORWARD:
            x += dx
            y += dy

        r = Robot(robot.config, x, y, robot.velocity, direction.theta, robot.id)

        return r if not r._on_wall() else robot

    @staticmethod
    def nop() -> "Differential45Command":
        """Get the NO-OP action."""
        return Differential45Command.NOP

    @staticmethod
    def beep() -> "Differential45Command":
        """Get the "Beep" action."""
        return Differential45Command.BEEP

def test_45_deg():
    """Test action space of 45 degrees."""
    agents_conf = (
        SapientinoAgentConfiguration(
            initial_position=(3,3),
            commands=Differential45Command,
            angle_parts=8,
        ),
        SapientinoAgentConfiguration(
            initial_position=(3,4),
            commands=actions.DifferentialGridCommand,
            angle_parts=4,
        ),
    )
    env = sapientino_dict(agents_conf)
    rollout(env)
