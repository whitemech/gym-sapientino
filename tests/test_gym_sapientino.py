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
import pytest

from gym_sapientino import SapientinoDictSpace, __version__
from gym_sapientino.core.configurations import SapientinoAgentConfiguration

NB_ROLLOUT_STEPS = 20


def test_version():
    """Test version."""
    assert __version__ == "0.2.0"


@pytest.mark.parametrize(
    "differential,continuous", [(False, False), (True, False), (False, True)]
)
def test_rollout(differential, continuous):
    """Test instantiation of the environment."""
    agent_config = SapientinoAgentConfiguration(
        differential=differential, continuous=continuous
    )
    env = SapientinoDictSpace(agent_configs=(agent_config,))

    env.reset()
    for _ in range(NB_ROLLOUT_STEPS):
        env.step(env.action_space.sample())
        env.render(mode="rgb_array")
