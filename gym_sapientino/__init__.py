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

"""The 'Sapientino' OpenAI Gym Environment."""

__version__ = "0.4.0"

import gymnasium as gym

from .core import actions
from .wrappers import SapientinoDictSpace, observations

gym.register(
    "Sapientino-v0",
    entry_point="gym_sapientino:SapientinoDictSpace",
    max_episode_steps=100,
)
