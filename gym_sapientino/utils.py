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

"""This module contains utility functions."""
from functools import reduce

import numpy as np
from gymnasium import spaces


# TODO: maybe prefer numpy functions for encode decode
def encode(obs: list[int], spaces: list[int]) -> int:
    """
    Encode an observation from a list of gym.Discrete spaces in one number.

    :param obs: an observation belonging to the state space (a list of gym.Discrete spaces)
    :param spaces: the list of gym.Discrete spaces from where the observation is observed.
    :return: the encoded observation.
    """
    assert len(obs) == len(spaces)
    sizes = spaces
    result = obs[0]
    shift = sizes[0]
    for o, size in list(zip(obs, sizes))[1:]:
        result += o * shift
        shift *= size

    return result


def decode(obs: int, spaces: list[int]) -> list[int]:
    """
    Decode an observation from a list of gym.Discrete spaces in a list of integers.

    It assumes that obs has been encoded by using the 'utils.encode' function.

    :param obs: the encoded observation
    :param spaces: the list of gym.Discrete spaces from where the observation is observed.
    :return: the decoded observation.
    """
    result = []
    sizes = spaces[::-1]
    shift = reduce(lambda x, y: x * y, sizes) // sizes[0]
    for size in sizes[1:]:
        r = obs // shift
        result.append(r)
        obs %= shift
        shift //= size

    result.append(obs)
    return result[::-1]


def set_to_zero_if_small(x) -> float:
    """Set to zero if it is a small number."""
    return 0.0 if np.isclose(x, 0.0) else x


def combine_boxes(*boxes: spaces.Box) -> spaces.Box:
    """Combine a list of gym.Box spaces into one.

    It merges a list of unidimensional boxes to one unidimensional box by
    combining along the only dimension. Limits are kept separate.
    Output type is np.float32.
    """
    # Unidimensional spaces
    assert all(len(space.shape) == 1 for space in boxes)

    # Concat
    lows = np.concatenate([space.low for space in boxes])
    highs = np.concatenate([space.high for space in boxes])

    return spaces.Box(lows, highs)
