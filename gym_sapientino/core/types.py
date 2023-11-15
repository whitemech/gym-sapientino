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

"""Define basic types."""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

import numpy as np

from gym_sapientino.utils import set_to_zero_if_small


@dataclass
class Direction:
    """A class to represent the direction."""

    theta: float

    def rotate(self, delta_theta: float):
        """Rotate of a certain amount."""
        th = self.theta + delta_theta
        if th < 0:
            th = 360.0 + th
        elif th >= 360.0:
            th = th % 360.0
        return Direction(th)

    def rotate_left(self, delta_theta: float) -> "Direction":
        """Rotate left of a certain amount."""
        if delta_theta < 0.0:
            raise ValueError("Only positive values are allowed.")
        return self.rotate(delta_theta)

    def rotate_90_left(self) -> "Direction":
        """Rotate the direction to the left."""
        return self.rotate_left(90.0)

    def rotate_right(self, delta_theta: float) -> "Direction":
        """Rotate right of a certain amount."""
        if delta_theta < 0.0:
            raise ValueError("Only positive values are allowed.")
        return self.rotate(-delta_theta)

    def rotate_90_right(self) -> "Direction":
        """Rotate the direction to the right."""
        return self.rotate_right(90.0)

    def sincos(self) -> Tuple[float, float]:
        """Return the pair (sin(theta), cos(theta)."""
        rad_theta = np.deg2rad(self.theta)
        sin_theta = set_to_zero_if_small(np.sin(rad_theta))
        cos_theta = set_to_zero_if_small(np.cos(rad_theta))
        return sin_theta, cos_theta


class Colors(Enum):
    """Enumeration for colors."""

    BLANK = "blank"
    WALL = "wall"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    PINK = "pink"
    BROWN = "brown"
    GRAY = "gray"
    PURPLE = "purple"
    ORANGE = "orange"

    def __str__(self) -> str:
        """Get the string representation."""
        return self.value

    def __int__(self) -> int:
        """Get the integer representation."""
        return color2int[self]


color2int: Dict[Colors, int] = {c: i for i, c in enumerate(list(Colors))}
id2color: Dict[str, Colors] = {
    " ": Colors.BLANK,
    "#": Colors.WALL,
    "r": Colors.RED,
    "g": Colors.GREEN,
    "b": Colors.BLUE,
    "y": Colors.YELLOW,
    "p": Colors.PINK,
    "o": Colors.ORANGE,
    "B": Colors.BROWN,
    "G": Colors.GRAY,
    "P": Colors.PURPLE,
}
color2id = dict(map(reversed, id2color.items()))  # type: ignore
