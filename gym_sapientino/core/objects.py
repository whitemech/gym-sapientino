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

"""Objects of the game."""

from typing import TYPE_CHECKING, Tuple

import numpy as np

from gym_sapientino.core.types import Colors, Direction

if TYPE_CHECKING:
    from gym_sapientino.core.actions import Command
    from gym_sapientino.core.configurations import SapientinoConfiguration


class Robot:
    """A class to represent a robot."""

    def __init__(
        self,
        config: "SapientinoConfiguration",
        x: float,
        y: float,
        velocity: float,
        theta: float,
        id_: int,
    ):
        """Initialize the robot."""
        self.config = config
        self.robot_config = config.agent_configs[id_]
        self._id = id_
        self.x = x
        self.y = y
        self.velocity = velocity
        self.direction = Direction(theta)

    @property
    def id(self) -> int:
        """Get the robot's id."""
        return self._id

    @property
    def discrete_x(self) -> int:
        """Get the discrete x coordinate."""
        rounded_x = int(np.round(self.x))
        x = min(rounded_x, self.config.columns - 1)
        return x

    @property
    def discrete_y(self) -> int:
        """Get the discrete y coordinate."""
        rounded = int(np.round(self.y))
        y = min(rounded, self.config.rows - 1)
        return y

    @property
    def position(self) -> Tuple[float, float]:
        """Get the position."""
        return self.x, self.y

    def move(self, x: float, y: float) -> "Robot":
        """Move to a location."""
        return Robot(self.config, x, y, self.velocity, self.direction.theta, self.id)

    def apply_velocity(self) -> "Robot":
        """Apply the velocity to change position."""
        new_x, new_y = self.x, self.y
        sin, cos = self.direction.sincos()
        new_x += self.velocity * cos
        new_y += -self.velocity * sin
        return Robot(
            self.config, new_x, new_y, self.velocity, self.direction.theta, self.id
        )

    def step(self, command: "Command") -> "Robot":
        """Compute the next location."""
        return command.step(self)

    @property
    def encoded_theta(self) -> int:
        """Encode the theta."""
        angle_step = 360 / self.robot_config.angle_parts
        return int(self.direction.theta / angle_step)

    def _on_wall(self) -> bool:
        """Check if the coordinate correspond to a wall or outside the map."""
        x, y = self.discrete_x, self.discrete_y
        if x < 0 or x >= self.config.rows:
            return True
        if y < 0 or y >= self.config.columns:
            return True
        cell = self.config.grid.cells[y][x]
        return cell.color == Colors.WALL
