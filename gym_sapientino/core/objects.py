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

from typing import Tuple

import numpy as np

from gym_sapientino.core.configurations import SapientinoConfiguration
from gym_sapientino.core.types import (
    COMMAND_TYPES,
    Colors,
    ContinuousCommand,
    DifferentialCommand,
    Direction,
    NormalCommand,
)
from gym_sapientino.utils import set_to_zero_if_small


class Robot:
    """A class to represent a robot."""

    def __init__(
        self,
        config: SapientinoConfiguration,
        x: float,
        y: float,
        velocity: float,
        theta: float,
        ang_velocity: float,
        id_: int,
    ):
        """Initialize the robot."""
        self.config = config
        self._id = id_
        self.x = x
        self.y = y
        self.velocity = velocity
        self.direction = Direction(theta)
        self.ang_velocity = ang_velocity

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
        return Robot(
            self.config,
            x,
            y,
            self.velocity,
            self.direction.theta,
            self.ang_velocity,
            self.id,
        )

    def apply_velocity(self) -> "Robot":
        """Apply the velocity to change position."""
        new_x, new_y = self.x, self.y
        sin, cos = self.direction.sincos()
        new_x += self.velocity * cos
        new_y += -self.velocity * sin
        new_dir = self.direction.rotate(self.ang_velocity).theta
        return Robot(
            self.config,
            new_x,
            new_y,
            self.velocity,
            new_dir,
            self.ang_velocity,
            self.id,
        )

    def step(self, command: COMMAND_TYPES) -> "Robot":
        """Compute the next location."""
        if isinstance(command, NormalCommand):
            return self._step_normal(command)
        elif isinstance(command, DifferentialCommand):
            return self._step_differential(command)
        elif isinstance(command, ContinuousCommand):
            return self._step_continuous(command)
        else:
            raise ValueError("Command not recognized.")

    def _step_normal(self, command: NormalCommand) -> "Robot":
        x, y = self.x, self.y
        if command == command.DOWN:
            y += 1
        elif command == command.UP:
            y -= 1
        elif command == command.RIGHT:
            x += 1
        elif command == command.LEFT:
            x -= 1

        new_robot = Robot(
            self.config,
            x,
            y,
            self.velocity,
            self.direction.theta,
            self.ang_velocity,
            self.id,
        )

        # Check if not wall
        if new_robot._on_wall():
            return self

        return new_robot

    def _step_differential(self, command: DifferentialCommand) -> "Robot":
        dx = (
            1 if self.direction.theta == 0 else -1 if self.direction.theta == 180 else 0
        )
        dy = (
            -1
            if self.direction.theta == 90
            else +1
            if self.direction.theta == 270
            else 0
        )
        x, y = self.x, self.y
        direction = self.direction
        if command == command.LEFT:
            direction = direction.rotate_90_left()
        elif command == command.RIGHT:
            direction = direction.rotate_90_right()
        elif command == command.FORWARD:
            x += dx
            y += dy
        elif command == command.BACKWARD:
            x -= dx
            y -= dy

        new_robot = Robot(
            self.config,
            x,
            y,
            self.velocity,
            direction.theta,
            self.ang_velocity,
            self.id,
        )

        # Check if not wall
        if new_robot._on_wall():
            return self

        return new_robot

    def _step_continuous(self, command: ContinuousCommand) -> "Robot":
        velocity = self.velocity
        direction = self.direction
        ang_velocity = self.ang_velocity
        x, y = self.x, self.y
        if command in {command.LEFT, command.RIGHT}:
            sign = 1.0 if command == command.LEFT else -1.0
            ang_velocity += sign * self.config.angular_acceleration
            ang_velocity = set_to_zero_if_small(ang_velocity)
        elif command in {command.FORWARD, command.BACKWARD}:
            sign = -0.5 if command == command.BACKWARD else 1.0
            velocity += sign * self.config.acceleration
            velocity = set_to_zero_if_small(velocity)

        velocity = self.config.clip_velocity(velocity)
        ang_velocity = self.config.clip_angular_velocity(ang_velocity)

        r = Robot(self.config, x, y, velocity, direction.theta, ang_velocity, self.id)
        new_robot = r.apply_velocity()

        # Check if not wall
        if new_robot._on_wall():
            return self

        return new_robot

    @property
    def encoded_theta(self) -> int:
        """Encode the theta."""
        theta_integer = int(self.direction.theta)
        return theta_integer // 90

    def _on_wall(self) -> bool:
        """Check if the coordinate of this robot corresponds to a wall."""
        x, y = self.discrete_x, self.discrete_y
        cell = self.config.grid.cells[y][x]
        return cell.color == Colors.WALL
