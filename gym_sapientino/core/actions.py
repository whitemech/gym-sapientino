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
"""Everithing concerning the various action spaces."""

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from gym_sapientino.core.objects import Robot
from gym_sapientino.utils import set_to_zero_if_small

if TYPE_CHECKING:
    from gym_sapientino.core.configurations import SapientinoAgentConfiguration


class Command(Enum):
    """Base class for command classes.

    Subclasses need to define the actual enum values.
    Do not instantiate directly.
    """

    def step(self, _robot: Robot) -> Robot:
        """Move a robot according to the command."""
        raise NotImplementedError

    @staticmethod
    def nop() -> "Command":
        """Get the NO-OP action."""
        raise NotImplementedError

    @staticmethod
    def beep() -> "Command":
        """Get the "Beep" action."""
        raise NotImplementedError


class GridCommand(Command):
    """Action space to move the agent on a grid.

    The agent moves as a point on a grid on the cardinal directions.
    """

    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    BEEP = 4
    NOP = 5

    def __str__(self) -> str:
        """Get the string representation."""
        if self == self.LEFT:
            return "<"
        elif self == self.RIGHT:
            return ">"
        elif self == self.UP:
            return "^"
        elif self == self.DOWN:
            return "v"
        elif self == self.BEEP:
            return "o"
        elif self == self.NOP:
            return "_"
        else:
            raise ValueError("Shouldn't be here...")

    def step(self, robot: Robot) -> Robot:
        """Move a robot according to the command."""
        x, y = robot.x, robot.y
        if self == self.DOWN:
            y += 1
        elif self == self.UP:
            y -= 1
        elif self == self.RIGHT:
            x += 1
        elif self == self.LEFT:
            x -= 1

        r = Robot(robot.config, x, y, robot.velocity, robot.direction.theta, robot.id)

        return r if not r._on_wall() else robot

    @staticmethod
    def nop() -> "GridCommand":
        """Get the NO-OP action."""
        return GridCommand.NOP

    @staticmethod
    def beep() -> "GridCommand":
        """Get the "Beep" action."""
        return GridCommand.BEEP


class DifferentialGridCommand(Command):
    """Action space to move the agent on a grid.

    Like GridCommand but with one difference: the agent has now a direction and
    it can go forward backward. Motion is relative to the current orientation.
    Orientations span by 90°.
    """

    LEFT = 0
    FORWARD = 1
    RIGHT = 2
    BACKWARD = 3
    BEEP = 4
    NOP = 5

    def __str__(self) -> str:
        """Get the string representation."""
        if self == self.LEFT:
            return "<"
        elif self == self.RIGHT:
            return ">"
        elif self == self.FORWARD:
            return "^"
        elif self == self.BACKWARD:
            return "v"
        elif self == self.BEEP:
            return "o"
        elif self == self.NOP:
            return "_"
        else:
            raise ValueError("Shouldn't be here...")

    def step(self, robot: Robot) -> Robot:
        """Move a robot according to the command."""
        dx = (
            1
            if robot.direction.theta == 0
            else -1
            if robot.direction.theta == 180
            else 0
        )
        dy = (
            -1
            if robot.direction.theta == 90
            else +1
            if robot.direction.theta == 270
            else 0
        )
        x, y = robot.x, robot.y
        direction = robot.direction
        if self == self.LEFT:
            direction = direction.rotate_90_left()
        elif self == self.RIGHT:
            direction = direction.rotate_90_right()
        elif self == self.FORWARD:
            x += dx
            y += dy
        elif self == self.BACKWARD:
            x -= dx
            y -= dy

        r = Robot(robot.config, x, y, robot.velocity, direction.theta, robot.id)

        return r if not r._on_wall() else robot

    @staticmethod
    def nop() -> "DifferentialGridCommand":
        """Get the NO-OP action."""
        return DifferentialGridCommand.NOP

    @staticmethod
    def beep() -> "DifferentialGridCommand":
        """Get the "Beep" action."""
        return DifferentialGridCommand.BEEP


class ContinuousCommand(Command):
    """Action space to move the agent on the plane.

    With these actions, the robot moves on the plane continuously.
    The agent can accelerate, decelerate, rotate right, left.
    """

    LEFT = 0
    FORWARD = 1
    RIGHT = 2
    BACKWARD = 3
    BEEP = 4
    NOP = 5

    def __str__(self) -> str:
        """Get the string representation."""
        cmd = ContinuousCommand(self.value)
        if cmd == ContinuousCommand.LEFT:
            return "<"
        elif cmd == ContinuousCommand.RIGHT:
            return ">"
        elif cmd == ContinuousCommand.FORWARD:
            return "^"
        elif cmd == ContinuousCommand.BACKWARD:
            return "v"
        elif cmd == ContinuousCommand.BEEP:
            return "o"
        elif cmd == ContinuousCommand.NOP:
            return "_"
        else:
            raise ValueError("Shouldn't be here...")

    def step(self, robot: Robot) -> Robot:
        """Move a robot according to the command."""
        velocity = robot.velocity
        direction = robot.direction
        x, y = robot.x, robot.y
        if self in {self.LEFT, self.RIGHT}:
            sign = 1.0 if self == self.LEFT else -1.0
            delta_theta = sign * robot.robot_config.angular_speed
            direction = robot.direction.rotate(delta_theta)
        elif self in {self.FORWARD, self.BACKWARD}:
            sign = -1.0 if self == self.BACKWARD else 1.0
            velocity += sign * robot.robot_config.acceleration
            velocity = set_to_zero_if_small(velocity)
        velocity = self.clip_velocity(velocity, robot_config=robot.robot_config)

        # Move
        r = Robot(robot.config, x, y, velocity, direction.theta, robot.id)
        r = r.apply_velocity()

        # Wall?
        if not r._on_wall():
            return r
        else:
            return Robot(robot.config, x, y, 0.0, direction.theta, robot.id)

    @staticmethod
    def clip_velocity(
        velocity: float, robot_config: "SapientinoAgentConfiguration"
    ) -> float:
        """Clip velocity."""
        return float(
            np.clip(velocity, robot_config.min_velocity, robot_config.max_velocity)
        )

    @staticmethod
    def nop() -> "ContinuousCommand":
        """Get the NO-OP action."""
        return ContinuousCommand.NOP

    @staticmethod
    def beep() -> "ContinuousCommand":
        """Get the "Beep" action."""
        return ContinuousCommand.BEEP
