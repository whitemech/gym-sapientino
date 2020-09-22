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

"""State representiations for different Sapientino game."""

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

from numpy import clip

from gym_sapientino.core.configurations import SapientinoConfiguration
from gym_sapientino.core.objects import Cell, Robot, SapientinoGrid
from gym_sapientino.core.types import (
    COMMAND_TYPES,
    Colors,
    DifferentialCommand,
    Direction,
    NormalCommand,
)


class SapientinoState(ABC):
    """Abstract class to represent a Sapientino state."""

    def __init__(self, config: "SapientinoConfiguration"):
        """Initialize the state."""
        self.config = config

        self.score = 0
        self._grid = SapientinoGrid(config)
        self._robots: List[Robot] = []

    @property
    def grid(self) -> SapientinoGrid:
        """Return the grid."""
        return self._grid

    @property
    def robots(self) -> Tuple[Robot, ...]:
        """Get the list of robots."""
        return tuple(self._robots)

    @abstractmethod
    def step(self, command: COMMAND_TYPES) -> float:
        """Do a step."""
        raise NotImplementedError

    def reset(self) -> "SapientinoState":
        """Reset the state."""
        return type(self)(self.config)

    @property
    @abstractmethod
    def is_finished(self) -> bool:
        """Check whether the game is finished."""

    @property
    @abstractmethod
    def last_commands(self) -> Sequence[COMMAND_TYPES]:
        """Get the list of last commands."""


class SapientinoStateSingleRobot(SapientinoState):
    """The state of the game (one robot)."""

    def __init__(self, config: SapientinoConfiguration):
        """Initialize the state."""
        super().__init__(config)

        assert config.nb_robots == 1, "Can support only one robot."
        self._robots = [Robot(config, 3, 2, Direction.UP, 0)]
        self.last_command: COMMAND_TYPES = (
            NormalCommand.NOP if config.differential else DifferentialCommand.NOP
        )

    @property
    def robot(self) -> Robot:
        """Gee the robot."""
        return self._robots[0]

    def step(self, command: COMMAND_TYPES) -> float:
        """Do a step."""
        reward = 0.0

        self._robots[0] = self.robot.step(command)
        self.last_command = command

        if not (0 <= self.robot.x < self.config.columns):
            reward += self.config.reward_outside_grid
            self.robot.x = int(clip(self.robot.x, 0, self.config.columns - 1))
        if not (0 <= self.robot.y < self.config.rows):
            reward += self.config.reward_outside_grid
            self.robot.y = int(clip(self.robot.y, 0, self.config.rows - 1))

        if command == command.BEEP:
            position = self.robot.x, self.robot.y
            cell = self.grid.cells[position]
            cell.beep()
            if cell.color != Colors.BLANK:
                self.grid.color_count[cell.color] += 1
            if cell.bip_count >= 2:
                reward += self.config.reward_duplicate_beep

        reward += self.config.reward_per_step
        return reward

    @property
    def last_command_beep(self) -> bool:
        """Return whether the last command was a beep."""
        return self.last_command.value == 4

    def to_dict(self) -> dict:
        """Encode into a dictionary."""
        return {
            "x": self.robot.x,
            "y": self.robot.y,
            "theta": self.robot.encoded_theta,
            "beep": int(self.last_command_beep),  # 0 or 1
            "color": self.current_cell.encoded_color,
        }

    @property
    def current_cell(self) -> Cell:
        """Get the current cell."""
        return self.grid.cells[self.robot.position]

    @property
    def is_finished(self) -> bool:
        """Check whether the game has ended."""
        return False

    @property
    def last_commands(self) -> Sequence[COMMAND_TYPES]:
        """Get last commands."""
        return [self.last_command]


def make_state(config: SapientinoConfiguration) -> SapientinoState:
    """Make the state, according to the configuration."""
    if config.nb_robots == 1:
        return SapientinoStateSingleRobot(config)
    raise ValueError
