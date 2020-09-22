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

import itertools
from collections import defaultdict
from typing import Dict, Tuple

from gym_sapientino.core.configurations import SapientinoConfiguration
from gym_sapientino.core.constants import TOKENS
from gym_sapientino.core.types import (
    COMMAND_TYPES,
    Colors,
    DifferentialCommand,
    Direction,
    NormalCommand,
    color2int,
)


class Robot:
    """A class to represent a robot."""

    def __init__(self, config: SapientinoConfiguration, id_: int = 0):
        """Initialize the robot."""
        self.config = config

        self._id = id_

        self._initial_x = 3
        self._initial_y = 2
        self._initial_th = 90

        self.x = self._initial_x
        self.y = self._initial_y
        self.direction = Direction(self._initial_th)

    @property
    def id(self) -> int:
        """Get the robot's id."""
        return self._id

    @property
    def position(self) -> Tuple[int, int]:
        """Get the position."""
        return self.x, self.y

    def reset(self) -> None:
        """Reset the robot."""
        self.x = self._initial_x
        self.y = self._initial_y
        self.direction = Direction(self._initial_th)

    def step(self, command: COMMAND_TYPES):
        """Execute a command."""
        if isinstance(command, NormalCommand):
            self._step_normal(command)
        elif isinstance(command, DifferentialCommand):
            self._step_differential(command)
        else:
            raise ValueError("Command not recognized.")

    def _step_normal(self, command: NormalCommand):
        if command == command.DOWN:
            self.y -= 1
        elif command == command.UP:
            self.y += 1
        elif command == command.RIGHT:
            self.x += 1
        elif command == command.LEFT:
            self.x -= 1

    def _step_differential(self, command: DifferentialCommand):
        dx = 1 if self.direction.th == 0 else -1 if self.direction.th == 180 else 0
        dy = 1 if self.direction.th == 90 else -1 if self.direction.th == 270 else 0
        if command == command.LEFT:
            self.direction = self.direction.rotate_left()
        elif command == command.RIGHT:
            self.direction = self.direction.rotate_right()
        elif command == command.FORWARD:
            self.x += dx
            self.y += dy
        elif command == command.BACKWARD:
            self.x -= dx
            self.y -= dy

    @property
    def encoded_theta(self) -> int:
        """Encode the theta."""
        return self.direction.th // 90


class Cell:
    """A class to represent a cell on the grid."""

    def __init__(self, config: SapientinoConfiguration, x: int, y: int, color: Colors):
        """Initialize the cell."""
        self.config = config
        self.x = x
        self.y = y
        self.color = color
        self.bip_count = 0

    def reset(self) -> None:
        """Reset the cell."""
        self.bip_count = 0

    @property
    def encoded_color(self) -> int:
        """Encode the color."""
        return color2int[self.color]

    def beep(self) -> None:
        """Do a beep."""
        self.bip_count += 1


class BlankCell(Cell):
    """A blank cell."""

    def __init__(self, config: SapientinoConfiguration, x: int, y: int):
        """Initialize the blank cell."""
        super().__init__(config, x, y, Colors.BLANK)


class SapientinoGrid:
    """The grid of the Sapientino environment."""

    def __init__(self, config: SapientinoConfiguration):
        """Initialize the grid."""
        self.config = config

        self.rows = config.rows
        self.columns = config.columns

        self.cells: Dict[Tuple[int, int], Cell] = {}
        self.color_count: Dict[Colors, int] = defaultdict(lambda: 0)

        self._populate_token_grid()

    def _populate_token_grid(self):
        # add color cells
        for t in TOKENS:
            x, y = t[2], t[3]
            color = t[1]
            color_cell = Cell(self.config, x, y, Colors(color))
            self.cells[(x, y)] = color_cell

        # add blank cells
        for x, y in itertools.product(
            range(self.config.columns), range(self.config.rows)
        ):
            if (x, y) not in self.cells:
                self.cells[(x, y)] = BlankCell(self.config, x, y)

    def reset(self) -> None:
        """Reset the grid."""
        for t in self.cells.values():
            t.reset()
