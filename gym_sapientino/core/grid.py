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

"""Classes to represent a Sapientino map."""
from typing import Dict, Iterator, List

from gym_sapientino.core.types import Colors, color2int, id2color


class Cell:
    """A class to represent a cell on the grid."""

    def __init__(self, x: int, y: int, color: Colors):
        """Initialize the cell."""
        self.x = x
        self.y = y
        self.color = color

    @property
    def encoded_color(self) -> int:
        """Encode the color."""
        return color2int[self.color]


class SapientinoGrid:
    """The grid of the Sapientino environment."""

    def __init__(self, cells: List[List[Cell]]):
        """Initialize the grid."""
        self.cells: List[List[Cell]] = cells
        self.color_count: Dict[Colors, int] = {}
        self.counts: List[List[int]] = []
        self.reset()

    def reset(self):
        """Reset the state of the grid."""
        self.color_count: Dict[Colors, int] = {}
        self.counts = [[0] * self.columns for _ in range(self.rows)]

    def get_bip_counts(self, c: Cell):
        """Get counts."""
        return self.counts[c.y][c.x]

    def do_beep(self, c: Cell):
        """Do a beep for a cell."""
        self.counts[c.y][c.x] += 1

    @property
    def rows(self):
        """Get the number of rows."""
        return len(self.cells)

    @property
    def columns(self):
        """Get the number of columns."""
        return len(self.cells[0])

    def iter_cells(self) -> Iterator[Cell]:
        """Iterate over cells."""
        for i in range(len(self.cells)):
            for j in range(len(self.cells[i])):
                yield self.cells[i][j]


def _from_character_to_color(char: str) -> Colors:
    """From character to cell."""
    if len(char) != 1:
        raise ValueError("only single character are accepted.")
    return id2color[char]


def from_map(map_str: str) -> SapientinoGrid:
    """
    Get a grid from a map.

    The function expects a string representation of a map.
    Each rows of characters correspond to a row of cells in the grid of the environment.
    The allowed characters are:
    - ' ', the empty cell;
    - '#', a wall;
    - 'r', a cell with color red;
    - 'g', a cell with color green;
    - 'b', a cell with color blue;
    - 'y', a cell with color yellow;
    - 'p', a cell with color pink;
    - 'o', a cell with color orange;
    - 'B', a cell with color brown;
    - 'G', a cell with color gray;
    - 'P', a cell with color purple;

    The character '|' will be ignored. It is useful as a separator of rows
    during the editing (some text editors might remove trailing spaces of a row).

    An example of grid map is:

      |P bB g |
      | bp G r|
      |G   pg |
      | rpG PB|
      |rP Bg b|

    """
    content = map_str.replace("|", "")
    cells_str = content.splitlines(keepends=False)
    if len(cells_str) <= 0:
        raise ValueError("No row found.")
    if len(cells_str[0]) <= 0:
        raise ValueError("No column found.")
    nb_rows, nb_columns = len(cells_str), len(cells_str[0])
    cells = []
    for i in range(nb_rows):
        row = []
        for j in range(nb_columns):
            if len(cells_str[i]) != nb_columns:
                raise ValueError("Got rows of different size")
            cell = cells_str[i][j]
            color = _from_character_to_color(cell)
            row.append(Cell(j, i, color))
        cells.append(row)
    grid = SapientinoGrid(cells)
    return grid
