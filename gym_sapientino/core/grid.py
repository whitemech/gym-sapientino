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
from pathlib import Path
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
        self.counts = [[0] * self.columns for _ in range(self.columns)]

    def get_bip_counts(self, c: Cell):
        """Get counts."""
        return self.counts[c.y][c.x]

    def do_beep(self, c: Cell):
        """Do a beep for a cell."""
        self.counts[c.y][c.x] += 1

    @property
    def rows(self):
        """Get the number of rows."""
        # TODO allow different number of rows and columns.
        return len(self.cells)

    @property
    def columns(self):
        """Get the number of columns."""
        # TODO allow different number of rows and columns.
        return len(self.cells[0])

    def iter_cells(self) -> Iterator[Cell]:
        """Iterate over cells."""
        for i in range(len(self.cells)):
            for j in range(len(self.cells[i])):
                yield self.cells[i][j]


def _from_character_to_color(char: str) -> Colors:
    """From character to cell."""
    assert len(char) == 1, "only single character are accepted."
    assert char in id2color, f"character not supported: '{char}'"
    return id2color[char]


def from_map(path_to_map: Path) -> SapientinoGrid:
    """Get a grid from a map."""
    content = path_to_map.read_text()
    cells_str = [list(line[:-1]) for line in content.splitlines(keepends=False)]
    assert len(cells_str) > 0, "No row found."
    assert len(cells_str[0]) > 0, "No column found."
    nb_rows, nb_columns = len(cells_str), len(cells_str[0])
    cells = []
    for i in range(nb_rows):
        row = []
        for j in range(nb_columns):
            assert len(cells_str[i]) == nb_columns, "Got rows of different size"
            cell = cells_str[i][j]
            color = _from_character_to_color(cell)
            row.append(Cell(j, i, color))
        cells.append(row)
    grid = SapientinoGrid(cells)
    return grid
