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

"""Pygame-based rendering."""
from typing import Any, Callable, Dict, Type

import pygame

from gym_sapientino.core.constants import white
from gym_sapientino.core.objects import Cell, Robot, SapientinoGrid
from gym_sapientino.core.states import SapientinoState
from gym_sapientino.core.types import Colors
from gym_sapientino.rendering.base import Renderer

ROBOT_COLORS = [
    "red",
    "lightblue",
    "orange",
    "yellow",
    "white",
    "rosybrown",
    "pink",
]


class PygameRenderer(Renderer):
    """Pygame-based renderer."""

    def __init__(
        self,
        state: SapientinoState,
        offx: int = 40,
        offy: int = 100,
        radius: int = 5,
        size_square: int = 40,
    ):
        """Initialize the Pygame-based renderer."""
        super().__init__(state)

        # rendering configurations
        self.offx: int = offx
        self.offy: int = offy
        self.radius: int = radius
        self.size_square: int = size_square

        self._type_to_handler: Dict[Type, Callable] = {Robot: self._draw_robot}

        pygame.init()
        pygame.display.set_caption("Sapientino")
        self._screen = pygame.display.set_mode([self.win_width, self.win_height])
        self.myfont = pygame.font.SysFont("Arial", 30)

    @property
    def win_width(self) -> int:
        """Get the window width."""
        return self.size_square * self.config.columns + self.offx * 2

    @property
    def win_height(self) -> int:
        """Get the window height."""
        return self.size_square * self.config.rows + self.offy + (self.offy // 2)

    def render(self, mode="human") -> None:
        """Render."""
        self._fill_screen()
        self._draw_score_label()
        self._draw_last_command()
        self._draw_game_objects()

        if mode == "human":
            pygame.display.update()
        elif mode == "rgb_array":
            screen = pygame.surfarray.array3d(self._screen)
            # swap width with height
            return screen.swapaxes(0, 1)

    def reset(self, state: "SapientinoState") -> None:
        """Reset the state."""
        self._state = state

    def close(self):
        """Close the renderer."""
        pygame.display.quit()
        pygame.quit()

    def _fill_screen(self):
        self._screen.fill(white)

    def _draw_score_label(self):
        score_label = self.myfont.render(
            str(self.state.score), 100, pygame.color.THECOLORS["black"]
        )
        self._screen.blit(score_label, (20, 10))

    def _draw_last_command(self):
        cmds = self.state.last_commands
        s = "".join(f"{str(c)}" for c in cmds)
        count_label = self.myfont.render(s, 100, pygame.color.THECOLORS["brown"])
        self._screen.blit(count_label, (60, 10))

    def _draw_game_objects(self):
        self._draw_grid(self.state.grid)
        for r in self.state.robots:
            self._draw_robot(r)

    def _get_handler(self, arg: Any) -> Callable:
        arg_type = type(arg)
        return self._type_to_handler.get(arg_type, lambda arg: None)

    def draw(self, arg) -> None:
        """Draw an object."""
        handler = self._get_handler(arg)
        handler(arg)

    def _draw_robot(self, r: Robot) -> None:
        """Draw a robot."""
        dx = int(self.offx + r.x * self.size_square)
        dy = int(self.offy + (self.config.rows - r.y - 1) * self.size_square)
        pygame.draw.circle(
            self._screen,
            ROBOT_COLORS[r.id],
            [dx + self.size_square // 2, dy + self.size_square // 2],
            2 * self.radius,
            0,
        )
        ox = 0
        oy = 0
        if r.direction.th == 0:  # right
            ox = self.radius
        elif r.direction.th == 90:  # up
            oy = -self.radius
        elif r.direction.th == 180:  # left
            ox = -self.radius
        elif r.direction.th == 270:  # down
            oy = self.radius

        pygame.draw.circle(
            self._screen,
            pygame.color.THECOLORS["black"],
            [dx + self.size_square // 2 + ox, dy + self.size_square // 2 + oy],
            5,
            0,
        )

    def _draw_grid(self, g: SapientinoGrid):
        """Draw the grid."""
        for i in range(0, g.columns + 1):
            ox = self.offx + i * self.size_square
            pygame.draw.line(
                self._screen,
                pygame.color.THECOLORS["black"],
                [ox, self.offy],
                [ox, self.offy + g.rows * self.size_square],
            )

        for i in range(0, g.rows + 1):
            oy = self.offy + i * self.size_square
            pygame.draw.line(
                self._screen,
                pygame.color.THECOLORS["black"],
                [self.offx, oy],
                [self.offx + g.columns * self.size_square, oy],
            )

        for cell in g.cells.values():
            self._draw_cell(cell)

    def _draw_cell(self, c: Cell) -> None:
        """Draw a cell."""
        if c.color == Colors.BLANK:
            return
        dx = int(self.offx + c.x * self.size_square)
        dy = int(self.offy + (self.config.rows - c.y - 1) * self.size_square)
        sqsz = (
            dx + 5,
            dy + 5,
            self.size_square - 10,
            self.size_square - 10,
        )
        pygame.draw.rect(self._screen, pygame.color.THECOLORS[str(c.color)], sqsz)
        if c.bip_count >= 1:
            pygame.draw.rect(
                self._screen,
                pygame.color.THECOLORS["black"],
                (
                    dx + 15,
                    dy + 15,
                    self.size_square - 30,
                    self.size_square - 30,
                ),
            )
