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

    def __init__(self, state: SapientinoState):
        """Initialize the Pygame-based renderer."""
        super().__init__(state)

        self._type_to_handler: Dict[Type, Callable] = {Robot: self._draw_robot}

        pygame.init()
        pygame.display.set_caption("Sapientino")
        self._screen = pygame.display.set_mode(
            [self.config.win_width, self.config.win_height]
        )
        self.myfont = pygame.font.SysFont("Arial", 30)

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
        dx = int(r.config.offx + r.x * r.config.size_square)
        dy = int(r.config.offy + (r.config.rows - r.y - 1) * r.config.size_square)
        pygame.draw.circle(
            self._screen,
            ROBOT_COLORS[r.id],
            [dx + r.config.size_square // 2, dy + r.config.size_square // 2],
            2 * r.config.radius,
            0,
        )
        ox = 0
        oy = 0
        if r.direction.th == 0:  # right
            ox = r.config.radius
        elif r.direction.th == 90:  # up
            oy = -r.config.radius
        elif r.direction.th == 180:  # left
            ox = -r.config.radius
        elif r.direction.th == 270:  # down
            oy = r.config.radius

        pygame.draw.circle(
            self._screen,
            pygame.color.THECOLORS["black"],
            [
                dx + r.config.size_square // 2 + ox,
                dy + r.config.size_square // 2 + oy,
            ],
            5,
            0,
        )

    def _draw_grid(self, g: SapientinoGrid):
        """Draw the grid."""
        for i in range(0, g.columns + 1):
            ox = g.config.offx + i * g.config.size_square
            pygame.draw.line(
                self._screen,
                pygame.color.THECOLORS["black"],
                [ox, g.config.offy],
                [ox, g.config.offy + g.rows * g.config.size_square],
            )

        for i in range(0, g.rows + 1):
            oy = g.config.offy + i * g.config.size_square
            pygame.draw.line(
                self._screen,
                pygame.color.THECOLORS["black"],
                [g.config.offx, oy],
                [g.config.offx + g.columns * g.config.size_square, oy],
            )

        for cell in g.cells.values():
            self._draw_cell(cell)

    def _draw_cell(self, c: Cell) -> None:
        """Draw a cell."""
        if c.color == Colors.BLANK:
            return
        dx = int(c.config.offx + c.x * c.config.size_square)
        dy = int(c.config.offy + (c.config.rows - c.y - 1) * c.config.size_square)
        sqsz = (
            dx + 5,
            dy + 5,
            c.config.size_square - 10,
            c.config.size_square - 10,
        )
        pygame.draw.rect(self._screen, pygame.color.THECOLORS[str(c.color)], sqsz)
        if c.bip_count >= 1:
            pygame.draw.rect(
                self._screen,
                pygame.color.THECOLORS["black"],
                (
                    dx + 15,
                    dy + 15,
                    c.config.size_square - 30,
                    c.config.size_square - 30,
                ),
            )
