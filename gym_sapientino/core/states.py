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
from abc import ABC
from typing import Dict, List, Sequence, Tuple

from numpy import clip

from gym_sapientino.core.configurations import SapientinoConfiguration
from gym_sapientino.core.grid import Cell, SapientinoGrid
from gym_sapientino.core.objects import Robot
from gym_sapientino.core.types import Colors
from gym_sapientino.core.actions import Command


class SapientinoState(ABC):
    """Abstract class to represent a Sapientino state."""

    def __init__(self, config: "SapientinoConfiguration"):
        """Initialize the state."""
        self.config = config

        self.score = 0
        self._grid = self.config.grid
        self._grid.reset()
        self._robots: List[Robot] = [
            Robot(config, c.initial_position[0], c.initial_position[1], 0.0, 90.0, i)
            for i, c in enumerate(self.config.agent_configs)
        ]
        self._last_commands: List[Command] = [
            ac.commands.nop() for ac in self.config.agent_configs
        ]

    @property
    def grid(self) -> SapientinoGrid:
        """Return the grid."""
        return self._grid

    @property
    def robots(self) -> Tuple[Robot, ...]:
        """Get the list of robots."""
        return tuple(self._robots)

    def step(self, commands: Sequence[Command]) -> float:
        """Do a step."""
        assert len(commands) == len(self.robots), "Some commands are missing."
        total_reward = 0.0

        next_robots = [c.step(r) for c, r in zip(commands, self.robots)]
        self._last_commands = list(commands)

        for i in range(len(next_robots)):
            reward, next_robots[i] = self._force_border_constraints(next_robots[i])
            total_reward += reward

            total_reward += self._do_beep(next_robots[i], commands[i])

        total_reward += self.config.reward_per_step
        self._robots = next_robots
        return total_reward

    def reset(self) -> "SapientinoState":
        """Reset the state."""
        return type(self)(self.config)

    @property
    def is_finished(self) -> bool:
        """Check whether the game is finished."""
        return False

    @property
    def current_cells(self) -> Sequence[Cell]:
        """Get the current cell."""
        result = []
        for r in self._robots:
            result.append(self.grid.cells[r.discrete_y][r.discrete_x])
        return result

    @property
    def last_commands(self) -> Sequence[Command]:
        """Get the list of last commands."""
        return self._last_commands

    def to_dict(self) -> Tuple[Dict, ...]:
        """Encode into a dictionary."""
        return tuple(
            {
                "discrete_x": round(r.x),
                "discrete_y": round(r.y),
                "x": r.x,
                "y": r.y,
                "velocity": r.velocity,
                "theta": r.encoded_theta,
                "angle": r.direction.theta,
                "beep": int(self.last_commands[i] == self.last_commands[i].beep()),
                "color": self.current_cells[i].encoded_color,
            }
            for i, r in enumerate(self.robots)
        )

    def _force_border_constraints(self, r: Robot) -> Tuple[float, Robot]:
        reward = 0.0
        x, y = r.x, r.y
        if not (0 <= r.x < self.config.columns - 1):
            reward += self.config.reward_outside_grid
            x = int(clip(r.x, 0, self.config.columns - 1))
            r.velocity = 0.0
        if not (0 <= r.y < self.config.rows - 1):
            reward += self.config.reward_outside_grid
            y = int(clip(r.y, 0, self.config.rows - 1))
            r.velocity = 0.0
        return reward, r.move(x, y)

    def _do_beep(self, robot: Robot, command: Command) -> float:
        reward = 0.0
        if command == command.beep():
            cell = self.grid.cells[robot.discrete_y][robot.discrete_x]
            self.grid.do_beep(cell)
            if cell.color != Colors.BLANK:
                if cell.color not in self.grid.color_count:
                    self.grid.color_count[cell.color] = 0
                self.grid.color_count[cell.color] += 1
            if self.grid.get_bip_counts(cell) >= 2:
                reward += self.config.reward_duplicate_beep

        return reward


def make_state(config: SapientinoConfiguration) -> SapientinoState:
    """Make the state, according to the configuration."""
    return SapientinoState(config)
