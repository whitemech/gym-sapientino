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

"""Base classes for the rendering layer."""

from abc import ABC, abstractmethod
from typing import Any

from gym_sapientino.core.configurations import SapientinoConfiguration
from gym_sapientino.core.states import SapientinoState


class Renderer(ABC):
    """Abstract renderer."""

    def __init__(self, state: SapientinoState):
        """
        Initialize the renderer.

        :param state: the game state.
        """
        self._state = state

    @property
    def state(self) -> SapientinoState:
        """Get the state."""
        return self._state

    @property
    def config(self) -> SapientinoConfiguration:
        """Get the configuration."""
        return self._state.config

    @abstractmethod
    def reset(self, state: "SapientinoState") -> None:
        """
        Reset the viewer.

        :param state: the state to reset the viewer to.
        :return: None
        """

    @abstractmethod
    def render(self, mode: str = "human") -> Any:
        """
        Render the state of the game.

        :param mode: the rendering mode.
        :return: None
        """

    @abstractmethod
    def close(self):
        """Close the viewer."""
