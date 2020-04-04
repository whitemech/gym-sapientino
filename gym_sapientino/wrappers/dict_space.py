# -*- coding: utf-8 -*-

"""Sapientino environments using a "dict" state space."""

from gym.spaces import Dict, Discrete

from gym_sapientino.sapientino_env import Sapientino, SapientinoState, SapientinoConfiguration, Direction


class SapientinoDictSpace(Sapientino):
    """A Breakout environment with a dictionary state space.
    The components of the space are:
    - Robot x coordinate (Discrete)
    - Robot y coordinate (Discrete)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._x_space = Discrete(self.configuration.columns)
        self._y_space = Discrete(self.configuration.rows)
        self._theta_space = Discrete(self.configuration.nb_theta)
        self._beep_space = Discrete(2)
        self._color_space = Discrete(self.configuration.nb_colors)

    @property
    def observation_space(self):
        return Dict({
            "x": self._x_space,
            "y": self._y_space,
            "theta": self._theta_space,
            "beep": self._beep_space,
            "color": self._color_space
        })

    def observe(self, state: SapientinoState):
        """Observe the state."""
        return state.to_dict()

