"""Definitions of observations spaces.

SapientinoDictSpace is the most generic class, which returns all
the information in form of a dictionary. Features classes define an observation
space and extract information accordingly.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import gym
import numpy as np
from gym import spaces

from gym_sapientino import SapientinoDictSpace, utils

DictSpace = Dict[str, Any]


class Features(ABC, gym.Wrapper):
    """Base class for all observation spaces.

    By subclassing from this class, we can define any observation space
    computed over SapientinoDictSpace.
    The member "dict_state" stores last observation received from
    SapientinoDictSpace.
    """

    def __init__(self, env: SapientinoDictSpace):
        """Initialize."""
        gym.Wrapper.__init__(self, env)
        assert isinstance(self.env, SapientinoDictSpace)
        self.dict_space = env.observation_space
        self.observation_space = self.compute_space()

    def reset(self, **kwargs):
        """Gym reset."""
        self.dict_state = self.env.reset(**kwargs)
        return self.observation(self.dict_state)

    def step(self, action):
        """Gym step."""
        self.dict_state, reward, done, info = self.env.step(action)
        return self.compute_observation(self.dict_state), reward, done, info

    @abstractmethod
    def compute_space(self) -> gym.Space:
        """Compute observation space."""
        pass

    @abstractmethod
    def compute_observation(self, observation: DictSpace) -> Any:
        """Transform according to observation space."""
        pass


class DiscreteFeatures(Features):
    """Discrete features.

    Discrete positions on the grid.
    """

    def compute_space(self) -> spaces.MultiDiscrete:
        """Compute observation space."""
        x_space: spaces.Discrete = self.dict_space.spaces["discrete_x"]
        y_space: spaces.Discrete = self.dict_space.spaces["discrete_y"]
        beep_space: spaces.Discrete = self.dict_space.spaces["beep"]
        return spaces.MultiDiscrete([x_space.n, y_space.n, beep_space])

    def compute_observation(self, observation: DictSpace) -> Any:
        """Transform according to observation space."""
        new_state = (
            observation["discrete_x"],
            observation["discrete_y"],
            observation["beep"],
        )
        return new_state


class DiscreteAngleFeatures(Features):
    """Discrete features with orientation.

    Discrete positions on the grid and discrete orientation.
    """

    def compute_space(self) -> spaces.MultiDiscrete:
        """Compute observation space."""
        x_space: spaces.Discrete = self.dict_space.spaces["discrete_x"]
        y_space: spaces.Discrete = self.dict_space.spaces["discrete_y"]
        theta_space: spaces.Discrete = self.dict_space.spaces["theta"]
        beep_space: spaces.Discrete = self.dict_space.spaces["beep"]
        return spaces.MultiDiscrete([x_space.n, y_space.n, theta_space.n, beep_space.n])

    def compute_observation(self, observation: DictSpace) -> Any:
        """Transform according to observation space."""
        new_state = (
            observation["discrete_x"],
            observation["discrete_y"],
            observation["theta"],
            observation["beep"],
        )
        return new_state


class ContinuousFeatures(Features):
    """Continuous features with orientation on the plane."""

    def compute_space(self) -> spaces.Box:
        """Compute observation space."""
        # Position on plane
        x_space: spaces.Box = self.dict_space.spaces["x"]
        y_space: spaces.Box = self.dict_space.spaces["y"]

        # Try with cos, sin, instead of angle
        cos_space = spaces.Box(-1, 1, shape=[1])
        sin_space = spaces.Box(-1, 1, shape=[1])

        # Try with derivative in components
        dx_space = spaces.Box(-float("inf"), float("inf"), shape=[1])
        dy_space = spaces.Box(-float("inf"), float("inf"), shape=[1])

        # Beep as real
        beep_space = spaces.Box(0.0, 1.0, shape=[1])

        # Join sapientino features
        merged = utils.combine_boxes(
            x_space,
            y_space,
            cos_space,
            sin_space,
            dx_space,
            dy_space,
            beep_space,
        )
        return merged

    def compute_observation(self, observation: DictSpace) -> Any:
        """Transform according to observation space."""
        # Deg to radians
        cos = np.cos(observation["angle"] / 180 * np.pi)
        sin = np.sin(observation["angle"] / 180 * np.pi)

        new_state = np.array(
            [
                observation["x"],
                observation["y"],
                cos,
                sin,
                observation["velocity"] * cos,
                observation["velocity"] * sin,
                observation["beep"]
            ],
            dtype=float,
        )
        return new_state
