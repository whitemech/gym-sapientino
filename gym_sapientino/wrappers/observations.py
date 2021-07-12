"""Definitions of observations spaces.

SapientinoDictSpace is the most generic class, which returns all
the information in form of a dictionary. Features classes define an observation
space and extract information accordingly. 
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence, Type

import gym
import numpy as np
from gym import spaces

from gym_sapientino import SapientinoDictSpace, utils

DictSpace = Dict[str, Any]


class Features(ABC):
    """Base class for all observation spaces.

    By subclassing from this class, we can define any observation space
    computed over SapientinoDictSpace.
    The member "dict_state" stores last observation received from
    SapientinoDictSpace. UseFeatures is a wrapper useful to apply features
    to the environment.
    """

    def __init__(self, dict_space: spaces.Dict):
        """Initialize.

        :param dict_space: the input observation space.
        """
        self.dict_space = dict_space
        self.observation_space = self.compute_space()

    @abstractmethod
    def compute_space(self) -> gym.Space:
        """Compute observation space."""
        pass

    @abstractmethod
    def compute_observation(self, observation: DictSpace) -> Any:
        """Transform according to observation space."""
        pass


class UseFeatures(gym.ObservationWrapper):
    """Choose a set of features for each robot in the environment.

    The feature classes of this module define the observation space for one
    agent only. Use this class to apply one feature space for each robot.
    """

    def __init__(self, env: SapientinoDictSpace, features: Sequence[Type[Features]]):
        """Initialize.

        :param features: one feature for each robot.
        """
        # Check
        if len(features) != env.configuration.nb_robots:
            raise ValueError(
                f"Wrong number of features: expected {env.configuration.nb_robots}")

        # Store
        super().__init__(env)
        self.features = [
            features[i](env.observation_space[i])
            for i in range(len(features))
        ]

        # Obs space
        self.observation_space = spaces.Tuple([
            self.features[i].compute_space()
            for i in range(len(features))
        ])

    def observation(self, observation):
        """Compute an observation with features."""
        assert len(observation) == len(self.features)
        return [
            self.features[i].compute_observation(observation[i])
            for i in range(len(observation))
        ]


class DiscreteFeatures(Features):
    """Discrete features.

    Discrete positions on the grid.
    """

    def compute_space(self) -> spaces.MultiDiscrete:
        """Compute observation space."""
        x_space: spaces.Discrete = self.dict_space.spaces["discrete_x"]
        y_space: spaces.Discrete = self.dict_space.spaces["discrete_y"]
        beep_space: spaces.Discrete = self.dict_space.spaces["beep"]
        return spaces.MultiDiscrete([x_space.n, y_space.n, beep_space.n])

    def compute_observation(self, observation: DictSpace) -> Any:
        """Transform according to observation space."""
        new_state = np.array([
            observation["discrete_x"],
            observation["discrete_y"],
            observation["beep"],
        ], dtype=int)
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
        new_state = np.array([
            observation["discrete_x"],
            observation["discrete_y"],
            observation["theta"],
            observation["beep"],
        ], dtype=int)
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
        cos = np.reshape(np.cos(observation["angle"] / 180 * np.pi), [1])
        sin = np.reshape(np.sin(observation["angle"] / 180 * np.pi), [1])

        new_state = np.concatenate(
            [
                observation["x"],
                observation["y"],
                cos,
                sin,
                observation["velocity"] * cos,
                observation["velocity"] * sin,
                np.reshape(observation["beep"], [1]),
            ],
            dtype=np.float32,
        )
        return new_state
