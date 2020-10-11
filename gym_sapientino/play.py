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

"""Helper functions to record frames and play with the game interactively."""

import shutil
from pathlib import Path

import gym
import pygame
from PIL import Image


class FrameCapture(gym.Wrapper):
    """Capture frames from the game."""

    def __init__(self, dest_dir: str, *args, **kwargs):
        """
        Initialize the callback.

        :param dest_dir: the destination directory.
        """
        super().__init__(*args, **kwargs)
        self.dest_dir = Path(dest_dir)
        if self.dest_dir.exists():
            shutil.rmtree(self.dest_dir)
        self.dest_dir.mkdir()

        self.current_episode_step = 0
        self.current_episode = 0

    def save_frame(self) -> None:
        """
        Save the frame.

        :return: None
        """
        rgb_array = self.render(mode="rgb_array")
        img = Image.fromarray(rgb_array)
        step = self.current_episode_step
        episode = self.current_episode
        filename = "{:06}-{:010}.jpeg".format(episode, step)
        img.save(self.dest_dir / filename)

    def step(self, action):
        """Do a step and record a frame."""
        self.current_episode_step += 1
        result = super().step(action)
        self.save_frame()
        return result

    def reset(self, **kwargs):
        """Reset the environment and record a frame."""
        self.current_episode_step = 0
        self.current_episode += 1
        result = super().reset()
        self.save_frame()
        return result


def play(env: gym.Env) -> None:
    """Play interactively with the environment."""
    print("Press 'Q' to quit.")
    assert env.unwrapped.configuration.nb_robots == 1, "Can only play with one robot."
    env.reset()
    env.render()
    quitted = False
    while not quitted:
        event = pygame.event.wait()
        cmd = 5
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                quitted = True
            elif event.key == pygame.K_LEFT:
                cmd = 0
            elif event.key == pygame.K_UP:
                cmd = 1
            elif event.key == pygame.K_RIGHT:
                cmd = 2
            elif event.key == pygame.K_DOWN:
                cmd = 3
            elif event.key == pygame.K_SPACE:
                cmd = 4

            env.step([cmd])
            env.render()
    env.render()
    env.close()
