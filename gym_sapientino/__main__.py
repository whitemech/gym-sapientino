#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2019-2023 Marco Favorito, Roberto Cipollone, Luca Iocchi
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

"""Main entrypoint to play with Sapientino."""

import argparse

from gymnasium import Env

from gym_sapientino import Sapientino, play
from gym_sapientino.core.configurations import SapientinoAgentConfiguration
from gym_sapientino.play import FrameCapture
from gym_sapientino.sapientino_env import SapientinoConfiguration


def parse_arguments():
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser("sapientino")
    parser.add_argument(
        "--differential", action="store_true", help="Differential action space."
    )
    parser.add_argument(
        "--continuous", action="store_true", help="Continuous action space."
    )
    parser.add_argument("--record", action="store_true", help="Record the play.")
    parser.add_argument("--frames", action="store_true", help="Record single frames.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    c = SapientinoAgentConfiguration(args.differential, args.continuous)
    agent_configs = (c,)
    env: Env = Sapientino(SapientinoConfiguration(agent_configs))
    if args.frames:
        env = FrameCapture("frames", env)
    if args.record:
        raise NotImplementedError(
            "Recordings changed in new gymnasium. Not implemented yet"
        )

    play.play(env)
