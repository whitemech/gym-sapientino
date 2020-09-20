#!/usr/bin/env python3
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

"""Main entrypoint to play with Sapientino."""

import argparse

from gym_sapientino.sapientino_env import SapientinoConfiguration
from gym_sapientino.wrappers.dict_space import SapientinoDictSpace


def parse_arguments():
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser("sapientino")
    parser.add_argument(
        "--differential", action="store_true", help="Differential action space."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    env = SapientinoDictSpace(SapientinoConfiguration(differential=args.differential))
    env.play()
