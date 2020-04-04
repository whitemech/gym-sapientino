#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gym_sapientino.sapientino_env import SapientinoConfiguration
from gym_sapientino.wrappers.dict_space import SapientinoDictSpace

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser("sapientino")
    parser.add_argument("--differential", action="store_true", help="Differential action space.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    env = SapientinoDictSpace(SapientinoConfiguration(differential=args.differential))
    env.play()
