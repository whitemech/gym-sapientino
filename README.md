<h1 align="center">
  <b>gym-sapientino</b>
</h1>

<p align="center">
  <a href="https://pypi.org/project/gym-sapientino">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/gym-sapientino">
  </a>
  <a href="https://pypi.org/project/gym-sapientino">
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/gym-sapientino" />
  </a>
  <a href="">
    <img alt="PyPI - Status" src="https://img.shields.io/pypi/status/gym-sapientino" />
  </a>
  <a href="">
    <img alt="PyPI - Implementation" src="https://img.shields.io/pypi/implementation/gym-sapientino">
  </a>
  <a href="">
    <img alt="PyPI - Wheel" src="https://img.shields.io/pypi/wheel/gym-sapientino">
  </a>
  <a href="https://github.com/whitemech/gym-sapientino/blob/master/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/whitemech/gym-sapientino">
  </a>
</p>
<p align="center">
  <a href="">
    <img alt="test" src="https://github.com/whitemech/gym-sapientino/workflows/test/badge.svg">
  </a>
  <a href="">
    <img alt="lint" src="https://github.com/whitemech/gym-sapientino/workflows/lint/badge.svg">
  </a>
  <a href="">
    <img alt="docs" src="https://github.com/whitemech/gym-sapientino/workflows/docs/badge.svg">
  </a>
  <a href="https://codecov.io/gh/whitemech/gym-sapientino">
    <img alt="codecov" src="https://codecov.io/gh/whitemech/gym-sapientino/branch/master/graph/badge.svg?token=FG3ATGP5P5">
  </a>
</p>
<p align="center">
  <a href="https://img.shields.io/badge/flake8-checked-blueviolet">
    <img alt="" src="https://img.shields.io/badge/flake8-checked-blueviolet">
  </a>
  <a href="https://img.shields.io/badge/mypy-checked-blue">
    <img alt="" src="https://img.shields.io/badge/mypy-checked-blue">
  </a>
  <a href="https://img.shields.io/badge/isort-checked-yellow">
    <img alt="" src="https://img.shields.io/badge/isort-checked-yellow">
  </a>
  <a href="https://img.shields.io/badge/code%20style-black-black">
    <img alt="black" src="https://img.shields.io/badge/code%20style-black-black" />
  </a>
  <a href="https://www.mkdocs.org/">
    <img alt="" src="https://img.shields.io/badge/docs-mkdocs-9cf">
  </a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/whitemech/gym-sapientino/master/docs/sapientino-homepage.gif" />
</p>

## Description

This is a configurable Gynmasium environment that implements one or more agents moving in a plane.
The software was originally inspired by a game for kids called
[_Sapientino_](https://it.wikipedia.org/wiki/Sapientino), but it has been extended in various ways.

Each agent moves on a 2D environment,
where each cell can be coloured or blank.
When a robot is on a coloured cell, it can
execute a _beep_ action, meaning it has visited the cell (this is meant to represent any interaction with the current location).

### Features

The environment is compliant with the `Gymnasium` APIs.

**Agents** There can be one or more agents in the same map. Each agent has its own action space.

**Actions** There are three default action spaces. The first allows the agent to move in the four cardinal directions. The second requires the agent to rotate by 90°, then move in discrete steps. The third instead allows the agent to accellerate and decelerate both in the angular and linear coordinates. This last modality does not implement a grid-world environment. For these actions, and how to implement your own, you can see `gym_sapientino/core/actions.py`.

**Observations** The `Sapientino` class has a dictionary observation space that contains all the current information. For personalizing the observation space you can subclass the `Features` class in `gym_sapientino/wrappers/observations.py`. We provide discrete and continuous features wrappers.

**Rewards** It is possible to specify per-step rewards, but for general reward functions, the user should wrap this environment.

**Map** The map can be easily configured using ASCII strings. For example, combining

    |P bB g |
    | bp G r|
    |G   pg |
    | rpG PB|
    |rP Bg b|

with `ContinuousCommand` generates:

![continuous control gif](/docs/continuous.gif)

For a more complete documentation of how to use the environment, and the available options see this [notebook](docs/quickstart.ipynb).


## Install

Install with `pip`:

    pip install gym_sapientino

Or, for a more updated version, install from github:

    git clone https://github.com/whitemech/gym-sapientino.git
    cd gym-sapientino
    pip install .

## Development

- Clone the repo:
```bash
git clone https://github.com/whitemech/gym-sapientino.git
cd gym-sapientino
```

- Install [Poetry](https://python-poetry.org/)
- Optionally select the Python version:
```bash
poetry env use python3.9
```

- Install with development dependencies:
```bash
poetry install
```

## Tests

To run tests: `tox`

Please look at the `tox.ini` file for the full list of supported commands and tests.


## License

gym-sapientino is released under the GNU General Public License v3.0 or later (GPLv3+).

Copyright 2019-2020 Marco Favorito, Roberto Cipollone, Luca Iocchi

## Authors

- [Luca Iocchi](https://sites.google.com/a/dis.uniroma1.it/iocchi/home)
- [Roberto Cipollone](https://cipollone.github.io/)
- [Marco Favorito](https://marcofavorito.github.io/)

## Credits

The code is largely inspired by [RLGames](https://github.com/iocchi/RLGames.git)

If you want to use this environment in your research, please consider
citing this conference paper:

```
@inproceedings{Giacomo2019FoundationsFR,
  title={Foundations for Restraining Bolts: Reinforcement Learning with LTLf/LDLf Restraining Specifications},
  author={Giuseppe De Giacomo and L. Iocchi and Marco Favorito and F. Patrizi},
  booktitle={ICAPS},
  year={2019}
}
```
