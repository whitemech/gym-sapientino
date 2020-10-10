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

OpenAI Gym Sapientino environment using Pygame.

<p align="center">
  <img src="https://raw.githubusercontent.com/whitemech/gym-sapientino/master/docs/sapientino-homepage.gif" />
</p>

## Description

The environment is inspired by a game for kids called 
[_Sapientino_](https://it.wikipedia.org/wiki/Sapientino).
 
A robot moves on a gridworld-like environment, 
where each cell can be coloured. 
When a robot is on a coloured cell, it can 
run a _beep_, meaning it has visited the cell.

The environment is compliant with the 
[OpenAI Gym](https://github.com/openai/gym/) APIs.
The idea is that the designer of the experiment
should implement the actual reward by wrapping the environment. 

## Dependencies

The environment is implemented using Pygame.

On Ubuntu, you need the following libraries:
```
sudo apt-get install python3-dev \
    libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev \
    libsdl1.2-dev  libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev
```

On MacOS (not tested):
```
brew install sdl sdl_ttf sdl_image sdl_mixer portmidi  # brew or use equivalent means
conda install -c https://conda.binstar.org/quasiben pygame  # using Anaconda
```

## Install

Install with `pip`:

    pip install gym_sapientino
    
Or, install from source:

    git clone https://github.com/whitemech/gym-sapientino.git
    cd gym-sapientino
    pip install .

## Development

- clone the repo:
```bash
git clone https://github.com/whitemech/gym-sapientino.git
cd gym-sapientino
```
    
- Create/activate the virtual environment:
```bash
poetry shell --python=python3.7
```

- Install development dependencies:
```bash
poetry install
```
    
## Tests

To run tests: `tox`

To run only the code tests: `tox -e py3.7`

To run only the linters: 
- `tox -e flake8`
- `tox -e mypy`
- `tox -e black-check`
- `tox -e isort-check`

Please look at the `tox.ini` file for the full list of supported commands. 

## Docs

To build the docs: `mkdocs build`

To view documentation in a browser: `mkdocs serve`
and then go to [http://localhost:8000](http://localhost:8000)

## License

gym-sapientino is released under the GNU General Public License v3.0 or later (GPLv3+).

Copyright 2019-2020 Marco Favorito, Luca Iocchi

## Authors

- [Luca Iocchi](https://sites.google.com/a/dis.uniroma1.it/iocchi/home)
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
