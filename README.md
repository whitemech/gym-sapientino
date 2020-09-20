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
>>>>>>> Stashed changes


Gym Sapientino environment using Pygame.

## Links

- GitHub: [https://github.com/whitemech/gym-sapientino](https://github.com/whitemech/gym-sapientino)
- PyPI: [https://pypi.org/project/gym_sapientino/](https://pypi.org/project/gym_sapientino/)
- Documentation: [https://whitemech.github.io/gym-sapientino](https://whitemech.github.io/gym-sapientino)
- Changelog: [https://whitemech.github.io/gym-sapientino/history/](https://whitemech.github.io/gym-sapientino/history/)
- Issue Tracker:[https://github.com/whitemech/gym-sapientino/issues](https://github.com/whitemech/gym-sapientino/issues)
- Download: [https://pypi.org/project/gym_sapientino/#files](https://pypi.org/project/gym_sapientino/#files)


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

