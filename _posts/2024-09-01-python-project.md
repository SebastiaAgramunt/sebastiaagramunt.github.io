---
title: Setup a python project
author: sebastia
date: 2024-09-01 18:05:00 +0800
categories: [Python]
tags: [computer science]
pin: true
toc: true
render_with_liquid: false
math: true
---

The easiest way to distribute your code is to packagify it, this is create a package that can be `pip` installed, in this post I am going to show how to do that.

## Project structure

First you need to define a project structure, our project is going to be pure python, this is, no compiled code, just python and compiled dependencies like `numpy`. Create the following files in your new project directory, we will call the new package `mypackage`.

```bash
.
├── README.md
├── pyproject.toml
├── src
│   └── mypackage
│       ├── __init__.py
│       ├── modulea.py
│       └── moduleb.py
└── tests
    ├── __init__.py
    ├── test_modulea.py
    └── test_moduleb.py
```

The `pyproject.toml` is where all the dependencies will be defined, we will get over it in the next section. For now, leave the `__init__.py` files empty and a custom function to `modulea.py`

```python
def add(a, b):
    return a + b
```

and another fucntion example to `moduleb.py`

```python
def subtract(a, b):
    return a - b
```
We'll get to the testing part in a while but basically that's it, this is a pure python package.

## Importing without installing

Python pacakges and modules work importing files from directories. As an example create a new virtual environment in the root directory of the project:

```bash
PYTHON_VERSION=3.12.4
pyenv shell ${PYTHON_VERSION}
python -m venv .venv
```

Activate the environment and import addition from modulea:

```bash
source .venv/bin/activate
python -c "from src.mypackage.modulea import add; print(add(3,4))"
```

This works, see that `src.mypackage.modulea` is the path in directories to the file `modulea.py`. Now go one directory up and try the same

```bash
cd ..
python -c "from src.mypackage.modulea import add; print(add(3,4))"
```
the error you get is

```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'src'
```
that means you can't import, you are no longer in the project directory. How can we import the package from anywhere having our virtual environment activated?.It's necessary to install the pacakge into the virtual evnvironment, we will do this in the following sections.

## Defining a project with pyproject.toml

The "new" (since 2016) standard to specify dependencies in a package is the `pyproject.toml` file, it was introduced in [PEP 518](https://peps.python.org/pep-0518/), here is an example

```toml
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mypackage"
version = "0.1.0"
description = "A simple example project"
authors = [
    { name = "Your Name", email = "you@example.com" }
]
license = { text = "MIT" }
readme = "README.md"
keywords = ["example", "setuptools", "pyproject"]
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
requires-python = ">=3.9,<3.13"
dependencies = [
    "requests>=2.20",
    "numpy>=1.18",
    "pandas>=2.0",
    "pipdeptree>2.23"
]

[project.urls]
"Homepage" = "https://example.com"
"Repository" = "https://github.com/example/mypackage"

[tool.setuptools]
packages = ["mypackage"]
package-dir = {"" = "src"}

[project.optional-dependencies]
dev = [
    "tox>=4.2",
    "pytest>=8.3",
    "pytest-cov>=5",
    "ruff>=0.7"
]

```

Install the package in a new environment

```bash
# install python version create environment and activate it.
PYTHON_VERSION=3.12.4
pyenv shell ${PYTHON_VERSION}
python -m venv .venv
source .venv/bin/activate

# upgrade pip and install package
pip install --upgrade pip
python -m pip install .
```

That will install all dependencies but not the ones listed in optional. To install all depenencies including the optional (in this example the `dev`) just run

```bash
pip install .[dev]

# if you use zsh like me, do
pip install '.[dev]'

# or if you want to install the package in editable mode
python -m pip install -e '.[dev]'
```

Test that the package is installed

```
python -c "from mypackage.modulea import add; print(add(3,4))"
```

## Test your package

The package `unittest` comes by default with python, it's convenient and easy, under the directory `tests` add two files to make unit tests of your code

In `test_modulea.py`, copy

```python
import unittest
from mypackage.modulea import add

class TestModule1(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
        self.assertEqual(add(-1, 1), 0)
        self.assertEqual(add(0, 0), 0)

if __name__ == '__main__':
    unittest.main()
```

and in `test_moduleb.py`

```python
import unittest
from mypackage.moduleb import subtract

class TestModule2(unittest.TestCase):
    def test_subtract(self):
        self.assertEqual(subtract(2, 1), 1)
        self.assertEqual(subtract(1, 1), 0)
        self.assertEqual(subtract(0, 1), -1)

if __name__ == '__main__':
    unittest.main()
```

Since we have included a main function you can execute the tests by simply calling the scripts as

```bash
.venv/bin/python tests/test_modulea.py
.venv/bin/python tests/test_moduleb.py
```

but you can also run them all with pytest, just do

```bash
.venv/bin/python -m pytest
```

from the root directory. You can also run

```bash
.venv/bin/python -m unittest discover -s tests -p "test_*.py"
```

to run the same tests. This command calls the `unittest` command line, tells it to discover tests that are under the directory `tests` and the test file names have the pattern `test_*.py`. Clearly `pytest` is more simple, among other features Pytest automatically discovers test files and functions without requiring special naming conventions like "test_*.py". But `unittest` comes by default with the newer python versions. I would use `pytest` in every modern project always placing it under the `[dev]` depencencies.

Adding coverage is also important, it helps you track the if all your functionality is tested. We can get a coverage report along with our test run for `pytest` using the package `pytest-cov`. Run the tests and the coverage with

```bash
.venv/bin/python -m pytest --cov=src/mypackage tests/
```

The output should be something like this:

```bash
platform darwin -- Python 3.11.2, pytest-8.3.3, pluggy-1.5.0
rootdir: /Users/sebas/tmp/mypackage
configfile: pyproject.toml
plugins: cov-5.0.0
collected 2 items                                                                                                                                                                                                                                                                                                                                   

tests/test_modulea.py .                                                                                                                                                                                                                                                                                                                       [ 50%]
tests/test_moduleb.py .                                                                                                                                                                                                                                                                                                                       [100%]

---------- coverage: platform darwin, python 3.11.2-final-0 ----------
Name                        Stmts   Miss  Cover
-----------------------------------------------
src/mypackage/__init__.py       0      0   100%
src/mypackage/modulea.py        2      0   100%
src/mypackage/moduleb.py        2      0   100%
-----------------------------------------------
TOTAL                           4      0   100%
```

All tests pass and the coverage is 100%, meaning that all our functionality has at least one test.

## Testing with TOX

This section may be controversial, TOX is a tool that allows you to test in different python versions locally. This means, testing in your operating system, not different architectures and operating systems. This is fine as long as you are aware of it. In most cases you won't need to compile code. Let's see with an example how it works, create a file `tox.ini` in the root of the project with the content

```toml
[tox]
envlist = py39, py310, py311, py312
isolated_build = True
skip_missing_interpreters = False

[testenv]
deps =
    pytest>=7.0
    pytest-cov>=2.12
commands =
    pytest --cov=mypackage --cov-report=term-missing {posargs}
```

Now run `tox` in your environment (has been installed in the `pyproject.toml`) and lo and behold it creates new environments for each python version and runs the tests with coverage. In another post we will see how to set up automatic testing for different platforms on github using github actions.

## Linting with Ruff

Code should be organised and structured in a consensual way. That is why some compaines publish their code style (e.g. [Python's style at Google](https://google.github.io/styleguide/pyguide.html) ). To achieve this in an automated way, for that linters are useful. The Python Enhancement Proposal 8 or [PEP8](https://peps.python.org/pep-0008/) proposed a series of best practices on writing code that can be reviewed in [flake8rules.com](https://www.flake8rules.com/) with each rule consisting of a letter and a number,

* Error Codes: Starting with E (e.g., E123, E501) to represent style issues.
* Warning Codes: Starting with W (e.g., W503).
* Complexity Checks: Codes starting with C (e.g., C901), usually related to cyclomatic complexity.
* Other Code Categories: Codes like F, N, and D for various kinds of issues such as undefined names (F), naming conventions (N), or docstring style (D).


A popular linter in python is [Ruff](https://github.com/astral-sh/ruff) and this is the one we will use for the project but other tools are [flake8](https://github.com/PyCQA/flake8) or [black](https://github.com/psf/black). I already included `ruff` in our environment `pyproject.toml` so you don't need to install it, just add the configuration in `pyproject.toml` appending the following

```toml
[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = ["E", "W", "F"]

[tool.ruff.format]
docstring-code-format = true
docstring-code-line-length = 72
```

Here when we run `ruff check` on the command line the program will find errors (E), warnings (W) and undifined names (F), also will complain for code lines larger than 88 (not PEP8 but pretty standard line lenght) and will also check for the docstrings format and length. This is a basic configuration but you can go very complex from here. Check the [Ruff documentation](https://docs.astral.sh/ruff/) for more information.

## Recap

Here we just showed the basics to setup a python project in pure python, in following posts we will learn how to automate continuous integraiont / continous development CI/CD, package building and docker containers for development and testing.
