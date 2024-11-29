---
title: Python virtual environments with Poetry
author: sebastia
date: 2024-11-01 20:36:00 +0800
categories: [Python]
tags: [computer science]
pin: true
toc: true
render_with_liquid: false
math: true
---


[Python Poetry](https://python-poetry.org) is a tool for package dependency management and packaging. It has become very popular among developers.

## TLDR

To create a virtual environment, first install poetry in your project

```bash
pyenv install 3.12.2 -f
pyenv shell 3.12.2

# create a venv with poetry exec
python -m venv .venv_poetry
.venv_poetry/bin/python -m pip install -U pip setuptools
.venv_poetry/bin/python -m pip install poetry

# config poetry to create the venv in the current directory
.venv_poetry/bin/poetry config virtualenvs.prefer-active-python true
.venv_poetry/bin/poetry virtualenvs.in-project true

```

Add place the `pyproject.toml` file in the directory:

```
[tool.poetry]
package-mode = false

[tool.poetry.dependencies]
python = ">=3.10,<3.13"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

```bash
# add dependencies
.venv_poetry/bin/poetry add numpy
.venv_poetry/bin/poetry add pandas
.venv_poetry/bin/poetry add matplotlib

# add development dependencies
poetry add pytest --group dev

# install (create environment) with all dependencies
.venv_poetry/bin/poetry install

# install (create environment) without dev dependencies
.venv_poetry/bin/poetry install --no-dev

# execute python in the new environment
.venv/bin/python --version

# lock the dependencies
.venv_poetry/bin/poetry lock
```
To create a new repository read the specific section in this post.

## Install Poetry

Similar to `virtualenv`, `poetry` is a package of python so you can install via `pip install`. The most common is to install it in the default interpreter, the global in `pyenv` or the default in the system `/usr/local/bin/python3` (in MacOS).

Define a global python using pyenv, as an example we will use `3.12`.

```bash
GLOBAL_PYTHON=3.12
pyenv global ${GLOBAL_PYTHON}
```

Install poetry on it (see [official documentation](https://python-poetry.org/docs/#installation)) and check the help (just to see that it works)

```bash
python -m pip install -U pip setuptools
python -m pip install poetry
poetry --help
```

Then you will have `poetry` available anytime you are in the pyenv global interpreter on your shell.

Instead of the described method I prefer to always create a new virtual environment in the project to install `poetry`. The reason is that I may use poetry for certain projects, not for everything so to me it is better to install it per project. To do this, navigate to your python project, install the python version:

```bash
pyenv install 3.12.2 -f
pyenv shell 3.12.2
```

and create the environment in e.g. `.venv_poetry` then install `poetry`:

```bash
python -m venv .venv_poetry
.venv_poetry/bin/python -m pip install -U pip setuptools
.venv_poetry/bin/python -m pip install poetry
```

Now all you need to do to use `poetry` is to execute the binary in this environment: simply call `.venv_poetry/bin/poetry`. To see that this works you can show the poetry config:

```bash
.venv_poetry/bin/poetry config --list
```

To remove `poetry` simply remove the virutal environment `.venv_poetry`.

## Create a virtual environment 

Before creating a new environment we need to select the python version. To me, the best way is to still use `pyenv` to install and manage python versions and select the shell version using `pyenv shell` command. Then you can tell `poetry` to use the current activated python binary to create the environment. Let's do this, as before we create a virtual environment encapsulating `poetry` as:

```bash
pyenv install 3.12.2 -f
pyenv shell 3.12.2

python -m venv .venv_poetry
.venv_poetry/bin/python -m pip install -U pip setuptools
.venv_poetry/bin/python -m pip install poetry
```

Now in the poetry config we need to modify a couple of parameters:

```bash
.venv_poetry/bin/poetry config virtualenvs.prefer-active-python true
.venv_poetry/bin/poetry virtualenvs.in-project true
```

This will enable using the current activated python and also make it live in the current directory. Create a `pyproject.toml` with the following content

```
[tool.poetry]
package-mode = false

[tool.poetry.dependencies]
python = "^3.11"
numpy = "^2.1.3"
pandas = "^2.2.3"

[tool.poetry.group.dev.dependencies]
pytest = "^8.3.3"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

see that I set `package-mode=false` indicating that I don't want this configuration to be a package, just want to specify the dependencies for a virtual environment. You can tell that this `pyproject.toml` is built by poetry, first because the build-system is poetry but also every other section is `tool.poetry`. Don't worry if you are working with python packages (remember this example is just a toml to install dependencies in a virtual environment) you can always pip install the project, just need to compile the wheel with poetry and pip install it.

Let's continue with installing the virtual environment specified. Say you want to install the new environment using python 3.11, install this version first using `pyenv`

```bash
pyenv install 3.11 -f
pyenv shell 3.11
```

Then install the environment as

```bash
.venv_poetry/bin/poetry install
```

See that this will the usual `.venv` directory where all the environment is saved. Check that the python version used is `3.11` and not `3.12` by running `.venv/bin/python --version`. Now you activate the environment and use it with `source .venv/bin/activate`.

## Creating packages using poetry

It is super simple to create a new package, simply create a new directory, install poetry and run `poetry init`, see the code:

```bash
mkdir my_package && cd my_package
python -m venv .venv_poetry
.venv_poetry/bin/python -m pip install -U pip setuptools
.venv_poetry/bin/python -m pip install poetry
.venv_poetry/bin/poetry config virtualenvs.prefer-active-python true
.venv_poetry/bin/poetry virtualenvs.in-project true
```

then initialize the package with

```bash
.venv_poetry/bin/poetry init
```

and follow the steps in the command line. Poetry will ask you for the dependencies, the name of the project, the license etc... Finally will create the `pyproject.toml`. Create the `README.md` yourself with `touch README.md` otherwise poetry will raise an error before installing. Now add the source code:

```bash
mkdir my_package
touch my_package/__init__.py
touch my_package/modulea.py
```
In `modulea.py` place something like (as an example):

```python
import numpy as np

def numpy_max(a, b):
    return np.max(a, b)
```

now install the project

```bash
.venv_poetry/bin/poetry install
```

and check that you can import the package from your new environment

```bash
.venv/bin/python -c "import my_package"
```

And the only thing you are left to do is to code your new flashy package.


## Creating and publishing wheels

Python wheels are the artifacts that you download from [pypi](https://pypi.org/) repository to install packages. A wheel is basically a zipped file that containes the source code (python files) and compiled code that is platform specific (if any). Here I will show you how to create and publish a package using poetry

To build the wheel of your project simply run

```bash
.venv_poetry/bin/poetry build 
```

you will see a new diretory in your project called `dist` where two new files will be placed with extension `.whl` and `.tar.gz`. The latter is simply the zipped project and is platform independent, it is useful to unzip and compile for different platforms. We will go into the detail of this in another post where we will build modules using compiled C++ and shared libraries. Here we deal with pure python code so our project will be platform independend as long as the dependencies are available for every platform (e.g. numpy contains compiled C++ code and is available on many platforms and architectures).

The next step is to publish the wheel, for that you need to tell `poetry` where it should be published, include the details in the `pyproject.toml`, for instance:

```
[tool.poetry.repositories]
custom = { url = "https://your.custom.repo.url" }
my_pypi = { url = "https://pypi.org/project/my_package_name/" }
```

where obviously you need to change the URLs and names to your own. Then you can run

```bash
poetry publish --repository custom
```

to publish to your custom repo. Also to add a new repository from which you want to download packages add the following to your `pyproject.toml`:

```
[[tool.poetry.source]]
name = "custom_repo"
url = "https://your.custom.repo.url"
priority = "supplemental"
```

the fact that it is supplemental indicates `poetry` that the primary where to look files from is pypi. It will look for wheels there and then if it can't find them it will default to your custom repository.
