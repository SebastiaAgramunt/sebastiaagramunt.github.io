---
title: UV python package and project manager
author: sebastia
date: 2024-11-24 6:15:00 +0800
categories: [Python]
tags: [computer science]
pin: true
toc: true
render_with_liquid: false
math: true
---

[uv](https://github.com/astral-sh/uv) is a more modern tool to build environments and manage python projects. It's built in Rust and claims to be extremely fast in resolving dependencies (10x-100x speedup compared to pip). 

## TLDR

General steps for creating a virtual environment

```bash
# install exec as
curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --verbose

# install a python version and create virtual environment named general in ~/.venvs
uv python install 3.12
uv venv ~/.venvs/general --python 3.12

# activate environment
source ~/.venvs/general/bin/activate

# install some packages
uv pip install numpy pandas

# see the list of packages and versions
uv pip list

#  # install from pyproject.com
# create env in current directory
uv venv .venv

# sync packages
uv sync
```

General steps to create a package (assuming we have python 3.13)

```bash
# create package structure
uv init --name=newpackage --build-backend=setuptools --package

# create first module (dumb file for example)
touch src/newpackage/modulea.py
echo "import numpy as np" >> src/newpackage/modulea.py

# add dependencies
uv add numpy pandas matplotlib

# see tree depencencies
uv tree

# lock/pin dependencies
uv lock

# build & publish
uv build && uv publish
```

## Install uv

It is possible to install `uv` system-wide using a bash installer for MacOS and Linux, I would recommend this if you decide `uv` is your definite tool to build projects. If that's the case, run on Linux or MacOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --verbose
```

that will install `uv` in `~/.cargo/bin/uv`. Checking on the file `install.sh` just the two binaries `uv` and `uvx` will be installed, not any other files. This is good, we just care about the binaries at this point.

Make sure you have `~/.cargo/bin/` directory in your `PATH` variable and execute `uv --help` to display the helper.

## Managing Python versions

`uv` is actually a great tool to substitute `pyenv` if you don't like the latter. With a simple command you can install and manage several python versions for your system:

```bash
uv python --help
```

Let's install `3.13` version

```bash
uv python install 3.13
```

And see that it has been installed in a directory that you can get from `uv python dir`, in my case (MacOS intel) `~/.local/share/uv/python`. 

Running `uv python list` will show all available versions (including the ones installed in `pyenv` tool) and the ones you can install.


## Managing Python environments

As mentioned before a python environment requires fist a python version, we can first fix the python version by running

```bash
uv python pin 3.13
```

And then we create the environment in `.venv` as

```bash
uv venv .venv
```

Check that the python version for the new environment is the correct one with `.venv/bin/python --version`.

Begin installing packages (no need to activate the environment if you are alrady in the directory where `.venv` lives)

```bash
uv pip install numpy pandas
```

and check which packages are in the environment

```bash
uv pip list
```

Alternatively you can use `uv pip freeze`. 

## Create a python project

`uv` can manage the creation of a basic project structure. First select your python version

```bash
uv python pin 3.13
```

Then run

```bash
uv init --help
```
among many options you will see options for the kind of project to be built:

```
--package                        Set up the project to be built as a Python package
--no-package                     Do not set up the project to be built as a Python package
--app                            Create a project for an application
--lib                            Create a project for a library
```

Let's create a package by running

```bash
uv init --name=newpackage --build-backend=setuptools --package
```

this will create a project with the structure (run `tree`):

```bash
.
├── README.md
├── pyproject.toml
└── src
    └── newpackage
        └── __init__.py
```

if you `cat pyproject.toml` you will get the definition of your project:

```
[project]
name = "newpackage"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = []

[project.scripts]
newpackage = "newpackage:main"

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
```

Now you can start adding dependencies

```bash
uv add numpy pandas matplotlib ruff
```

This will automatically create a virtual environment (in `.venv`) and install the package, also will modify the `pyproject.toml` with the package.Note, you can remember dependencies running `uv remove pkg`, for instance `uv remove pandas` to remove pandas.

A very convenient tool is the `tree`, we used `pipdeptree` package previously but the nice thing about `uv` is that this tooling comes by default. Just run

```bash
uv tree
```

that shows

```
newpackage v0.1.0
├── matplotlib v3.9.2
│   ├── contourpy v1.3.0
│   │   └── numpy v2.1.3
│   ├── cycler v0.12.1
│   ├── fonttools v4.54.1
│   ├── kiwisolver v1.4.7
│   ├── numpy v2.1.3
│   ├── packaging v24.1
│   ├── pillow v11.0.0
│   ├── pyparsing v3.2.0
│   └── python-dateutil v2.9.0.post0
│       └── six v1.16.0
├── numpy v2.1.3
└── pandas v2.2.3
    ├── numpy v2.1.3
    ├── python-dateutil v2.9.0.post0 (*)
    ├── pytz v2024.2
    └── tzdata v2024.2
```

Finally you can even pin the dependencies using

```bash
uv lock
```
that will generate a file `uv.lock` that contains the package version along with the hash and the specific wheel to download from pypi on every platform, similarly to what we have seen in pipvenv post in file `Pipfile.lock`. As mentioned several times in these series of posts, one may want to use a locked environment when running a service and not when defining a package.

In some ocasions you may want to build and publish the python package, for that uv has the commands `build` and `publish`. We won't get into the details of it in this post (we'll have a later post dedicated to it), just remember that you can handle this with `uv`.

## A more complete `pyproject.toml` and how to install in `uv` and `pip`

In this section we'll show a better `pyproject.toml` and how to install it using `pip` and `uv`. One of the good things about `uv` is that it seems to be completely compatible with `pip`, the default dependency manager. That makes it ideal for any project as the default user won't use `uv`.

```
[project]
name = "newpackage"
version = "0.1.0"
description = "My new package"
authors = [
    { name = "the author" }
]
license = { text = "MIT license" }
readme = "README.md"

requires-python = ">=3.9,<3.13"
dependencies = [
    "matplotlib>=3.9",
    "numpy>=1.26,<2.0.0",
    "pandas>=2.2",
    "scipy>=1.13",
    "tifffile>=2024.8",
    "pip>=24.3.1",
]

[project.optional-dependencies]
dev = [
    "coverage>=7.6",
    "pytest>=8.3",
    "ruff>=0.7",
]

# Ruff is a great tool for linting
[tool.ruff]
line-length = 99

[tool.ruff.lint]
select = [
    # Pyflakes
    "F",
    # Pycodestyle & Warnings
    "E",
    "W",
    # isort for unsorted imports
    "I001"
]

[tool.ruff.format]
quote-style = "single"
indent-style = "space"
docstring-code-format = true
docstring-code-line-length = 20

# build system, use setuptools, the default for Python
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

# python CLI, structure scripts/my_script.py funcion main
# once installed the environment, activate it and run `my_cli` on terminal
# to run the CLI
[project.scripts]
my_cli = "scripts.my_script:main"

## In case you run a private pypi repository, uncomment and change URL
# [[tool.uv.index]]
# name="pypi-internal-server"
# url = "http://pypi-server.mydomain.com:8081/repository/"
# priority = "supplemental"
```

To install, create the enviornment and sync

```bash
uv venv .venv
uv sync

# to install with the optional dependencies (dev in our case)
uv sync --all-extras
```

Equivalently in pip you can do

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install . --extra-index-url http://pypi-server.mydomain.com:8081/repository/

# to install with optional dependencies
.venv/bin/python -m pip install .[dev] --extra-index-url http://pypi-server.mydomain.com:8081/repository/
```

Now, there's a lot here, let me explain, the first part just specifies the python versions and the dependencies. Then we have the tool `ruff`, more on that later, the build system (`setuptools` as the default tool in python) and finally a CLI and an internal pypi repository. 

Let me begin by `ruff` tells you what lines of your code are not properly formatted (linting) and also formats them (changes the format of the code, a formater). Run it with `uv run ruff check .`. Imagine you have a file with the following content in your package:

```python
import os
import sys  # Unused import

def example_function(x, y):
    return x + y

def unused_function(a, b):  # Unused function
    return a * b

print(example_function(1, 2))

def _make_ssl_transport(
    rawsock, protocol, sslcontext, waiter=None,
    *, server_side=False, server_hostname=None,
    extra=None, server=None,
    ssl_handshake_timeout=None,
    call_connection_made=True):
    '''Make an SSL transport.'''
    if waiter is None:
      pass

    if extra is None:
      extra = {}

```

Ruff (with the above configuration) here will complain in several places. In the first line it will raise an error because the imports are not sorted (using isort rule). In the first and second lines it will also raise error, this time `F401` because the packages sys and os are imported but never used. The unused function will raise `F841` and finally (not the case in this example) if any line would exceed 99 characters it would raise `E501`. Ruff is good to keep you code clean, check all the rules running ` uv run ruff rule --all` and all the linters with `uv run ruff linter`.  Once the errors are identified you can fix them as suggested by `ruff` with

```bash
uv run ruff check . --fix
```

Finally format the code with

```bash
uv run ruff format
```

In the documentation ruff developers say that " the formatter is designed as a drop-in replacement for Black". It indeed formats the function for us to a much nicer one!. Check the [rules](https://docs.astral.sh/ruff/rules/) and the [formater](https://docs.astral.sh/ruff/formatter/) in the official documentation.

Finally we have added a private repository in the `pyproject.toml`. The private repository is a URL where we can host wheels, the software artifacts containing a package (most of the times is basically a zipped repository). In some organizations we publish packages in an internal repository but by default python tries to find all packages in [pypi](https://pypi.org/). Adding the final lines with the appropiate URL will tell `uv` that it may have to look at that repository too. We have defined this for `uv` only in the `pyproject.toml`, actually it is not possible to define it for `pip`. In that case we simply add the extra index at the time of installation through `--extra-index-url http://pypi-server.mydomain.com:8081/repository/`. An alternative for `pip` is to add a `pip.conf` file (see more details [here](https://pip.pypa.io/en/stable/topics/configuration/)).

This section ended up being a bit long but I wanted to show a good `pyproject.toml` file with most of the things you need on a python package project. Hope it is helpful!. I will likely write two other posts about `ruff` and how to setup your own `pypi` server securely, in a much more detailed way.

## Speed benchmark 

uv promisses large speedups in resolving dependencies on their [weppage](https://docs.astral.sh/uv/), about 10x to 100x compared to pip. In this section we will test how fast each engine is capable of resonving basic dependencies. We will compare `uv` with other popular tools like `pip`, `conda`, `mabmba` and `poetry`. I will use my 2019 macbook pro AMD with 16 GB of memory. Also to compare apples to apples I will start building from scratch each environment without caching packages, i.e. downloading all the packages each time. We will ask the dependency manager to install the following:

```
pandas scikit-learn flask fastapi matplotlib requests pytest boto3 pyyaml cryptography jupyterlab seaborn pillow sqlalchemy
```

on a python 3.11 version.

Let's begin with the tool presented in this post, `uv`.


### uv

```bash
uv python install 3.11
uv python pin 3.11
rm -rf .venv
uv venv .venv
time uv pip install pandas scikit-learn flask fastapi matplotlib requests pytest boto3 pyyaml cryptography jupyterlab seaborn pillow sqlalchemy --no-cache-dir
```

with result:

```
 3.08s user 8.33s system 148% cpu 7.668 total
```


### pip

```bash
pyenv install 3.11.2
pyenv shell 3.11.2

rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
time pip install pandas scikit-learn flask fastapi matplotlib requests pytest boto3 pyyaml cryptography jupyterlab seaborn pillow sqlalchemy --no-cache-dir
```

with result:

```
32.32s user 7.25s system 66% cpu 59.662 total
```

### Conda

Run

```bash
pyenv shell miniconda3-latest
conda remove --name myenv --all -y
```

create the file `environment.yml`, this way we can time better the process
```
name: myenv
dependencies:
  - python=3.11
  - pandas
  - scikit-learn
  - flask
  - fastapi
  - matplotlib
  - requests
  - pytest
  - boto3
  - pyyaml
  - cryptography
  - jupyterlab
  - seaborn
  - pillow
  - sqlalchemy
```

```bash
conda clean --all -y
time conda env create -f environment.yml
```

getting a time of 43 seconds.

### Mamba

Run

```bash
pyenv shell mambaforge
conda remove --name myenv --all -y
```

create same file `environment.yml` as the `conda` section. 

```bash
conda clean --all -y
time conda env create -f environment.yml
```

geting a time of 52 seconds.

### Poetry

```bash
poetry config virtualenvs.in-project true
pyrenv shell 3.11.9
poetry env use 3.11.9
```

The `pyproject.toml` that has to be placed in the project directory

```
[tool.poetry]
name = "speed-test"
version = "0.1.0"
description = ""
authors = ["Your Name <you@example.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "3.11.9"
pandas = "*"
scikit-learn = "*"
flask = "*"
fastapi = "*"
matplotlib = "*"
requests = "*"
pytest = "*"
boto3 = "*"
pyyaml = "*"
cryptography = "*"
jupyterlab = "*"
seaborn = "*"
pillow = "*"
sqlalchemy = "*"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

Then run


```bash
time poetry install --no-cache
```

Which takes 25.05s user 14.10s system 100% cpu 38.877 total.

### Benchmark conclusions

Summing up, `uv` seems to be the fastest by far, around ~8x compared to `pip`. Very promissing!

| tool   | time to resolve depencencies(seconds) |
|--------|---------------------------------------|
| uv     | 07.66                                 |
| pip    | 59.62                                 |
| conda  | 43                                    |
| mamba  | 52                                    |
| poetry | 39.87                                 |


## Conclusions

uv is not only super fast in resolving depencencies, it manages your python versions and by default creates the environment in the local directory. With `uv` you don't need anything else, no `pyenv` for managing your python versions, or no `poetry` to build and publish your wheels. Even creating a new project boilerplate is super easy!. I have been reading out there and seems that the only drawback is that the dependency management is a bit less strict compared to `poety`. To me this is fine, this tool simply works. Another big plus of `uv` is that it is perfectly compatible with `pip`, this is, the `pyproject.toml` defined for `uv` can be installed using `pip` seamlessly as it follows the standard of [PEP-621](https://peps.python.org/pep-0621/) not forcing the users of your package to use `uv` but the most general tool `pip` if they want to.