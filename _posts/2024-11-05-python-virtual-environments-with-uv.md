---
title: UV python package and project manager
author: sebastia
date: 2024-11-05 6:15:00 +0800
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

# install a python version and create virtual environment
uv python install 3.13
uv python pin 3.13
uv venv .venv

# install some packages
uv pip install numpy pandas

# see the list of packages and versions
uv pip list
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
that will generate a file `uv.lock` that contains the package version along with the hash and the specific wheel to download from pypi on every platform, similarly to what we have seen in [pipvenv](./2024-08-31-python-virtual-environments-with-pipenv.md) in file `Pipfile.lock`. As mentioned several times in these series of posts, one may want to use a locked environment when running a service and not when defining a package.

In some ocasions you may want to build and publish the python package, for that uv has the commands `build` and `publish`. We won't get into the details of it in this post (we'll have a later post dedicated to it), just remember that you can handle this with `uv`.

## Speed benchmark 

In this section we will test how fast each engine is capable of resonving basic dependencies and time it!. In an empty directory execute:


### uv

```bash
uv python install 3.11
uv python pin 3.11
uv venv .venv
time uv pip install pandas scikit-learn flask fastapi matplotlib requests pytest boto3 pyyaml cryptography jupyterlab seaborn pillow sqlalchemy --no-cache-dir
```

with result:

```
 2.90s user 8.10s system 128% cpu 8.555 total
```


### pip

```bash
pyenv install 3.11.2
pyenv shell 3.11.2

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
time pip install pandas scikit-learn flask fastapi matplotlib requests pytest boto3 pyyaml cryptography jupyterlab seaborn pillow sqlalchemy --no-cache-dir
```

with result:

```
32.84s user 6.27s system 77% cpu 50.639 total
```

### conda

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
conda clean --all
time conda env create -f environment.yml
```

getting a time of 45 seconds.

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

Summing up, `uv` seems to be the fastest by far. Very promissing

| tool   | time to resolve depencencies(seconds) |
|--------|---------------------------------------|
| uv     | 4.479                                 |
| pip    | 50.63                                 |
| conda  | 45                                    |
| poetry | 39.87                                 |


## Conclusions

uv is not only super fast in resolving depencencies, it manages your python versions and by default creates the environment in the local directory. With `uv` you don't need anything else, no `pyenv` for managing your python versions, or no `poetry` to build and publish your wheels. Even creating a new project boilerplate is super easy!. I have been reading out there and seems that the only drawback is that the dependency management is a bit less strict compared to `poety`. To me this is fine, this tool simply works. 