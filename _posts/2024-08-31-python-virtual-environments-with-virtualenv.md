---
title: Python virtual environments with virtualenv
author: sebastia
date: 2024-09-01 10:15:00 +0800
categories: [Python]
tags: [computer science]
pin: true
toc: true
render_with_liquid: false
math: true
---


[virtualenv](https://virtualenv.pypa.io/en/latest/index.html) is a convenient tool to install virtual environemnts. Part of it is already implemented in venv for python>=3.3.

## Install virtualenv

`virtualenv` is a package of python so you can install via `pip install`. We could install `virtualenv` in a virtual environment and call it from there. Instead if you decide to use `virtualenv` for your projects the most common is to install it in the default interpreter, the global in `pyenv` or the default in the system `/usr/local/bin/python3` (in MacOS).

Define a global python using pyenv, as an example we will use `3.11.2`.

```bash
GLOBAL_PYTHON=3.11.2
pyenv global ${GLOBAL_PYTHON}
```

Install virtualenv on it and check the help (just to see that it works)

```bash
python -m pip install virtualenv
python -m virtualenv --help
```

## Create virtual environment with virtualenv

Now create a virtual environment with a custom python version (assuming we use `pyenv` for managing the versions) and that you installed `virtualenv` in the global pyenv python manager. You need to point to the specific python binary version for your new environment, so first install the python version for your environment if you don't have it

```bash
PYTHON_VERSION=3.12.4
pyenv install ${PYTHON_VERSION}
```

Make sure you are in the global python version in your pyenv, that's where we installed virtualenv.

```bash
pyenv shell --unset
rm -rf .python-version
```

Now you are ready to create the virtual environment

```bash
PYTHON_PATH="$(pyenv root)/versions/$PYTHON_VERSION/bin/python"
virtualenv -p ${PYTHON_PATH} .venv
```

The environmet can be found now in `.venv` directory.

## Activate and add dependencies

Activate it as usual with `source .venv/bin/activate` and install with `pip`, for instance `pip install numpy`. Then deactivate with `deactivate` and remove virtual environment by just deleting the directory `.venv`.

```bash
source .venv/bin/activate
pip install numpy

deactivate
rm -rf .venv
```
