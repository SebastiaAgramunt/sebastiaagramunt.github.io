---
title: Python virtual environments with pipenv
author: sebastia
date: 2024-09-01 23:05:00 +0800
categories: [Python]
tags: [computer science]
pin: true
toc: true
render_with_liquid: false
math: true
---

As they describe in [pipenv documentation](https://pipenv.pypa.io/en/latest/): Pipenv is a Python virtualenv management tool that supports a multitude of systems and nicely bridges the gaps between pip, python (using system python, pyenv or asdf) and virtualenv. So, virtualenv and pyenv? seems like a desirable integration!.

## TLDR pipenv

Using `pyenv` install `pipenv` in global pyenv, then create a virtual environment in the current directory

```bash
# install pipenv
GLOBAL_PYTHON=3.12.4
pyenv global ${GLOBAL_PYTHON}

python -m pip install pipenv

# create environment
pyenv shell --unset
rm -rf .python-version

PYTHON_VERSION=3.10.14
PYTHON_PATH="$(pyenv root)/versions/$PYTHON_VERSION/bin/python"

export PIPENV_VENV_IN_PROJECT=1
pipenv --python ${PYTHON_PATH}

# install some packages
pipenv install numpy
pipenv install pandas

# show dependency tree
pipenv graph

# launch python
.venv/bin/python -c "import numpy as np"

# activate environment
source .venv/bin/activate

# deactivate
deactivate environment
```

## Install

As virtualenv we need to pip install to our global python (which again I choose to `3.11.2`)

```bash
GLOBAL_PYTHON=3.11.2
pyenv global ${GLOBAL_PYTHON}

python -m pip install pipenv
python -m pipenv --help
```

## Create a new environment

Before creating the environment make sure you are in the `global` pyenv version where you installed `pipenv` as a command line tool:

```bash
pyenv shell --unset
rm -rf .python-version
```

Now create a virtual environment for a specific python version indicating where is the binary of that python in pyenv.

```bash
PYTHON_VERSION=3.10.14
PYTHON_PATH="$(pyenv root)/versions/$PYTHON_VERSION/bin/python"

export PIPENV_VENV_IN_PROJECT=1
pipenv --python ${PYTHON_PATH}
```

this will create a virtual environment in your current directory under `.venv` and a `Pipfile` too. If `PIPENV_VENV_IN_PROJECT` is not set to 1 the environment will be installed somewhere else in the system (in MacOS in `~/.local/share/virtualenvs/`). 

Start installing packages

```bash
pipenv install numpy
pipenv install pandas
```

And the dependencies will be added in `Pipfile`:

```toml
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
numpy = "*"
pandas = "*"

[dev-packages]

[requires]
python_version = "3.10"
python_full_version = "3.10.14"

```
This is not saying much about the numpy and pandas versions, it basically allows any version of those packages. If you want something more specific for numpy for instance, you can run `pipenv install "numpy>=1.20,<2.0"`.


But obviously we have installed two packages and in our eviroinment they have a specific version. That is specified in the new file created `Pipfile.lock`, those are the pinned dependencies we were mentioning earlier in this post. Let's inspect a bit `Pipfile.lock`:

```json
  "numpy": {
      "hashes": [
          "sha256:08801848a40aea24ce16c2ecde3b756f9ad756586fb2d13210939eb69b023f5b",
          "sha256:0937e54c09f7a9a68da6889362ddd2ff584c02d015ec92672c099b61555f8911",
          "...",
          ]
      "index": "pypi",
      "markers": "python_version >= '3.10'",
      "version": "==2.1.0"
  }
```

First notice the field "version", it specifies exactly the version installed in the virtual environment, also "python_version" indicates for which python versions this package is compatible with. This is great to know what can you modify in your environment and solve dependencies. A third non so trivial part are the hashes, these are the checksums for the downoaded packages allowed for `numpy` package. Or in other words, when we download the numpy wheel (zipped package in python) and we do the checksum, it has to mach one of the ones in the list, with this we avoid corruption of the file for security. Also, all these checksums are the hashes for all the platforms and operating systems!. So, if you develop in MacOS and you send your friend this lock file, he will have the checksum in the file and so it will be compatible with his system (at least in theory).

Let's continue by activating environment created

```bash
source .venv/bin/activate
```

A very nice feature of pipenv is `pipenv graph` where it is displayed on the screen the dependencies of your packages:

```bash
pandas==2.2.2
├── numpy [required: >=1.22.4, installed: 2.1.0]
├── python-dateutil [required: >=2.8.2, installed: 2.9.0.post0]
│   └── six [required: >=1.5, installed: 1.16.0]
├── pytz [required: >=2020.1, installed: 2024.1]
└── tzdata [required: >=2022.7, installed: 2024.1]
```

This feature is very useful if trying to resolve complex inconsistencies in your package. In this example you can see that even though we manually installed `numpy` and `pandas` we only needed to install `pandas` as `numpy` is already a dependency of `pandas`. 

## Pin the environment

We have seen that being able to specify the python package versions is important for other developers to build the same environment. If your friend uses `pipenv` too, it will be enough to send him the `Pipfile` and the `Pipfile.lock`. Otherwise you can create the `requirements.txt` file too with

```bash
pipenv requirements > requirements.txt
```
In our example the contents of the file will be

```
-i https://pypi.org/simple
numpy==2.1.0; python_version >= '3.10'
pandas==2.2.2; python_version >= '3.9'
python-dateutil==2.9.0.post0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'
pytz==2024.1
six==1.16.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'
tzdata==2024.1; python_version >= '2'
```

Then, for instance, if you want to use `venv` module to recreate the environment you can do

```bash
rm -rf .venv
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

which, if you do `pip freeze` gives the versions

```
numpy==2.1.0
pandas==2.2.2
python-dateutil==2.9.0.post0
pytz==2024.1
six==1.16.0
tzdata==2024.1
```
that are the same. 


## Remove a pipenv environment

Finally, remove the environment with:

```bash
rm -rf .venv
rm -rf Pipfile
rm -rf Pipfile.lock
```

## Conclusions

I've always preferred `venv` for building virtual environments since it comes by default with Python. I'm not a fan of `conda`, but I recently explored `pipenv` and was pleasantly surprised. I'll still use `venv` for my repositories but will definitely use `pipenv` for:

- Upgrading Python dependencies: `pipenv graph` makes it much easier to upgrade both Python and package versions.
- Generating a more detailed `requirements.txt`, including Python version info, which is useful for developers using `venv` or other tools.
- Cross-platform builds: Though I haven't needed it yet, `Pipfile.lock` could simplify building environments on different platforms and reduce build times, thanks to its platform-specific hashes.