---
title: Virtual Environments in Python
author: sebastia
date: 2024-08-04 21:05:00 +0800
categories: [Python]
tags: [computer science]
pin: true
toc: true
render_with_liquid: false
math: true
---

We have seen so far how to get different python versions in your system and briefly how to install packages using `pip install`. In real projects you want to take control over both, python version as well as packages versions and also having a fixed combination of these per project. 

Virtual environments allows you to create a completely separated environment combining a python version with a set of packages in a specific version. In this post we will show how to create a virtual environment in different ways.


## TLDR

Save the following `create_environment.sh` in the place where you want to create the virtual environment changing the python version


```bash
#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION=3.12.2

# set python version from your pyenv
PYTHON_PATH="$(pyenv root)/versions/$PYTHON_VERSION/bin/python"

# remove env and recreate
rm -rf ${SCRIPT_DR}/.venv

${PYTHON_PATH} -m venv ${SCRIPT_DIR}/.venv
${SCRIPT_DIR}/.venv/bin/python -m pip install --upgrade pip
```

Then install manually your dependencies like

```bash
.venv/bin/python -m pip install numpy
.venv/bin/python -m pip install pandas
.venv/bin/python -m pip install matplotlib
```
To pin all dependencies, freeze the environemnt into a file like

```bash
.venv/bin/python -m pip freeze > requirements.txt
```

if you want to recreate this environment you can use `requirements.txt` to reinstall all the same packages. Just create a new environment in e.g. `.venv` and run

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Sometimes it is convenient to have a single environment to run in different projects, for instance different analysis on data science. In this case I create a virtual environment in `~/.venvs` 

```bash
#!/bin/bash

PYTHON_VERSION=3.12.2
ENVIRONMENT_DIR=~/.venvs/data_science

# set python version from your pyenv
PYTHON_PATH="$(pyenv root)/versions/$PYTHON_VERSION/bin/python"

${PYTHON_PATH} -m venv ${ENVIRONMENT_DIR}
${ENVIRONMENT_DIR}/bin/python -m pip install --upgrade pip
```

Then activate it and start installing your packages

```bash
source ~/.venvs/data_science/bin/activate

pip install numpy
pip install pandas
pip install matplotlib
```

## Create a virtual environment with Venv module

Python has a default module to create virtual environments called [venv](https://docs.python.org/3/library/venv.html), the syntax is very simple, `python -m venv path/to/your/new/venv`. First select or install the python version with which you want to create the virtual environment, I normally use `pyenv` (see the <a href="../pyenv">pyenv post</a> for further reference) to do this, let's install `3.12.4` as an example


```bash
PYTHON_VERSION=3.12.4
pyenv install ${PYTHON_VERSION}
pyenv shell ${PYTHON_VERSION}
```

Now with `python --version` you should get `3.12.4`. Next you can create the virtual environment in the current directory as

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
```

And you will see this directory in the current one, jus type `ls -lhat .` . You will see three directories `include`, `lib` and `bin` and a file `pyenv.cfg`. The file just has metadata of the command used to create the virtual environment, the executable path and the python version. The real bread and butter is in the directories. 

`lib` is where all installed packages live, try to `python -m pip install numpy` and `ls` to `.venv/lib/python3.12/site-packages`, there will be a directory with your `numpy`, if you wan to investigate further go for instance to `random` and check the library there `ls -lhat .venv/lib/python3.12/site-packages/numpy/random`, in my case there is a file `_mt19937.cpython-312-darwin.so` corresponding to the shared library for the [Mersenne Twister](https://en.wikipedia.org/wiki/Mersenne_Twister). We'll see more on compiled libraries for python in a later post, it's not the topic here.

In `bin` directory there are the executables for python and pip. Run `ls -lhat .venv/bin` to inspect it. 

```bash
drwxr-xr-x  14 user  group   448B  5 Aug 18:59 .
-rwxr-xr-x   1 user  group   233B  5 Aug 18:59 f2py
-rw-r--r--   1 user  group   2.0K  5 Aug 18:37 activate
-rw-r--r--   1 user  group   8.8K  5 Aug 18:37 Activate.ps1
-rw-r--r--   1 user  group   918B  5 Aug 18:37 activate.csh
-rw-r--r--   1 user  group   2.1K  5 Aug 18:37 activate.fish
-rwxr-xr-x   1 user  group   238B  5 Aug 18:37 pip3.12
-rwxr-xr-x   1 user  group   238B  5 Aug 18:37 pip3
-rwxr-xr-x   1 user  group   238B  5 Aug 18:37 pip
lrwxr-xr-x   1 user  group     6B  5 Aug 18:37 python3.12 -> python
lrwxr-xr-x   1 user  group     6B  5 Aug 18:37 python3 -> python
lrwxr-xr-x   1 user  group    46B  5 Aug 18:37 python -> ~/.pyenv/versions/3.12.4/bin/python
```
See that python executable is actually a symbolic link to the original python binary from your freshly installed python `3.12.4` in `~/.pyenv`. This means that python is not really copied in your virtual environment but rather just using the one you used to create the environment. At the same time there's `python3.12`, `python3` both pointing to python in the directory. Find also `pip`, the default python package manager, this one is not a symbolic link and executing it will install packages inside your virtual environment as explained before. Finally there is `activate` (or `activate.chs`, `activate.fish`... for different shells) that modifies your `PATH` when it is sourced so that it can find `pip` and `python` in your virtual environment.

To activate the environment simply:

```bash
source .venv/bin/activate
```

(deactivate with `deactivate`). Then every time you type `python` it will get this environment. However, I'm a special guy and I'm not very fan of activating environments when I'm working in a specific project, I just call `python` and `pip` directy from the directory of the virtual environment:

```bash
# calling python
.venv/bin/python

# calling pip to install numpy
.venv/bin/python -m pip install numpy

# or equivalently
.venv/bin/pip install numpy
```

For convenience I always create a bash script and place it in the root of my repository, for instance

```bash
#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION=3.12.2

# set python version from your pyenv
PYTHON_PATH="$(pyenv root)/versions/$PYTHON_VERSION/bin/python"

# remove env and recreate
rm -rf ${SCRIPT_DR}/.venv

${PYTHON_PATH} -m venv ${SCRIPT_DIR}/.venv
${SCRIPT_DIR}/.venv/bin/python -m pip install --upgrade pip

# # install any kind of dependencies
# ${SCRIPT_DIR}/.venv/bin/python -m pip install -r requirements.txt

# #or install repository package
# ${SCRIPT_DIR}/.venv/bin/python -m pip install ${SCRIPT_DIR}
```

which will install a new virtual enviroment in the directory where the script is placed using python 3.12.2. 

> TIP: Sometimes there's no need to create a new repository per project. For instance, if you are a data scientist crunching some numbers here and there you may want to use the same virtual environment without paying too much attention to what is the python or package versions. For those cases create virtual environments in your home directory like `mkdir ~/.venvs && python -m venv ~/.venvs/data_science`, and activate with `source ~/.venvs/data_science/bin/activate`.
{: .prompt-tip }

# Install and pin dependencies on a virtual environment

Once created a virtual environment, the most common is to install your dependencies. No matter what you do in python, it's 99% probable that you need an external package. In command line you can just `pip install package` whatever dependency specifying the version so that another developer can reproduce your code with exactly the same results. The same could be applied for services (e.g. REST APIs) for which need to be shut down, deleted and rebuilt and started. For this reason we must pin dependencies for reproducible outcomes, we do that with a file called `requirements.txt`. 

Let's install `numpy`, `pandas` and `matplotlib` to our virtual environment

```bash
.venv/bin/python -m pip install numpy
.venv/bin/python -m pip install pandas
.venv/bin/python -m pip install matplotlib
```

pip will interpret at this point which is the most suitable version for each of these packages, usually the latest one if your python version is quite new, to see which ones were installed specifically we can run

```bash
.venv/bin/python -m pip freeze
```

In my case, I see `numpy==1.21.6`, `pandas==1.3.5` and `matplotlib==3.5.3` and a bunch of other packages. Two things to note here, first, the `==` means that it uses that specific version on fhe package and no other. The second why are there other packages if we just installed three?. Turns out these three packages use other packages at the same time and those have to be pinned too!. You can see the whole depencency with a package I recently found (and I love) called `pipdeptree`:


```bash
# install pipdeptree
.venv/bin/python -m pip install pipdeptree

# run pipdeptree
.venv/bin/pipdeptree
```

which gives the following:

```bash
matplotlib==3.5.3
  - cycler [required: >=0.10, installed: 0.11.0]
  - fonttools [required: >=4.22.0, installed: 4.38.0]
  - kiwisolver [required: >=1.0.1, installed: 1.4.5]
    - typing-extensions [required: Any, installed: 4.7.1]
  - numpy [required: >=1.17, installed: 1.21.6]
  - packaging [required: >=20.0, installed: 24.0]
  - Pillow [required: >=6.2.0, installed: 9.5.0]
  - pyparsing [required: >=2.2.1, installed: 3.1.4]
  - python-dateutil [required: >=2.7, installed: 2.9.0.post0]
    - six [required: >=1.5, installed: 1.16.0]
pandas==1.3.5
  - numpy [required: >=1.17.3, installed: 1.21.6]
  - python-dateutil [required: >=2.7.3, installed: 2.9.0.post0]
    - six [required: >=1.5, installed: 1.16.0]
  - pytz [required: >=2017.3, installed: 2024.1]
pip==24.0
pipdeptree==2.9.6
setuptools==40.8.0
```

So `matplotlib` depends on `cycler`, `fonttools`, `kiwisolver`... , `pandas` on `numpy`, `python-dateutil`..., `pipdeptree` does not depend on any other package and finally we have the default `pip`, and `setuptools` that come with any python installation. See that the specific version of `matplotlib` (and `pandas`) pinned and their dependencies are pinned too (e.g. `Pillow==9.5.0`). In all the sub-dependencies we have a `required>=` indicating that the package could work with another version of the sup-dependency as long as this requirement is met. The task of `pip` is to figure out the versions of the dependencies and supdependencies so that everything is compatible.

Why would one not pin a dependency when building a package?. Easy, python packages need to be flexible, they should work for a range of python versions and package dependencies, if you constrict too much your package nobody will use it as it will be incompatible with other packages, but we will talk more about that in another post.

Getting back to topic, we have a pinned version of the packages, how can we give it to someone (another developer) to install the same exact environment as you? Introducing the file `requirements.txt`. Seems legacy but still everyone is using it to pin dependencies. To generate it, run

```bash
.venv/bin/python -m pip freeze > requirements.txt
```

and there you go!. Pass it to your friend, make sure he uses the same python version as you and he/she'll need to run


```bash
.venv/bin/python -m pip instll -r requirements.txt
```

This will get him exactly the same environment.
