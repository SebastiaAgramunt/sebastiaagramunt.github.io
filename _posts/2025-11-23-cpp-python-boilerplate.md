---
title: C++ Python package boilerplate
author: sebastia
date: 2025-11-23 10:28:00 +0800
categories: [C++, Python]
tags: [computer science]
pin: true
toc: true
render_with_liquid: false
math: true
---

Previously in <a href="../../posts/cpp-python-extension">C++ basic Python extension</a> we learned the basic mechanism on building a C++ extension for Python. Here in this post we will be more practical and we will create a full end to end package that is fully tested and builds the wheels for different platforms and architectures. As before we will use pybind11 to create the bindings. This entire repository [python-boilerplate](https://github.com/SebastiaAgramunt/python-boilerplate) lives in my GitHub account and not in the [blogging-code](https://github.com/SebastiaAgramunt/blogging-code) where I usually publish.

## Project structure

The files and directories are the following after runnint `tree` command

```bash
.
├── Docker
│   ├── Dockerfile-python-3.13
│   └── build-run.sh
├── MANIFEST.in
├── README.md
├── include
│   └── matmul.h
├── pyproject.toml
├── scripts
│   ├── build_wheel.sh
│   └── example.py
├── setup.py
├── src
│   ├── bindings.cpp
│   ├── matmul.cpp
│   └── package_example
│       ├── __init__.py
│       └── operations.py
└── tests
    └── test_matmul.py
```

In the `src` we include all the `*.cpp` files, including the `bindings.cpp` written using pybind11. The `src` directory also contains the python package files under the package name directory `package_example`. The `include` directory has all the C++ headers. The `Docker` directory contains a docker file and a bash script to build and run the image. Then the usual `README.md` for documenting the build, publication etc. The `test` directory is where we place the tests, sometimes it is also recommended to create tests for the C++ part before binding it to Python, however not to overcomplicate things in this boilerplate we just create python tests using the bindings. The way we build the package is done with `setup.py` and `pyproject.toml`.

## Building the package

The code in `matmul.cpp`, `matmul.h` and `bindings.cpp` is simply a matrix multiplication and its bindings in C++ so we won't really comment into that, go to <a href="../../posts/cpp-python-extension">C++ basic Python extension</a>  to learn more. Comparing to that post the build is different, let's start by the `pyproject.toml` file:

```toml
[build-system]
requires = ["setuptools>=64", "wheel", "pybind11>=2.10"]
build-backend = "setuptools.build_meta"


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
    "I001",
]

[tool.ruff.format]
quote-style = "single"
indent-style = "space"
docstring-code-format = true
docstring-code-line-length = 20

[tool.mypy]
python_version = "3.13"
ignore_missing_imports = true
exclude = "^(build/|\\.venv/)"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
```

In this file we only specify the build system, which is setuptools which is the default built system in python but it is not part of the Python standard library. Tools we will use in this project use setuptools like pybind11 and cibuildwheel. Aside from the build-system we have configuration information for [ruff](https://github.com/astral-sh/ruff), [mypy](https://github.com/python/mypy) and [pytest](https://docs.pytest.org/en/stable/).


The file `setup.py` contains the real bread and butter on the compilation of the project, something that in the previous post we did manually. 

```python
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages
from pathlib import Path
import sysconfig

__version__ = '0.0.1'

REPO_PATH = Path(__file__).resolve().parent

PACKAGE_NAME = 'package_example'

PYTHON_LIB_INCLUDES = sysconfig.get_path('include')
PACKAGE_LIB_INCLUDES = REPO_PATH / 'include'

SRC_FILES = [
    str(REPO_PATH / 'src' / 'matmul.cpp'),
    str(REPO_PATH / 'src' / 'bindings.cpp'),
]

EXTRA_COMPILE_ARGS = ['-O3', '-std=c++17']

ext_modules = [
    Pybind11Extension(
        PACKAGE_NAME + '._core',
        SRC_FILES,
        include_dirs=[PYTHON_LIB_INCLUDES, str(PACKAGE_LIB_INCLUDES)],
        extra_compile_args=EXTRA_COMPILE_ARGS,
        define_macros=[('VERSION_INFO', __version__)],
    ),
]

setup(
    name=PACKAGE_NAME,
    version=__version__,
    author='Sebastia Agramunt Puig',
    author_email='contact@agramunt.me',
    url='https://github.com/SebastiaAgramunt/python-boilerplate',
    description='Example package with C++ extension',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    python_requires='>=3.9,<3.14',
    install_requires=[
        'numpy>=1.20',
    ],
    extras_require={'test': ['pytest', 'ruff', 'mypy', 'pre-commit']},
)
```

Starting from the beginning, we have a variable `__version__`, this will be the version of our package, change it on every release. For convenience we define some variables `PACKAGE_NAME`, is the name we will give to our package. Then the variable `PYTHON_LIB_INCLUDES` is where our python header files live (i.e. `Python.h`), needed for the bindings compilation. We define the includes of our project in `PACKAGE_LIB_INCLUDES` and finally the source files in a list of `SRC_FILES`. Some optimization for the compiler may be needed so I added performance flags like `-O3` and `c++17` standard. Then we define the external modules in the `ext_modules` variable, the inputs are obvious. The final setup is defined through the function `setup` from setuptools. Here we specify the python version range, the required packages and as a bonus the extra requirements that we may want to use for testing.

A file that is sometimes disregarded is the `MANIFEST.in` file:

```bash
# MANIFEST.in needed to include non-Python files in the package to build wheel distributions
# e.g. C++ source files, headers, README, pyproject.toml, etc.

include pyproject.toml
include README.md

recursive-include src *.py *.cpp *.hpp *.h
recursive-include include *.hpp *.h
```

this is key if you want to release wheels. Essentially it tells python to include files from the directory and ship them in your compiled wheel.

To install from source just create a new environment

```bash
rm -rf .venv
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
```

and then use pip to install it

```bash
.venv/bin/python -m pip install .
```

This will compile C++, the bindings and add your python code. After installing you should be able to see the compiled file and python sources:

```bash
ls -lhat .venv/lib/python3.13/site-packages/package_example 
```

In my case (running this in MacOS) I find the file `_core.cpython-313-darwin.so`, that is our C++ shared library. Also the file `operations.py` and the `__init__.py`.

To really confirm the package is installed and working, run the example script

```bash
.venv/bin/python -c "import package_example"
.venv/bin/python scripts/example.py
```

## Building the wheel locally

Instead of building from source each time we can build a wheel and pull this wheel to other projects. We will do this manually and also with GitHub actions. In this section we will learn the manual way of generating a wheel.

In `scripts/build_wheel.h` you will find a bash script that crates the package. The build is different for Linux or MacOS (we won't cover Windows here). We have a function to create an environment called `crate_venv` that is executed regardless, then depending on the operating system we use `build_wheel_linux` or `build_wheel_macos`. The wheel is built with the command

```bash
python -m build "$PROJECT_DIR"
```

and the wheels are placed in `dist` directory.

In the case of the wheel in Linux we do extra things. First we identify which architecture are we on `x86` or `aarch64` and give the platform tag `manylinux_2_28_x86_64` for the first and `manylinux_2_28_aarch64` for the latter. This will be used to repair the wheel.

Manylinux is a Linux compatibility standard for Python wheels. Its purpose is to allow developers to build binary wheels (wheels that contain compiled C/C++ code) that work on most Linux distributions, even very old ones. Linux distributions vary a lot, different glibc versions, compiler versions, system libraries... Manylinux solve this problem. Specifically in this case we will repair the wheel so that it is compatible with `glibc` version 2.28 and above for the two architectures.

We use [auditwheel](https://github.com/pypa/auditwheel) as mentioned to repair the wheel. 

```bash
auditwheel repair "$WHEEL_FILE" \
--plat "$PLATFORM_TAG" \
-w "$PROJECT_DIR/wheelhouse"
```

This program will include all the dependencies needed all libraries that are used in your package (shared objects) will be included. The problem with this is that it could potentially add super large libraries like `libcuda` if your project is compiled cuda code. You really don't need this library because it will be installed in the machine you will be running the code (otherwise how can you talk to the GPU?). To exclude libraries and make your wheel a bit smaller use the `--exclude` flag, i.e. `--exclude libcu* --exclude libnvcomp*`.

That's it, your repaired linux wheel will be saved in the `wheelhouse` directory.

## CI/CD on GitHub Actions

In `.github/workflows/build-wheels.yml` we placed some code that builds (compiles) the wheels, runs the tests and publishes the wheels when tagging a release. Let´s inspect the file `build-wheels.yml`:

```yml
name: Build wheels

on:
  push:
    branches: [ main, master ]
    tags: [ "v*" ] # only build/publish on version tags
  pull_request:
```

This indicates that the job will be triggered in the `main` and `master` branches (usually you just have one of these), on all pull requests and on tags starting with `v` (we will name our versions like `v0.0.1`).

Then we define two jobs, `build_wheels` and `publish_release_assests`. The firts job starts with


```yml
build_wheels:
  name: Build wheels on ${{ matrix.os }}
  runs-on: ${{ matrix.os }}

  strategy:
    fail-fast: false
    matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
```

Tells us to run the job in three operating systems. 


Then the steps to follow for each of the operating systems is

```yml
steps:
  - uses: actions/checkout@v4

  - name: Set up Python
    uses: actions/setup-python@v5
    with:
      python-version: "3.11"

  - name: Install cibuildwheel
  run: |
      python -m pip install --upgrade pip
      python -m pip install cibuildwheel

  - name: Build wheels with cibuildwheel
    env:
      CIBW_BUILD: "cp3{10,11,12,13}-*"
      CIBW_SKIP: "pp* *-musllinux_*"
      CIBW_ARCHS_MACOS: "x86_64 arm64"
      CIBW_TEST_REQUIRES: "pytest numpy"
      CIBW_TEST_COMMAND: "pytest -q {project}/tests"
    run: |
        cibuildwheel --output-dir wheelhouse

  - name: List built wheels
    run: ls wheelhouse

  - name: Upload wheels as artifact
    uses: actions/upload-artifact@v4
    with:
      name: wheels-${{ matrix.os }}
      path: wheelhouse/*.whl
```

The steps use the [actions/checkout@v4](https://github.com/actions/checkout), that only checks out your repository under the `$GITHUB_WORKSPACE` so that your runner on github can access it. The first action is just setting up python to be used by the next step, the `cibuildwheel` installation. Then we build the wheels using `cibuildwheel`, we specify architectures, python versions and testing requiremtents. This will build all the wheels and test our repository. Then just for sanity check we list the `wheelhouse` directory, where we decided to place the wheels in the previous step. Finally we upload the artifact using the [actions/upload-artifact@v4](https://github.com/actions/upload-artifact), this will store the wheels into an internal github storage. And that's it, after this action is executed we will have a bunch of wheels in different platforms and architectures already tested.

The second action is `publish_release_assests`, this one starts with

```yml
publish_release_assets:
  name: Attach wheels to GitHub Release
  needs: build_wheels
  runs-on: ubuntu-latest
  if: startsWith(github.ref, 'refs/tags/v')
```

which specifies that only needs to be run on `ubuntu-latest` and after `build_wheels` has run successfully. Also only trigger this job for a tagged release starting with `v`. The steps are the following

```yml
steps:
  - name: Download wheel artifacts
    uses: actions/download-artifact@v4
    with:
      pattern: wheels-*
      path: ./artifacts
      merge-multiple: true

  - name: List downloaded wheels
    run: |
      echo "Contents of ./artifacts:"
      ls -R ./artifacts || echo "No artifacts found"

  - name: Create / update GitHub Release and upload wheels
    uses: softprops/action-gh-release@v2
    with:
      files: artifacts/*.whl
    env:
      GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

The first downloads the weheels published internally to our `./artifacts` directory. The second just lists the wheels and the third uses `softprops/action-gh-release@v2` action to publish the wheels in the tagged release.

To trigger this job `publish_release_assets` we tag the release once we merge a PR to master. This can be done executing the following in your master branch locally:

```bash
VERSION=0.0.5
git tag -a v${VERSION} -m "v${VERSION}"
git push origin v${VERSION}
```

The job will be triggered and you need to wait a bit to see all the artifacts in the [repository/releases/tag/v0.0.5](https://github.com/SebastiaAgramunt/python-boilerplate/releases/tag/v0.0.5) in our example. That's it, you have all 

## Install the package from source

You can build the package locally

```bash
rm -rf .venv
# create a virtual environment with uv (yes, my new favorite tool)
uv venv .venv -p 3.13
uv pip install .
```

Now you can try and run the tests

```bash
uv pip install pytest
uv run pytest .
```

If you want to use `pyenv` instead you can do

```bash
pyenv shell 3.13
.venv/bin/python -m pip install .
```

and run the tests to try it

```bash
.venv/bin/python -m pip install pytest
.venv/bin/pytest .
```

## Install the pacakge from the wheel

Once you have your wheel uploaded It's very easy to pull the wheel from GitHub and install it in your environment. Let's create a new virtual environment with Python 3.13 on a Mac with the new ARM64 CPU chip.

As before create the virtual environment (I use `uv` now)

```bash
rm -rf .venv
uv venv .venv -p 3.13
```

Now you can install the wheel that we crated in CI/CD on GitHub actions

```bash
PKG_VERSION="0.0.5"
PKG_NAME="package_example-${PKG_VERSION}-cp313-cp313-macosx_11_0_arm64.whl"
PKG_URL="https://github.com/SebastiaAgramunt/python-boilerplate/releases/download/v${PKG_VERSION}"
WHEEL="${PKG_URL}/${PKG_NAME}"
uv pip install ${WHEEL}
```

and just import the package to see if it has been installed

```
.venv/bin/python -c "import package_example"
```

And that's it, you can tell your friends to install from your wheel direclty to their system!, all compiled, no problems!.

## Using the package

Now how do we use the pacakge, in `scripts/example.py` we have a example script to use the code:

```python
import numpy as np
import package_example as pe  # this uses your __init__.py exports


def main():
    # Create a random matrix A of shape (M, N)
    M, N = 4, 3
    A = np.random.randn(M, N).astype(np.float32)

    print('A:')
    print(A)

    B = np.random.randn(N, M).astype(np.float32)

    print('\nB:')
    print(B)

    # Prepare output matrix C
    C = np.zeros((M, M), dtype=np.float32)

    # multiply A and B using the C++ extension
    pe.matmul(A, B, C)

    print('\nA * B:')
    print(C)

    # Verify correctness using pure NumPy
    C_np = A @ B

    print('\nNumPy result:')
    print(C_np)

    print('\nDifference (should be near zero):')
    print(C - C_np)


if __name__ == '__main__':
    main()
```

Install the package using the environment and then run the script

```bash
.venv/bin/python scripts/example.py
```

In the script we just calcualte using the exposed function `matmul` in python (backend in pure C++) and the same in the usual `numpy`. We print out the difference of the two results which should be zero. I haven't tested the speedup but my implementation should be worse than numpy. Certainly, numpy already uses `BLAS` and `LAPACKE` libraries, which are already very optimized for numerical computing. This post is just an example on how to create C++ bindings.

## Final remarks

I hope this end to end python project for C++ bindings has been useful to you. I tried to add most of the basic ingredients to create it. Hope you can create amazing Python packages with C++ backend and obviously share them with the communtiy. Have fun coding.
