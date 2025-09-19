---
title: BLAS and LAPACK Install
author: sebastia
date: 2025-09-14 12:35:00 +0800
categories: [C++]
tags: [computer science, mathematics]
pin: true
toc: true
render_with_liquid: false
math: true
---

## Install OpenBlas and Lapacke in MacOS

In MacOS simply use [Homebrew](https://brew.sh/), if you don't have it just install with the command

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then you can install the two libraries as

```bash
brew install openblas lapack
```

These are installed in 

```bash
OPBENBLAS_INSTALL_DIR=$(brew --prefix openblas)
LAPACK_INSTALL_DIR=$(brew --prefix lapack)
```

There you will find the subdirectories `include` and `lib` that will be needed to compile and link your program that uses BLAS and LAPACK.

Apple’s [Accelerate](https://developer.apple.com/documentation/accelerate) provides BLAS/LAPACK with `-framework Accelerate`, but it does not provide LAPACKE. Use Homebrew’s lapack if you need `lapacke.h`. I'm good with the brew solution.

## Install OpenBlas and Lapack in Ubuntu

```bash
sudo apt-get update
sudo apt-get install -y libopenblas-dev liblapacke-dev
```
I tried the above in a Ubuntu docker container in my MacOS:

```bash
docker pull ubuntu

# run ubuntu container (open bash prompt)
docker run --rm -it --entrypoint bash ubuntu
```

And found the headers in 

* `/usr/include` for `lapack.h`, `lapacke.h`
* `/usr/include/x86_64-linux-gnu` for `cblas.h`

And libraries in

* `/usr/lib/x86_64-linux-gnu/` for `libblas.a`, `libblas.so`, `libopenblas.a`, `libopenblas.so` (static and dynamic libraries for BLAS).
* `/usr/lib/x86_64-linux-gnu/` for `liblapack.a`, `liblapack.so`, `liblapacke.a`, `liblapacke.so` (static and dynamic libraries for lapack and lapacke).

To double check the flags with `pkg-config`

```bash
apt-get install pkg-config

# includes
echo $(pkg-config --cflags openblas lapacke)

# libraries
echo $(pkg-config --libs   openblas lapacke)
```

## Install OpenBlas and Lapack in Conda

## Compile and install Openblas and Lapack

This is my favourite way to install libraries, download the source code and install in your project directory. I agree it takes more time but if you only use these libraries in one project may be worth just installing them in one directory. The project structure subdirectory for this install is

```bash
.
├── README.md
├── scripts
│   ├── build-run.sh
│   └── install-external-libraries.sh
└── src
    └── cblas_example.cpp
```

Let's first build the libraries

### Download and build OpenBlas

We can download the source code from github's official page, we will uinstall the most recent version currently which is `0.3.30`:

```bash

# create a directory where you will build the software
mkdir external
cd external

# select openblas version
OPENBLAS_VERSION="0.3.30"
OPENBLAS_URL="https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz"

# install dir, change this to wherever you want
INSTALL_DIR=${HOME}/libs

# download
wget ${OPENBLAS_URL}

# untar
tar -xvzf OpenBLAS-${OPENBLAS_VERSION}.tar.gz

# go to the untarred directory
cd OpenBLAS-${OPENBLAS_VERSION}

# compile the library
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      ..

make -j 64
make install
cd ../../..
```

Now, you will find the includes and libs the installation dir. Just ls the directories

```bash
ls $INSTALL_DIR/include/openblas
```

where you will find `cblas.h`, `lapack.h`, `lapacke.h`.

Also the libraries

```bash
ls $INSTALL_DIR/lib
```

to find `libopenblas.dylib` (in MacOS)