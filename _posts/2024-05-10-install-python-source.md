---
title: Install Python from source
author: sebastia
date: 2024-05-10 10:10:00 +0800
categories: [Python]
tags: [computer science]
pin: true
toc: true
render_with_liquid: false
math: true
---

In the previous posts of this series we have used installers that are platform specific so we don't need to go through the process of compiling Python. In this post we will explore how to build python from source and make it available in your system. It certainly is more cumbersome but worth learning.

## TLDR

```bash
# select version and create directory structure
export PYTHON_VERSION=3.12.3
export TEMPDIR=tmp_install

mkdir -p ~/.python/${PYTHON_VERSION}
mkdir -p ~/.python/${TEMPDIR}

# download and uncompress on tempdir
curl -O "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz" --output-dir ~/.python/${TEMPDIR}
tar -xvJf ~/.python/${TEMPDIR}/Python-${PYTHON_VERSION}.tar.xz -C ~/.python/${TEMPDIR} --strip-components=1

# configure & compile (be patient, this takes a while...)
cd ~/.python/${TEMPDIR}
./configure --enable-optimizations --prefix="${HOME}/.python/${PYTHON_VERSION}"
make
make install

# Then remove the temporary directory
rm -rf ~/.python/${TEMPDIR}

# now you can run python
cd $HOME
~/.python/${PYTHON_VERSION}/bin/python3

# make the binary available by exporting to PATH
echo "export PATH=${HOME}/.python/${PYTHON_VERSION}/bin:\$PATH" >> ~/.zshrc
source ~/.zshrc
```

Now run

```bash
python3
```

to open the python prompt


## Prerequisites

We need several libraries in our OS before we are able to compile python. In the following sections we will see how to install the needed libraries in different operating systems

### Linux Debian (e.g Ubuntu)


```bash
sudo apt update
sudo apt install build-essential \
                 zlib1g-dev \
                 libncurses5-dev \
                 libgdbm-dev \
                 libnss3-dev \
                 libssl-dev \
                 libsqlite3-dev \
                 libreadline-dev \
                 libffi-dev curl \
                 libbz2-dev \
                 liblzma-dev
```

### Linux RedHad (e.g. Fedora)

```bash
sudo dnf groupinstall "Development Tools"
sudo dnf install zlib-devel \
                 bzip2 \
                 bzip2-devel \
                 readline-devel \
                 sqlite \
                 sqlite-devel \
                 openssl-devel \
                 xz \
                 xz-devel \
                 libffi-devel
```

### MacOS

```bash
xcode-select --install
```


## Compile and install

We will install python in a custom directory in our home: `~/.python` and donwload the source files in `~/.python/tmp_dir`. Select a version from one of the versions in the [Python download page](https://www.python.org/downloads/), go to a specific release and get the link of the "XZ compressed source tarball". In this case we choose Python version `3.12.3`

```bash
# select version and create directory structure
export PYTHON_VERSION=3.12.3
export TEMPDIR=tmp_install

# creates the directories
mkdir -p ~/.python/${PYTHON_VERSION}
mkdir -p ~/.python/${TEMPDIR}
```

Now download the file source code to the temporary directory

```bash
curl -O "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz" --output-dir ~/.python/${TEMPDIR}
```

Then we need to uncompress the file into the temporary directory

```bash
tar -xvJf ~/.python/${TEMPDIR}/Python-${PYTHON_VERSION}.tar.xz -C ~/.python/${TEMPDIR} --strip-components=1
```

change directory to the temporary directory whwere we have unzipped the file and run configure having the prefix the place you want to install the software, in our case is `~/.python/${PYTHON_VERSION}`.

```bash
# configure & compile (be patient, this takes a while...)
cd ~/.python/${TEMPDIR}
./configure --enable-optimizations --prefix="${HOME}/.python/${PYTHON_VERSION}"
```

Finally compile and install:

```bash
make
make install
```

Now you have python executable in

```bash
~/.python/${PYTHON_VERSION}/bin/python3
```

To make it available to your system just install just export to PATH and source your `~/.bashrc` or `~/.zshrc`. The following will append the echo commands to the bash file while substituing `HOME` and `PYTHON_VERSION` but not `PATH`, since we don't want to write all the path there.

```bash
echo "export PATH=${HOME}/.python/${PYTHON_VERSION}/bin:\$PATH" >> ~/.zshrc
source ~/.zshrc
```

Whilst I prefer the first option, if it is easier to you you can just create an alias to your bash/zsh profile

```bash
echo alias python=${HOME}/.python/${PYTHON_VERSION}/bin/python3 >> ~/.zshrc
source ~/.zshrc
```

You are all set, just type `python` (if you have chosen aliasing) or `python3` (if you have chosen exporting path) and you will see the python prompt.
