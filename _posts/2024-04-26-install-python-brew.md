---
title: Install Python using HomeBrew
author: sebastia
date: 2024-04-26 10:10:00 +0800
categories: [Python]
tags: [computer science]
pin: true
toc: true
render_with_liquid: false
math: true
---

[HomeBrew](https://brew.sh/) is a popular Linux/MacOS package installer. We will use it to install python.

## MacOS

### Prerequisites



On MacOS Brew installs the software in `/usr/local/Cellar/` and creates symlinks to the binaries in `/usr/local/opt/` and `/usr/local/bin/`. 

Install HomeBrew if you don't have it as

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

The brew command will show in `/usr/local/bin/brew`, which should be part of the `PATH`. Install python running

```bash
brew install python
```

HomeBrew will install the latest python version, in my case `3.12`. It also creates a link in `/usr/local/bin/`. To open a python prompt jus type `python3`.

If we take a close look and run `ls -lhat /usr/local/bin | grep python3` we see that `python3` is a link to `/usr/local/Cellar/python@3.12/3.12.3/bin/python3` which in turn is a link to `/usr/local/Frameworks/Python.framework/Versions/3.12/bin/python3`. The real executable (and the path to the original installation) is the latter.

I always like to have an alias to `python` rather than `python3`. These days `python2` has been deprecated and is not used in any modern project.

```bash
# append the command to the file ~/.zshrc
echo alias pytyon=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 >> ~/.zshrc
```

## Linux (Ubuntu Dist)

As prerequisites you need to have `curl` and `git`.

```bash
sudo apt-get update && apt-get install -y curl git
```

Then install HomeBrew as before


```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

The binaries will be in `/home/linuxbrew/.linuxbrew/bin`, make sure you add this to your path in `.zshrc` or `.bashrc` 

```bash
echo 'export PATH="$PATH:/home/linuxbrew/.linuxbrew/bin"' >> ~/.zshrc
```

This bin is actually a symlink of the directory `/home/linuxbrew/.linuxbrew/Homebrew/bin/` that contains the `brew` binary. Finally source the file so that changes make effect

```bash
# if you use zsh
source ~/.zshrc

# if you use bash
source ~/.bashrc
```

Now install python executing

```bash
brew install python
```

This automatically installst the latest python version, which, in my case, is `3.12`. The binary can be found in `/home/linuxbrew/.linuxbrew/bin/python3`. This path has been previoulsy added to the `PATH` so we can go ahead and type in our terminal

```bash

# open a terminal in python.
python3

# same as before but specifying the version we just installed.
python3.12
```

<!-- 
## Install another version

Say we want to install python `3.8`, we will do:

```bash
brew install python@3.8
```

And check that is there:

```bash
ls -lhat /usr/local/Cellar | grep python
 -->
