---
title: Useful CLI tools
author: sebastia
date: 2025-02-18 23:09:00 +0800
categories: [Tools]
tags: [computer science]
pin: true
toc: true
render_with_liquid: false
math: true
---

This post shows quick installation for command line tools that I use in my day to day.

## LSD - ls Deluxe

A better `ls ` is [lsd](https://github.com/lsd-rs/lsd)

For MacOs

```bash
brew install lsd
```

or in Linux 

```bash
apt install lsd
```

Or install binaries, (this one is for arch64 MacOs)

```bash
VERSION="v1.1.5"
ARCHITECTURE="aarch64"
OS="apple-darwin"

FNAME=lsd-${VERSION}-${ARCHITECTURE}-${OS}
wget "https://github.com/lsd-rs/lsd/releases/download/${VERSION}/${FNAME}.tar.gz"

tar -xzvf ${FNAME}.tar.gz
mkdir -p ~/bin

cp ${FNAME}/lsd ~/bin
rm -rf ${FNAME}.tar.gz
rm -rf ${FNAME}
```

## Hyperfine

Hyperfine is a "timeit" tool, measure time of a bash command. Install on MacOS as

```bash
brew install hyperfine
```

And on linux

```bash
VERSION="v1.19.0"
ARCHITECTURE="x86_64"
OS="unknown-linux-musl"

FNAME="hyperfine-${VERSION}-${ARCHITECTURE}-${OS}"
wget "https://github.com/sharkdp/hyperfine/releases/download/${VERSION}/${FNAME}.tar.gz"

tar -xzvf ${FNAME}.tar.gz
mkdir -p ~/bin

cp ${FNAME}/hyperfine ~/bin
rm -rf ${FNAME}.tar.gz
rm -rf ${FNAME}
```


## Android platform tools

```bash
brew install android-platform-tools
```