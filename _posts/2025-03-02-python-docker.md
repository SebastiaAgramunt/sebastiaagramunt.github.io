---
title: Docker images for python
author: sebastia
date: 2024-09-01 10:15:00 +0800
categories: [Python]
tags: [computer science]
pin: true
toc: true
render_with_liquid: false
math: true
---

Docker is a containerization platform that allows developers to package applications and their dependencies into containers. Containers provide an isolated environment that runs the application the same way across different systems. In this post I'm going to provide some docker images to run on Python.

The code for this post is in my repository [blogging-code](https://github.com/SebastiaAgramunt/blogging-code/tree/main), subdirectory [python-dockerfiles](https://github.com/SebastiaAgramunt/blogging-code/tree/main/python-dockerfiles).

## Install Docker and Colima on MacOS

First we need `docker` as the engine

```bash
brew install docker
brew link docker
docker --version
```

And then install [Colima](https://github.com/abiosoft/colima) which is an open source alternative to Docker Desktop. 

```bash
brew install colima
```

Start colima as the engine for docker

```bash
brew start colima
```

When you no longer need your containers running make sure to do `brew stop colima`. 

## Python on Debian-based distributions

From [Python DockerHub page](https://hub.docker.com/_/python) we have several options to crate a customized docker image. Let's say wa want to run an application in python 3.12. Normally you would pick between the images `python:3.12`, `python:3.12-slim` or `python:3.12-alpine`. The first image is the largest ~915MB, and contains more libraries and tools than needed for running basic python (although not specified in the docs). The second image (the slim) is a simplified version of the first one and weights about 45MB, the latter (alpine) is an even more minimal version with size about 24MB. We won't use the alpine, it is so basic we can't even add users, also wouldn't be able to install numpy and other packages since numpy is based on `glibc` library and uses `musl` libc.


### Python-slim
To create new custom images we write Dockerfiles. Those are plaintext files that tell docker how to build the image. For the first example create a file named `Dockerfile-python-3.12-slim` with the following content

```bash
FROM python:3.12-slim

ARG USERNAME=user
ARG UID=1000
ARG GID=1000

RUN if getent group ${GID} >/dev/null; then \
        echo "Group with GID ${GID} already exists, using it."; \
        GROUP_NAME=$(getent group ${GID} | cut -d: -f1); \
    else \
        GROUP_NAME=${USERNAME}; \
        groupadd --gid ${GID} ${GROUP_NAME}; \
    fi && \
    useradd --uid ${UID} --gid ${GID} --create-home --shell /bin/bash ${USERNAME}

WORKDIR /home/${USERNAME}
RUN chown -R ${UID}:${GID} /home/${USERNAME}

USER ${USERNAME}
CMD ["/bin/bash"]
```

This starts from the image `python:3.12-slim` that docker will pull from dockerhub. The instructions are to create a new group if it doesn't exist and add a user. Why we do this?. Docker normally operates as root so if you create a new image and a container from it, you will be root. In this case if you mount a sensitive directory from your system (imagine you mount `/usr/lib`) and accidentally delete it in your container, you would be in trouble. A workaround I found for this not to happen is to crate a user in the container that is the same you have in the host with the same group id, thus removing the root by default.

The last part of the dockerfile sets the working directory (home for the user) and changes the privileges of it.  Finally the command USER sets the user that will be used when you execute a container.

Now we can build and run the image with

```bash
docker build -f Docker/Dockerfile-python-3.12 \
            --build-arg USERNAME=$(whoami) \
            --build-arg UID=$(id -u) \
            --build-arg GID=$(id -g) \
            -t python-3.12-image .
```

And then run and ssh into it with


```bash
docker run -it python-3.12-image /bin/bash
```

That's it, now you are in the container, type `python --version` to check that your version is `3.12`. In another terminal check the images and containers you have in the system

```bash
# show all containers (running and stopped)
docker ps -a 

# show images
docker images
```

stop the container and remove the image with

```bash
docker rm container
docker rmi python-3.12-image
```

Then prune

```bash
docker container prune
docker image prune
```

### Python compiled

In this case we will download and install python from source on docker, open a document with name `Dockerfile-python-3.12-build`, pick your version from [FTP Python page](https://www.python.org/ftp/python/).

```bash
FROM ubuntu:latest

ARG USERNAME=user
ARG UID=1000
ARG GID=1000

ENV PYTHON_VERSION=3.12.9

# install required packages to compile python
RUN set -x \
    && echo "Updating..." \
    && apt-get upgrade \
    && apt-get update \
    && echo "Installing Packages..." \
    && apt-get install -y \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libsqlite3-dev \
    libreadline-dev \
    libffi-dev curl \
    libbz2-dev \
    liblzma-dev \
    wget

# download and compile python
RUN cd usr/src && \
    PYTHON_VERSION_SHORT=${PYTHON_VERSION%.*} && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j 16 && \
    make altinstall && \
    ln -s /usr/local/bin/python${PYTHON_VERSION_SHORT} /usr/bin/python && \
    cd / && \
    rm -rf /usr/src/Python-${PYTHON_VERSION}.tgz /usr/src/

RUN if getent group ${GID} >/dev/null; then \
        echo "Group with GID ${GID} already exists, using it."; \
        GROUP_NAME=$(getent group ${GID} | cut -d: -f1); \
    else \
        GROUP_NAME=${USERNAME}; \
        groupadd --gid ${GID} ${GROUP_NAME}; \
    fi && \
    useradd --uid ${UID} --gid ${GID} --create-home --shell /bin/bash ${USERNAME}

WORKDIR /home/${USERNAME}
RUN chown -R ${UID}:${GID} /home/${USERNAME}

USER ${USERNAME}
CMD ["/bin/bash"]
```

### Mamba & Conda

A third docker image that can be useful is one containing mamba and conda. A container to do data science perhaps. I do not recommend this container to run apps or microservices. Let's name the dockerfile as `Dockerfile-mamba`.

```bash
FROM ubuntu:latest

ARG USERNAME=user
ARG UID=1000
ARG GID=1000

RUN set -x \
    && echo "Updating..." \
    && apt-get upgrade \
    && apt-get update \
    && echo "Installing Packages..." \
    && apt-get install -y \
    wget 

# Define Miniforge version and install path
ENV MINIFORGE_PATH=/opt/miniforge

# Download and install Miniforge
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/Miniforge.sh \
    && bash /tmp/Miniforge.sh -b -p $MINIFORGE_PATH \
    && rm /tmp/Miniforge.sh

# Set environment variables for Conda and Mamba
ENV PATH="$MINIFORGE_PATH/bin:$PATH"

RUN if getent group ${GID} >/dev/null; then \
        echo "Group with GID ${GID} already exists, using it."; \
        GROUP_NAME=$(getent group ${GID} | cut -d: -f1); \
    else \
        GROUP_NAME=${USERNAME}; \
        groupadd --gid ${GID} ${GROUP_NAME}; \
    fi && \
    useradd --uid ${UID} --gid ${GID} --create-home --shell /bin/bash ${USERNAME}

WORKDIR /home/${USERNAME}
RUN chown -R ${UID}:${GID} /home/${USERNAME}

USER ${USERNAME}
CMD ["/bin/bash"]
```

## Docker image with UV

Following the previous pattern you may not be surprised about this one

```bash
FROM ubuntu:latest

ARG USERNAME=user
ARG UID=1000
ARG GID=1000

RUN set -x \
    && echo "Updating..." \
    && apt-get upgrade \
    && apt-get update \
    && echo "Installing Packages..." \
    && apt-get install -y \
    wget \
    curl

RUN if getent group ${GID} >/dev/null; then \
        echo "Group with GID ${GID} already exists, using it."; \
        GROUP_NAME=$(getent group ${GID} | cut -d: -f1); \
    else \
        GROUP_NAME=${USERNAME}; \
        groupadd --gid ${GID} ${GROUP_NAME}; \
    fi && \
    useradd --uid ${UID} --gid ${GID} --create-home --shell /bin/bash ${USERNAME}

WORKDIR /home/${USERNAME}
RUN chown -R ${UID}:${GID} /home/${USERNAME}

USER ${USERNAME}

# install UV on user
RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --verbose 

CMD ["/bin/bash"]
```

Here instead of installing `uv` on root we install it on the user (goes after the statement `USER`).

## Building and running the images

As usual I have a quick recipe to build these images, a bash file (I will name it as `build-run.sh`) that can be used for the different builds

```bash
#!/bin/bash

THIS_DIR=$(dirname "$(realpath "$0")")

# # Dockerfile name
# DOCKERFILE=python-3.12
# DOCKERFILE=python-3.12-slim
# DOCKERFILE=python-3.12-build
# DOCKERFILE=mamba
DOCKERFILE=uv


build_image(){
    docker build -f Docker/Dockerfile-${DOCKERFILE} \
                --build-arg USERNAME=$(whoami) \
                --build-arg UID=$(id -u) \
                --build-arg GID=$(id -g) \
                 -t ${DOCKERFILE}-image .
}

run_image(){
    docker run \
    -v ${THIS_DIR}:/home/$(whoami) \
    -it \
    ${DOCKERFILE}-image \
     /bin/bash
}

croak(){
    echo "[ERROR] $*" > /dev/stderr
    exit 1
}

main(){
    if [[ -z "$TASK" ]]; then
        croak "No TASK specified."
    fi
    echo "[INFO] running $TASK $*"
    $TASK "$@"
}

main "$@"
```
simple uncomment the dockerfile you want to build (here we uncomment `uv`) and run the following commands to build an ssh

```bash
export TASK=build_image
./build-run.sh

export TASK=run_image
./build-run.sh
```

Place the dockerfiles under `Docker` directory as we show in the repository.