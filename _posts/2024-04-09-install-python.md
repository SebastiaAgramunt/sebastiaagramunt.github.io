---
title: Install Python
author: sebastia
date: 2024-04-21 10:10:00 +0800
categories: [Python]
tags: [computer science]
pin: true
toc: true
render_with_liquid: false
math: true
---

Python is a very popular scripting language, according to [StackOverflow 2023 survey](https://survey.stackoverflow.co/2023/) it is the third most commonly used language and the fourth in the ranking for professional developers. All in all, it is a very simple language to learn and thus is one of the first languages new developers incorporate in their projects. In this post we will see how to install Python in your system.

<!-- 
## Why Python?

Python is 

* Easy to learn: Python's simple and readable syntax makes it easy to learn for beginners.
* Easy to use: Python's minimalist approach allows developers to focus on problem-solving rather than complex syntax.
* Extensive standard library: Python's standard library provides a wide range of modules and packages that can be readily used in projects, reducing development time.
* Extensive open source library: The comunity has built and maintained projects for web, backend, scientific computing, data science, computer science... so that you don't need to build yourself.
* Large comunity and suport: Many users mean there is probably someone else with the same coding problem you are facing.
* Integration with other languages: Python can be easily integrated with other languages like C++ for better performance.
* Cross platform: Easy to run and deploy in different operating systems and computer architectures.

The huge standard library and packages and the ease for fast prototiping makes Python a great programming language in many jobs, specially for data sicence and machine learning. However, Python is not the best for every kind of task you may encounter in your programming job, the main cons are:

* Performance: It can be very slow compared to compiled languages like C++, as much as 10X slower.
* Higher memory consumption compared to other languages like C++ or Go
* Runtime Errors: Since python is interpreted, the only errors we will get are at runtime. In compiled languages we can catch many errors at compile-link time. -->

## TLDR

Tutorial to install python from installable in MacOS. 

* Download [python-3.12.3-macos11.pkg](https://www.python.org/ftp/python/3.12.3/python-3.12.3-macos11.pkg) (or choose your version, as of today latest is `3.12`). 
* Install using the installer helper that will place all the python files in `/Library/Frameworks/Python.framework/Versions/3.12/`.
* Create an alias, execute
```bash
echo alias pytyon=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 >> ~/.zshrc
```
* Run python by executing `python` in termial.

## Install Python with installable

There are many ways to install python in your computer, the most simple one is from [python.org](https://www.python.org/). Go to downloads and choose your operating system / architecture and version from the stable releases options.

As an example we will install Python `3.12.3` which is the most recent stable one at the time of writing this post. These version numbers are called major, minor and revision, in the example major is 3, minor is 12 and revision is 3. It is important to choose the right version of python as some software projects may have old functionality and may only work in an older python version.

For MacOS, first download the installer [python-3.12.3-macos11.pkg](https://www.python.org/ftp/python/3.12.3/python-3.12.3-macos11.pkg) (choose the appropiate installer for your OS) and proceed with the installation. Then, open a terminal and type `python3` or `python3.12` to start the python command prompt:

```bash
Python 3.12.3 (v3.12.3:f6650f9ad7, Apr  9 2024, 08:18:48) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

That's it, you are in python!. See that in the prompt it is specified the version `3.12.3` , the date it was released and the compiler (Clang in my case). Just to test type

```python
Python 3.12.3 (v3.12.3:f6650f9ad7, Apr  9 2024, 08:18:48) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> print("Hi there")
Hi there
```
That's it, you have successfully installed Python!.

## Install location

In MacOS python is installed in `/Library/Frameworks/Python.framework/Versions/3.12/` (for Windows it would be `C:\Users\<Username>\AppData\Local\Programs\Python\Python3.12`). If you installed any other python version, say `3.11.2`, just change the path to the corresponding major and minor versions. Let's explore what is in there, the main directories are:

<!-- * `bin` : Where the binaries (executables) are stored, this is python and pip among others.
* `include/python3.12`: All header files for the compiled libraries. For instance, the file `floatobject.h ` contains the C++ definition of the float object in Python. The headers in this directory are needed if we want to compile a C++ extension to be used in Python (a Python wrapper for a C++ library).
* `lib`: contains compiled libraries. Concretely it contains `libpython3.12.a` or `libpython3.12.so` wich is the (static or dynamic) compiled standard library for python. This directory is added automatically to `sys.path` . Also custom compiled packages are copied here.
* `share`: Includes miscellaneous files such as documentation, configuration files, and examples. It might also contain man pages, sample scripts, and other shared assets that don’t fit into the binary, header, or library categories.
 -->

| Directory          | Usage                                                                                                                                                                                                                                                                                     |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| bin                | Where the binaries (executables) are stored, this is python and pip among others.                                                                                                                                                                                                         |
| include/python3.12 | All header files for the compiled libraries. For instance, the file `floatobject.h ` contains the C++ definition of the float object in Python. The headers in this directory are needed if we want to compile a C++ extension to be used in Python (a Python wrapper for a C++ library). |
| lib                | contains compiled libraries. Concretely it contains `libpython3.12.a` or  `libpython3.12.so` wich is the (static or dynamic) compiled standard library for python. This directory is added automatically to `sys.path`. Also custom compiled packages are copied here.                    |
| share              | Includes miscellaneous files such as documentation, configuration files, and examples. It might also contain man pages, sample scripts, and other shared assets that don’t fit into the binary, header, or library categories.                                                            |

The python executable you run when typing `python3` in your terminal is `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3` (just type `which python3.12` or `which python3`).

The commands `python3.12` and `python3` are recognized because they are executables found in a directory from your `PATH` variable. To find these directories just `tr ':' '\n' <<< "$PATH"`, for MacOS you will see the path `/Library/Frameworks/Python.framework/Versions/3.12/bin`, in this directory is where `python3.12` executable lives, and also a simulated link to it `python3`.


## Install another python version

Let's say you need to use Python `3.11.7` for another project but you arleady have `3.12.3`, what do you do?. Uninstall and reinstall?. Luckily you don't need to do that. As we did previously, just download the installable [python-3.11.7-macos11.pkg](https://www.python.org/ftp/python/3.11.7/python-3.11.7-macos11.pkg) and execute it. Open a new terminal and type `python3 --version`, you will see that now instead of having `3.12.2` (as installed previously), it will show `3.11.7`. What just happend?. Run `tr ':' '\n' <<< "$PATH" | grep Python`, you will see two paths, first `/Library/Frameworks/Python.framework/Versions/3.11/bin` and second `/Library/Frameworks/Python.framework/Versions/3.12/bin`. So by installing a second version we get the new path first. When bash looks for an executable it goes sequentially from the first path to the last, if there are two `python3` executables in the `PATH`, bash will just pick the first, in this case `3.11`. But no panic, you can still run any python version installed, check it doing

```bash
python3.11 --version
python3.12 --version
```
so to run a python script `script.py` in python 3.12 just do `python3.12 script.py`.

We can still do another "hack" if you want to change `pythhon3` be either of the two installed versions, just add the python exec path as the first entry of your `PATH`:

```bash
# to run python3 and execute python3.12
export PATH="/Library/Frameworks/Python.framework/Versions/3.12/bin:${PATH}"

# the following will run script.py using python3.12
python3 script.py 


# to run python3 and execute python3.11
export PATH="/Library/Frameworks/Python.framework/Versions/3.11/bin:${PATH}"

# the following will run script.py using python3.11
python3 script.py 
```

## Make an alias to python executable

To pin a specific python version and run that whenever you type `python` just create an alias. For that open `.bashrc` or `.zshrc` depending if you use bash or zsh shell (look for the equivalent if you have another shell) and append `alias pytyon=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`. To append just open a terminal and do the following (assuming you use zsh):

```bash
# append the command to the file ~/.zshrc
echo alias pytyon=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 >> ~/.zshrc

# source ~/.zshrc to make the changes effective
source ~/.zshrc
```

## Uninstall

Get the list of your python installed

```bash
ls -lhat /Library/Frameworks/Python.framework/Versions/  
```

Remove specific versions, in this case we will remove `3.11` and `3.12`
```bash
sudo rm -rf /Library/Frameworks/Python.framework/Versions/3.11
sudo rm -rf /Library/Frameworks/Python.framework/Versions/3.12
```

or remove entire python

```bash
sudo rm -rf /Library/Frameworks/Python.framework/
```

Then remove the linked binaries

```bash
sudo rm /usr/local/bin/python3 /usr/local/bin/pip3
```

Then since you have probably modified the `$PATH`, just go and remove the specific exports and aliases in `~/.zshrc` or `~/.bashrc`. Also probably the installation has modified `~/.zprofile` or `~/.bash_profile` adding the paths `/Library/Frameworks/Python.framework...`. Comment or remove those lines. 

Finally remove the alias we created before in `~/.zshrc` (or `~/.bashrc`). The line `alias python=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`.

and restart the terminal by typing

```bash
reset
```

You are all set. Python is deleted from your system


<!-- 

## Install Python from source

There is the possibility of installing Python in a custom directory. Let's say we wish to have several python versions in the directory `~/.python_versions`, let's install Python `3.12.3` there:

```bash

# set a convenient variable for the version
PYTHON_VERSION=3.12.3

# create an installation directory
mkdir -p ~/.python_versions/${PYTHON_VERSION}

# download python tar.gz file
wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz

# untar and drive to the new directory
tar -xzf Python-${PYTHON_VERSION}.tgz
cd Python-${PYTHON_VERSION}

# configure to install in the desired directory
./configure --prefix=${HOME}/.python_versions/${PYTHON_VERSION} --enable-optimizations

# compile the code in 8 parallel processes (to run faster)
make -j8

# install
sudo make altinstall
```

Now you got python installed at the specified directory. To open a prompt just run:

```bash
~/.python_versions/3.12.3/bin/python3.12
```

Unfortunatelly the directory `~/.python_versions/3.12.3/bin/` is not in the `PATH` we can create a symlink for the exec

```bash
ln -s ${HOME}/.python_versions/3.12.3/bin/python3.12 /usr/local/bin/python3.12
```

Then, whenever you type `python3.12` on your terminal, it will direct you to that python installation. -->
<!-- 
## Versions

As you can imagine Python evolves over time and there are things that change, for instance in Python 3.8 it was introduced the walrus operator (`:=`), prior versions of python do not have this operaror so if you write a program having a single line with the walrus operator and use  aversion of python `<3.8` the program will fail at execution time. To keep track of all changes in python you can refer to the [Python Enhancement Program (a.k.a PEP)](https://peps.python.org/).

Different projects need for different python versions so in the same system we may want to have Python `3.12`, `3.11`, `3.7`... Can we have all these at the same time in the system?. Yes, just install new versions following the instructions on the previous section changing the version `PYTHON_VERSION` to the desired one. However this is cumbersome and not easy to maintain. In a following post I will show how to manage python versions using [pyenv](https://github.com/pyenv/pyenv).

## Structure of installation

Python is not only an executable, let's inspect the directories of the installation. If we take the custom installation the directory we created would be `${HOME}/.python_versions/3.12.3/`:

```bash
tree ${HOME}/.python_versions/3.12.3/ -L 1
```
We see four directories:  -->


<!-- 


to print "Hi There" in terminal. If you want to use `python` instead of `python3`, just create an alias. 

```bash
# if your shell is zsh
echo "alias python='python3'" >> ~/.zshrc
source ~/.zshrc

# if your shell is bash
echo "alias python='python3'" >> ~/.bashrc
source ~/.bashrc
``` -->