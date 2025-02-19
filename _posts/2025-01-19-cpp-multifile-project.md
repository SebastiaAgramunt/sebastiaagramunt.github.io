---
title: C++ Multiple file project
author: sebastia
date: 2025-01-19 7:05:00 +0800
categories: [C++]
tags: [computer science]
pin: true
toc: true
render_with_liquid: false
math: true
---


Normally when dealing with C++ projects, developers structure their code in source files and headers. Here I want to show how to create a simple project without external dependencies. The entire example can be found in my github repository [blogging-code](https://github.com/SebastiaAgramunt/blogging-code).

## File structure

Let's design a project with two modules, think of it as if module1 was a physics module and module2 was some sort of interfacing file with that module. It is a good design pattern to isolate sub-projects inside the same project. The file strcucture I propose for the example is (when I run `tree` in my project root directory):

```bash
.
├── include
│   ├── module1
│   │   ├── module1c1.hpp
│   │   └── module1c2.hpp
│   └── module2
│       ├── module2c1.hpp
│       └── module2c2.hpp
├── scripts
│   └── compile.sh
└── src
    ├── main.cpp
    ├── module1
    │   ├── module1c1.cpp
    │   └── module1c2.cpp
    └── module2
        ├── module2c1.cpp
        └── module2c2.cpp
```

In each cpp file we should place (in `mod1c1.cpp`):

```cpp
#include <iostream>

#include <module1/module1c1.hpp>

void mod1c1::foo(){
    std::cout << "mod1c1\n";
}
```

where we substitute `module1/mod1c1.hpp` and `mod1c1` depending on the filename. If for example we are working in file `mod2c1` we would substitue by `module2/mod2c1` and `mod2c1`. They are the same files but chainging the name.

Also the header files should be something like

```cpp
#ifndef INCLUDE_MOD1C1_HPP
#define INCLUDE_MOD1C1_HPP

#include <iostream>

class mod1c1{
public:
   void foo();
};

#endif

```

As before, for this example substitute the numbers of `mod1c1` to the corresponding ones according to the filename.

Finally we add a `main.cpp` file that will contain the `main` function (entrypoing) to generate an executable that uses all the modules and functions in the `src`. The contents of the file `main.cpp` are

```cpp
#include "module1/module1c1.hpp"
#include "module1/module1c2.hpp"
#include "module2/module2c1.hpp"
#include "module2/module2c2.hpp"

int main(){
    mod1c1 m1c1; m1c1.foo();
    mod1c1 m1c2; m1c2.foo();
    mod2c1 m2c1; m2c1.foo();
    mod2c2 m2c2; m2c2.foo();
}
```

## Compilation and linking for executable

It is time to compile the source files to the objects, first create a directory `build` where we are going to store all the builds, commonly we also create subdirectories for `obj` to store the objects, `bin` to store the binaries and `lib` for the libraries

```bash
# recreate build
rm -rf build
mkdir build

# create subdirectories
mkdir build/obj build/bin build/lib
```

Now compile the source files to objects and place them in the subfolder `obj`:

```bash
# compile all sources
g++ -std=c++17 -Iinclude -c src/module1/module1c1.cpp -o build/obj/moudle1c1.o
g++ -std=c++17 -Iinclude -c src/module1/module1c2.cpp -o build/obj/moudle1c2.o
g++ -std=c++17 -Iinclude -c src/module2/module2c1.cpp -o build/obj/moudle2c1.o
g++ -std=c++17 -Iinclude -c src/module2/module2c2.cpp -o build/obj/moudle2c2.o

# compile main
g++ -std=c++17 -Iinclude -c src/main.cpp -o build/obj/main.o
```

We are choosing the C++ standard with the flag `-std` (check the versions in [cpprefenence.com](https://en.cppreference.com/w/cpp)). The include directory is "included" with the flag `-I` and according to the project structure is `include`. Finally the flag `-c` is to tell the compiler to compile the source into an object file. For faster compilation use additional flags like `-O3` (aggressive optimization, you can do `O2` or `O1` for less optimization),  `-march=native` (optimizing for your specific CPU), `-flto` (link time optimization for cross file optimizations) and  `-ffast-math` (aggresive floating point optimizations).

After the sources have been compiled and we checked that there is no error the next step is to link the objects. The linker will check all the definitions of the objects and generate a `main` executable that when run will start with the `int main()` function. The function `main()` could be in any of the object files but the important is that when liking several objects there should be only one `main` function to generate an executable. Let's link all the objects with the command

```bash
# link all the objects
g++ build/obj/moudle1c1.o \
    build/obj/moudle1c2.o \
    build/obj/moudle2c1.o \
    build/obj/moudle2c2.o \
    build/obj/main.o \
    -o build/bin/main
```

See that we placed the executable in `build/bin/main`, execute the binary as

```bash
./build/bin/main
```

For convenience and make things faster I like to create scripts. Place this bash script in `scripts/compile.sh` with the content:

```bash
THIS_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(dirname ${THIS_DIR})

recreate_dirs(){
    # removing build directory
    echo "Removing ${ROOT_DIR}/build and recreating..."
    rm -rf ${ROOT_DIR}/build
    mkdir ${ROOT_DIR}/build

    # creating directories for the build
    mkdir ${ROOT_DIR}/build/obj
    mkdir ${ROOT_DIR}/build/bin
    mkdir ${ROOT_DIR}/build/lib
}

compile_exec(){
    recreate_dirs

    # compile to objects
    echo "Compiling objects for executable..."
    g++ -std=c++17 -Iinclude -c src/module1/module1c1.cpp -o ${ROOT_DIR}/build/obj/moudle1c1.o
    g++ -std=c++17 -Iinclude -c src/module1/module1c2.cpp -o ${ROOT_DIR}/build/obj/moudle1c2.o
    g++ -std=c++17 -Iinclude -c src/module2/module2c1.cpp -o ${ROOT_DIR}/build/obj/moudle2c1.o
    g++ -std=c++17 -Iinclude -c src/module2/module2c2.cpp -o ${ROOT_DIR}/build/obj/moudle2c2.o

    # compile the main to object
    g++ -std=c++17 -Iinclude -c src/main.cpp -o ${ROOT_DIR}/build/obj/main.o

    # link all the objects
    g++ ${ROOT_DIR}/build/obj/moudle1c1.o \
        ${ROOT_DIR}/build/obj/moudle1c2.o \
        ${ROOT_DIR}/build/obj/moudle2c1.o \
        ${ROOT_DIR}/build/obj/moudle2c2.o \
        ${ROOT_DIR}/build/obj/main.o \
        -o ${ROOT_DIR}/build/bin/main
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

To execute it you just need to run

```bash
# make the script executable
chmod +x scripts/compile.sh

export TASK=compile_exec
./scripts/compile.sh
```

The new generated executable will be found in `build/bin/main`, execute it as:

```bash
./build/bin/main
```

To get the prints of each function on your screen. 
## Remarks

What we have shown here is the bare bones of a C++ project. We created an executable that runs several functions defined in different files. 

This is the basics, in follow up posts we will learn how to compile code using [Makefile](https://www.gnu.org/software/make/manual/make.html) and [cmake](https://cmake.org/), better and more convenient tools to compile code. We will also learn to create static and dynamic libraries of our code to be used in other projects and also how to compile and link external libraries like [OpenCV](https://opencv.org/), [Boost](https://www.boost.org/), [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) or [cBLAS](https://www.gnu.org/software/gsl/doc/html/cblas.html). 

