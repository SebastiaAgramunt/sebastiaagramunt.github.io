---
title: C++ Compile Libraries
author: sebastia
date: 2025-01-19 8:23:00 +0800
categories: [C++]
tags: [computer science]
pin: true
toc: true
render_with_liquid: false
math: true
---

In C++ we constantly deal with libraries that are compiled in our system like the standard libraries or other libraries such as  [OpenCV](https://opencv.org/) (for computer vision) [Boost](https://www.boost.org/) (for linear algebra, pseudorandom number generation...). In this post we will learn about shared and static libraries, how to compile them and how to link them to your programs.

All the code in this post can be found in (cpp-compile-library)[https://agramunt.me/posts/cpp-compile-library/] supporting material of my [blogging-code](https://github.com/SebastiaAgramunt/blogging-code) github repository.

## File structure

For this demonstration we will generate a file structure like the following

```bash
.
├── include
│   └── matmul.h
├── scripts
│   └── compile.sh
└── src
    ├── main.cpp
    └── matmul.cpp
```

The `mathmul` file will contain a routine to multiply two matrices. In my example I define `matmul.h` with the contents

```cpp
#ifndef MATMUL_H
#define MATMUL_H

void matmul(const int* A, const int* B, int* C, int M, int N, int K);
void printmatrix(const int* A, int M, int N);

#endif
```

and `matmul.cpp` with

```cpp
#include <iostream>
#include "matmul.h"

// Matrices are indexed row-major in this example. E.g. if A is [M x N]
// If i,j are the row and column indices, the element A[i, j] is
// A[i, j] = A[i * N + j] // if row-index
// A[i, j] = A[j * M + i] // if column-index

void matmul(const int* A, const int* B, int* C, int M, int N, int K){
// Matrix multiplication, C[M x K] = A[M x N] * B[N x K]
// Multiplication is $\sum_n A[m, n] * B[n, k]$
    for(int m=0; m<M; m++){
        for(int k=0; k<K; k++){
            C[m * K + k] = 0;
            for(int n=0; n<N; n++){
                C[m * K + k] += A[m * N + n] * B[n * K + k];
            }
        }
    }
}

void printmatrix(const int* A, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << A[i * N + j] << " ";
        }
        std::cout << "\n";
    }
}
```

The plan is to compile `matmul.cpp` as a library and then use it in `main.cpp`. The latter file should contain something like

```cpp
#include <iostream>
#include "matmul.h"

int main(void){

    // A[M x N]
    int M = 2;
    int N = 3; 
    int* A = new int[M * N];

    for(int i=0; i < M * N; i++){
        A[i] = i;
    }
    std::cout << std::endl << "A:" << std::endl;
    printmatrix(A, M, N);

    // B[N x K]
    int K = 4;
    int* B = new int[N * K];

    for(int i=0; i < N * K; i++){
        B[i] = i;
    }
    std::cout << std::endl << "B:" << std::endl;
    printmatrix(B, N, K);

    // C[M x K]
    int* C = new int[M * K];
    matmul(A, B, C, M, N, K);

    std::cout << std::endl << "C = A x B: " << std::endl;
    printmatrix(C, M, K);

    return 0;
}
```

Where we define two matrices A (size 2x3) and B (size 3x4) that are filled with numbers from 0 to the maximum index of each matrix. We use the routine `matmul` to calculate a matrix C that results from the multiplication of A times B.

In the next sections we will compile all, compile with shared library and compile with static library and explain the difference between static and shared libraries

## Compile all the code

First let's compile all the code and link it as we did in the previous post. Run the following to create the file structure for the compiled objects:

```bash
rm -rf build
mkdir build

# creating directories for the build
mkdir build/obj
mkdir build/bin
mkdir build/lib
```

Then start compiling the files

```cpp
g++ -std=c++17 -Iinclude -c src/matmul.cpp -o build/obj/matmul.o
g++ -std=c++17 -Iinclude -c src/main.cpp -o build/obj/main.o
```

The `-I` flag precedes the path where to find the includes. The `-std` indicates the version for the standard library used.

Link to the final executable

```bash
g++ build/obj/matmul.o \
	build/obj/main.o \
    -o build/bin/main
```

And execute the compiled main to see the result

```bash
./build/bin/main
```

This will print out the expected result

```
A:
0 1 2 
3 4 5 

B:
0 1 2 3 
4 5 6 7 
8 9 10 11 

C = A x B: 
20 23 26 29 
56 68 80 92
```

## Shared library

A compiled shared library is a library that is loaded once in the computer and shared by different processes. Those processes can only read the code and not modify it, and execute in their own threads. This saves global processing memory as the library is stored once in "shared" for all processes. This makes the executable smaller as the library is not included in it but makes it more complex to run as we need to have the library saved somewhere then tell the linker where to find it. 

Let's compile the shared library. First, recreate the build directory

```bash
rm -rf build
mkdir build

# creating directories for the build
mkdir build/obj
mkdir build/bin
mkdir build/lib
```

Now compile the library `matmul` with the command

```bash
g++ -std=c++17 -Iinclude -c src/matmul.cpp -o build/obj/matmul.o
g++ -std=c++17 -shared -fPIC -Iinclude build/obj/matmul.o -o build/lib/libmatmul.so

// or in one step
// g++ -std=c++17 -shared -fPIC -Iinclude src/matmul.cpp -o build/lib/libmatmul.so
```

When compiling source code into a shared library using the `-shared` flag, the `-fPIC` flag is often required. This ensures that the resulting shared library is position-independent in memory (can be loaded regardless of the memory address managed by the OS). 

Now compile the `main.cpp` to an object `main.o`, this is the code that will use the shared library.

```bash
g++ -std=c++17 -Iinclude -c src/main.cpp -o build/obj/main.o
```

Finally link the compiled `main.o` with the library to generate the executable.

```bash
g++ build/obj/main.o \
    -Iinclude \
    -L./build/lib \
    -lmatmul \
    -Wl,-rpath,./build/lib \
    -o build/bin/main_dynamic
```

the `-L` flag indicates the directory where the library `libmatmul.so` is located. The `-l` is the flag that tells the linker which library to link, in our case `matmul`. Recall that the file is `libmatmul.so` and in the `-l` flag we don't include this "lib", it would fail to find the library otherwise. The `-Wl` option is used to pass options directly to the linker. For instance`-Wl,-rpath,./build/lib` tells the linker to set the runtime search path for shared libraries, so the executable can find shared libraries at runtime. The `-o` is to specify the output path and filename.

Once the executable is linked test it by running

```bash
./build/bin/main_dynamic
```

And you should get the same output as in the previous section. One of the consequences of using a shared library is that once linked we can't change the path of the compiled library (unless we also change the runtime path of the executable running `chrpath`). Let me explain this with an example: Move the file `build/lib/libmatmul.so` somewhere else and try to run the executable again, it won't run and will raise an error because it won't be able to find the shared library. The `-rpath` tells the executable where to find the library and is encoded in the executable after linking. As we will see, this won't happen in the static library


## Static Library

As opposed to a dynamic library the static library is included in the final executable and we don't need to specify a path at runtime. It uses more memory as each process has its own copied instructions, the executable is larger since it includes all the library code but as a bright side the executable is self contained as we mentioned. Let's compile the example.

Compile the library

```bash
g++ -std=c++17 -Iinclude -c src/matmul.cpp -o build/obj/matmul.o
ar rcs build/lib/libmatmul.a build/obj/matmul.o
```

The second command `ar` is simply the archiver, a command that combines several files into one (like zip but without compressing by default), the `rcs` tells the archiver to insert files, create the archive and create an index.

Next is to create the object of the main as usual

```bash
g++ -std=c++17 -Iinclude -c src/main.cpp -o build/obj/main.o
```

And the last step is to create the executable

```bash
g++ build/obj/main.o -o build/bin/main_static -L./build/lib -lmatmul
```

If you check, it is a very similar command compared to the one we used to generate the shared library.
## A bash script to compile everything

As usual, I have developed a bash script that can be useful to run all the tasks, compile the executable all from scratch, compile it using static library and a dynamic library.

```bash
#!/bin/bash

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
    g++ -std=c++17 -I${ROOT_DIR}/include -c ${ROOT_DIR}/src/matmul.cpp -o ${ROOT_DIR}/build/obj/matmul.o
    g++ -std=c++17 -I${ROOT_DIR}/include -c ${ROOT_DIR}/src/main.cpp -o ${ROOT_DIR}/build/obj/main.o

    # link all the objects
    g++ ${ROOT_DIR}/build/obj/matmul.o \
        ${ROOT_DIR}/build/obj/main.o \
        -o ${ROOT_DIR}/build/bin/main
}

compile_static(){
    recreate_dirs
    echo "Compiling objects for executable using static library..."

    # compile shared library
    g++ -std=c++17 -I${ROOT_DIR}/include -c ${ROOT_DIR}/src/matmul.cpp -o build/obj/matmul.o
    ar rcs ${ROOT_DIR}/build/lib/libmatmul.a ${ROOT_DIR}/build/obj/matmul.o

    # compile main object
    g++ -std=c++17 -I${ROOT_DIR}/include -c ${ROOT_DIR}/src/main.cpp -o ${ROOT_DIR}/build/obj/main.o

    # link
    g++ ${ROOT_DIR}/build/obj/main.o -o ${ROOT_DIR}/build/bin/main_static -L${ROOT_DIR}/build/lib -lmatmul
}


compile_dynamic(){
    recreate_dirs
    g++ -std=c++17 -Iinclude -c ${ROOT_DIR}/src/matmul.cpp -o ${ROOT_DIR}/build/obj/matmul.o
    g++ -std=c++17 -shared -fPIC -Iinclude ${ROOT_DIR}/build/obj/matmul.o -o ${ROOT_DIR}/build/lib/libmatmul.so

    g++ -std=c++17 -Iinclude -c ${ROOT_DIR}/src/main.cpp -o ${ROOT_DIR}/build/obj/main.o

    g++ ${ROOT_DIR}/build/obj/main.o \
    -I${ROOT_DIR}/include \
    -L${ROOT_DIR}/build/lib \
    -lmatmul \
    -Wl,-rpath,${ROOT_DIR}/build/lib \
    -o ${ROOT_DIR}/build/bin/main_dynamic
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

Simply save it in `scripts/compile.sh` file, make it executable `chmod +x scripts/compile.sh` and run with

```bash
export TASK=compile_exec
./scripts/compile.sh

export TASK=compile_dynamic
./scripts/compile.sh

export TASK=compile_static
./scripts/compile.sh
```

Bear in mind that we recreate the build directory in every execution. As mentioned previously, this is not the best way to compile a project, we normally use cmake or make. The bash script helps to understand the real bash commands used before we make things more complex with cmake.
## Comparison of dynamic libraries vs static libraries

Here is a summary of the comparison that we have already explained in the previous sections. Just a cheatsheet for the future

| Feature                        | Shared Library                    | Static Library                  |
|--------------------------------|------------------------------------|---------------------------------|
| **Memory Usage**               | Lower (shared across applications)| Higher (duplicated in each app)|
| **Executable Size**            | Smaller                           | Larger                          |
| **Deployment Simplicity**      | Requires library installation     | Self-contained executable       |
| **Update Flexibility**         | Can update library independently  | Requires app recompilation      |
| **Startup Performance**        | Potentially slower (dynamic linking)| Faster (prelinked)              |
| **Compatibility Concerns**     | Dependency on library versions    | None                            |

One more thing to check is the size of the executable for this example. The executable `main_dynamic` is 39952 bytes whereas the `main_static` is 40248, a difference of 296 bytes being larger the static executable (as it contains all the `matmul` library in the excutable itself
