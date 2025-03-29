---
title: Basic C++ python extension
author: sebastia
date: 2025-03-08 22:28:00 +0800
categories: [C++, Python]
tags: [computer science]
pin: true
toc: true
render_with_liquid: false
math: true
---


C++ and Python are very different programming languages, the first one is compiled and low level whereas the second one is interpreted. C++ is a lot faster than Python but, can we leverage the performance of C++ and the versatility in Python?. Yes, we can do such thing writing C++ extensions and create bindings for Python. In this post we will create a python package with compiled code using [pybind11](https://github.com/pybind/pybind11) library to create the python bindings. As usual you have the [blog](https://github.com/SebastiaAgramunt/blogging-code) with the [code](https://github.com/SebastiaAgramunt/blogging-code/tree/main/cpp-basic-cpp-python-extension).


## The C++ project

In a repository we will need coexisting python code and C++ code. In this example we will code a C++ matrix multiplication that we want to expose to Python. We define the following file structure

```bash
.
├── README.md
├── include
│   └── matmul.h
├── scripts
│   └── compile.sh
├── src
│   ├── bindings.cpp
│   ├── main.cpp
│   └── matmul.cpp
└── tests
    └── test_matmul.py
```

With the following contents for `matmul.h`:

```cpp
#ifndef MATMUL_H
#define MATMUL_H

void matmul(const float* A, const float* B, float* C, int M, int N, int K);
void printmatrix(const float* A, int M, int N);

#endif
```

and `matmul.cpp`

```cpp
#include <iostream>
#include "matmul.h"

// Matrices are indexed row-major in this example. E.g. if A is [M x N]
// If i,j are the row and column indices, the element A[i, j] is
// A[i, j] = A[i * N + j] // if row-major
// A[i, j] = A[j * M + i] // if column-major

void matmul(const float* A, const float* B, float* C, int M, int N, int K){
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

void printmatrix(const float* A, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << A[i * N + j] << " ";
        }
        std::cout << "\n";
    }
}
```
This file contains just two fucntions, `matmul` gets three matrices `A[M x K]`, `B[M x N]` and `C[N x K]` and returns the multiplication of $C = A \times B$. The matrices are single precision array of floats and we consider [row-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order).

We can use the `matmul` library in a main fucntion to compile a binary and test that our function is correct. For that we define a `main.cpp`:

```cpp
#include <iostream>
#include "matmul.h"

#define M 32
#define N 64
#define K 32

void initializeMatrices(float* A, float* B) {
    srand(7);

    for (int i = 0; i < M * N; ++i)
        A[i] = (rand() % 100) / 10.0f;  // Random float in range [0,10]

    for (int i = 0; i < N * K; ++i)
        B[i] = (rand() % 100) / 10.0f;  // Random float in range [0,10]
}

int main() {

    float* A = new float[M * N];
    float* B = new float[N * K];
    float* C = new float[M * K];

    initializeMatrices(A, B);
    matmul(A, B, C, M, N, K);

    std::cout << "C = A x B:" << std::endl;
    printmatrix(C, M, K);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
```
This is pretty simple code, we define the matrices, randomly initialize them (although we don't need C to be initialized randomly) and we perform the multiplication. Then we print out the results on screen. Let's compile this `main.cpp` entrypoint. First we manually create our usual build directory, then we compile the objects and lastly we link the objects into the final executable.


```bash
# create build directories
rm -rf build
mkdir -p build/obj
mkdir build/bin
mkdir build/lib

# compile to objects
g++ -std=c++17 -Iinclude -c src/matmul.cpp -o build/obj/matmul.o
g++ -std=c++17 -Iinclude -c src/main.cpp -o build/obj/main.o

# link all the objects
g++ build/obj/matmul.o \
    build/obj/main.o \
    -o build/bin/main
```

With this we can execute the `main` and see the result of the multiplication of the two matrices

```bash
./build/bin/main
```

## Understanding what are Python bindings

Now the question is, how do we convert this code so that we can run it with python?. I would like to use `matmul` function from python. We need to understand first that python is actually a collection of shared libraries that are loaded dynamically. Just create a new environment and let's inspect it

```bash
python -m venv .venv
source .venv/bin/activate
```

First with the tool `otool` in MacOs (`ldd` in Linux) let's see what are the libraries that the executable `python` depends on, type

```bash
otool -L .venv/bin/python
```

to see that 

```
.venv/bin/python:
	/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation (compatibility version 150.0.0, current version 2420.0.0)
	/Users/sebas/.pyenv/versions/3.12.4/lib/libpython3.12.dylib (compatibility version 3.12.0, current version 3.12.0)
	/usr/local/opt/gettext/lib/libintl.8.dylib (compatibility version 13.0.0, current version 13.0.0)
	/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1345.100.2)
```

These are the libraries that `python` binary expect to load at runtime. The most important is `libpython3.12.dylib` (that's for MacOS, you would see a `.so` file in Linux or a `.dll` file in Windows). It contains the compiled core of the Python interpreter, including the bytecode evaluator, built-in types, and other core runtime components. This library is used to embed Python into other applications or link with C/C++ extensions dynamically. Let's continue inspecting the paths that python uses. At runtime the interpreter includes a bunch of directories to look for libraries. Find them by running

```bash
python -c "import sys; print(sys.path)"
```
The output is:

```
['',
'/Users/sebas/.pyenv/versions/3.12.4/lib/python3.12',
'/Users/sebas/.pyenv/versions/3.12.4/lib/python3.12/lib-dynload',
'/Users/sebas/tmp/blogging-code/cpp-compile-link-external-lib/.venv/lib/python3.12/site-packages']
```

The first directory containes the file `libpython3.12.dylib`. If we go deeper into that directory there's `/Users/sebas/.pyenv/versions/3.12.4/lib/python3.12/lib-dynload` where you will find python files of really known packages (the standard library of python), that's `hashlib.py`, `datetime.py`, `dataclases.py`, `abc.py`... those are "packages" that come by default with the python installation. 

Let's take a look at the file `hashlib.py` file, some of the imports are `import _sha1`, `import _md5`, those are cryptographic algoritms. Where are those imports?. If you check the next path `/Users/sebas/.pyenv/versions/3.12.4/lib/python3.12/lib-dynload` there are files like `_sha1.cpython-312-darwin.so` and `_md5.cpython-312-darwin.so`. Libraries that are imported as modules, C++ shared libraries that can be loaded by the python interpreter. That's what we want to do, compile the C++ `matmul` funcion into some sort of shared library so that we can import in our python script. 


## Compiling a shared library for Python

In a previous C++ post I have shown how to compile a shared object, and this should be easy. However we cannot expect to compile C++ code directly to get a python shared object, we need to define how the C++ code translates into C++ python objects. For this we need the python C++ headers, to use the C++ python API. You can see the path for those by executing

```bash
python -c "import sysconfig; print(sysconfig.get_path('include'))"
```

which in my case is `/Users/sebas/.pyenv/versions/3.12.4/include/python3.12`, there you can find many headers but the most important is `Python.h` (which is basically all the other headers combined). 

Apart from the header you need the library `python3.12`, you can get it by asking the linking flags to your python:

```bash
python3-config --ldflags
```

which returns `-lintl -ldl -L/Users/sebas/.pyenv/versions/3.12.4/lib -Wl,-rpath,/Users/sebas/.pyenv/versions/3.12.4/lib -framework CoreFoundation`, that is to link against the libraier `intl`, `dl` and look for those libraries in the specified cirectory `-L`. Finally also tells the linker to add a runtime path `-rpath` at which the executable will try to find the libraries at runtime. The last is specific to `MacOS`, this provides utilities for the operating system. This output would be different if we were on Windows or a Linux machine.

At this point we have the includes and libraries that we need to compile the C++ code into Python. We could write our bindings using the definitons in `Python.h`. This library is the official Python API to write C code, it allows you to have full control of the program but it is generally more difficult to write code compared to other options (see next section).

## Compiling a shared library for Python using Pybind11

A more convenient library to compile your shared python packages is [pybind11](https://github.com/pybind/pybind11), which is a header only library that exposes C++ types in Python and vice versa. For this you will need the python headers and libraries (shown in previous section) and `pybind11` that can be installed with `pip install pybind11`.


For now we will write a `bindings.cpp` file with all the "translated code":

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "matmul.h"

namespace py = pybind11;

void matmul_py(py::array_t<float> A, py::array_t<float> B, py::array_t<float> C) {
    auto bufA = A.request(), bufB = B.request(), bufC = C.request();

    if (bufA.ndim != 2 || bufB.ndim != 2 || bufC.ndim != 2) {
        throw std::runtime_error("All matrices must be 2D");
    }

    size_t M = bufA.shape[0];
    size_t N = bufA.shape[1];
    size_t K = bufB.shape[1];

    if (bufB.shape[0] != N || bufC.shape[0] != M || bufC.shape[1] != K) {
        throw std::runtime_error("Matrix dimensions do not match for multiplication");
    }

    float* ptrA = static_cast<float*>(bufA.ptr);
    float* ptrB = static_cast<float*>(bufB.ptr);
    float* ptrC = static_cast<float*>(bufC.ptr);

    matmul(ptrA, ptrB, ptrC, M, N, K);  // same call as before
}


PYBIND11_MODULE(matrix_mul, m) {
    m.def("matmul", &matmul_py, "Matrix multiplication function");
}
```
The function `matmul_py` takes three python numpy arrays, A, B and C and first checks that they are dimension 2. After that, we get the shapes of the pointers and get the pointers to the memory of each array. Finally we can call the C++ function `matmul`. Lastly we define our `PYBIND11_MODULE`, we expose the function `matmul_py` to be called as `matmul` in Python. And that should be it, now is time to compile. Bear with me, I'm going to throw a bunch of bash commands while explaining them in inline comments, in the root directory of the project run:

```bash
# create a building python environment
rm -rf .venv_build
python -m venv .venv_build
.venv_build/bin/pip install --upgrade pip

# install pybind11 using pip
.venv_build/bin/python -m pip install pybind11

```
Create the directory to hold the objects, binaries and the library

```bash
rm -rf build

# creating directories for the build
mkdir -p build/obj
mkdir build/bin
mkdir build/lib
```

Now we can compile the objects including the python and pybind11 headers (the output of `python -m pybind11 --includes` is `-I/Users/sebas/.pyenv/versions/3.12.4/include/python3.12 -I/Users/sebas/tmp/blogging-code/cpp-basic-cpp-python-extension/.venv_build/lib/python3.12/site-packages/pybind11/include` in my setup).

```bash
g++ -std=c++17 -Iinclude -c src/matmul.cpp -o build/obj/matmul.o
g++ -std=c++17 -Iinclude \
                $(.venv_build/bin/python -m pybind11 --includes) \
                -c src/bindings.cpp \
                -o build/obj/bindings.o
```

And finally we create the shared object

```bash
# grep the name of the major and minor versions of python, i.e. if we use 3.12.8 this will return python3.12
# this is the name of the python library
python_library=python$(.venv_build/bin/python --version | awk '{print $2}' | awk -F. '{print $1"."$2}')

g++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python3-config --ldflags) \
    -l${python_library} \
    build/obj/matmul.o \
    build/obj/bindings.o \
    -o build/lib/matrix_mul$(python3-config --extension-suffix)
```

And you will see a file `matrix_mul.cpython-312-darwin.so` in your `build/lib` directory. This is your compiled library!. Let me explain some key concepts. The command `python3-config --ldflags` gives you the flags needed to compile the python extension (explained before), the `-l` flag is to specifically link a library, in my case `python3.12`, then `python3-config --extension-suffix` gives the python version, architecture and operating system. It is used commonly to name the extension.

How can you import it?. Change directory to where the shared library is and try to impor it from there

```bash
cd build/lib
python -c "import matrix_mul"
```

This works here because "current directory" is always in the `sys.path` for the interpreter. We should place the library in the `libs` directory of our environment and then it could be imported every time we open a python prompt. 

## Testing

Let's write a python script to test our library, this file will be called `test_matmul.py` and will be placed under `tests` directory. 

```python
import numpy as np
import sys
from pathlib import Path

# we can't really import the library (shared object) from a script
# unless it's in the sys.path
SHARED_LIBRARY_DIR = Path(__file__).parents[1] / "build" / "lib"
sys.path.insert(0, str(SHARED_LIBRARY_DIR))

# now we can import our compiled library
import matrix_mul

# Define matrix dimensions
M, N, K = 32, 64, 32

# Create random matrices A[MxN] and B[NxK]
A = np.random.rand(M, N).astype(np.float32)
B = np.random.rand(N, K).astype(np.float32)
C = np.zeros((M, K), dtype=np.float32)  # Initialize C with zeros

# Call the compiled function
matrix_mul.matmul(A, B, C)

# Verify with NumPy
C_np = np.dot(A, B)

# Check if the results match
assert np.allclose(C, C_np), f"something went wrong, C and C_np are not equal"

print(f"Tests passed!")
```

The first part adds the path to the library we just compile so that the python interpreter can find it. The rest of the script is self explanatory, we use `numpy` to compare the two matrix multiplications. To start "fresh" we create a new python environment and call the script

```bash
rm -rf .venv_test
python -m venv .venv_test
.venv_test/bin/pip install --upgrade pip

# install numpy, required for the script tests/test_matmul.py
.venv_test/bin/pip install numpy

# run the test
.venv_test/bin/python tests/test_matmul.py
```

When executing this you should see `Tests passed!` as the last output. Congrats! You have learned the basic of Python bindings for C++ extensions.