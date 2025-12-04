---
title: BLAS and LAPACK for Linear Algebra
author: sebastia
date: 2025-09-14 12:35:00 +0800
categories: [C++]
tags: [computer science, mathematics]
pin: true
toc: true
render_with_liquid: false
math: true
---

[BLAS](https://www.netlib.org/blas/) is the basic linear algebra subprograms library, It's a well tested library for basic algebraic operations. For instance, vector operations, dot products, matrix transpositions, multiplications etc. The [LAPACK](https://www.netlib.org/lapack/) library (Linear Algebra PACKage) is a higher level library built on top of BLAS: while BLAS provides low-level building blocks (vector, matrix operations), LAPACK implements full algorithms for solving core linear algebra problems. For instance, with LAPACK we cans olve systems of linear equations (LU decomposition), least square problems, eigenvalue problems, matrix factorizations (LU, Cholesky, QR, Schur). Lapack is oringally built in Fortran, so the indexes are colum-order instead of the usual row-order in C, that might be a source of error for an avid C developer, however it exist a C wrapper for LAPACK, called [LAPACKE](https://www.netlib.org/lapack/) (see [the user guide](https://www.netlib.org/lapack/lapacke.html)).

In this tutorial we will show how to install BLAS, LAPACK ans LAPACKE and compile a simple program that uses one function of each. Specifically we will install first [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS), which is a concrete implementation of the BLAS standard with optimizations and extensions that also bundles LAPACK, but not LAPACKE, which will be also installed in this post. Find the [code](https://github.com/SebastiaAgramunt/blogging-code/tree/main/blas-lapacke) in the GitHub [blogging-code](https://github.com/SebastiaAgramunt/blogging-code/tree/main) repository.


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

## Install OpenBlas and Lapack in Ubuntu

To install in Ubuntu use `apt-get`:

```bash
apt-get update
apt-get install -y libopenblas-dev liblapacke-dev
```

I tried the above in a Ubuntu docker container in my MacOS:

```bash
docker pull ubuntu
docker run --rm -it --entrypoint bash ubuntu
```

Find the headers in

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

## Compile and install Openblas and Lapack

This is my favourite way to install libraries, download the source code and install in your project directory. I agree it takes more time but if you only use these libraries in one project may be worth just installing them in one directory. The project structure subdirectory for this install is

```bash
.
├── README.md
├── scripts
│   ├── build-run.sh
│   └── install-external-libraries.sh
└── src
    ├── cblas_example.cpp
    └── lapacke_example.cpp
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
# we ran this in Ubuntu x86_64 architecture
# can be different in other OSs and arch.
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

### Download and build Lapack/Lapacke

Lapack and Lapacke are basically the same libraries, Lapack is the original library built in Fortran, which treats matrices by default in column order. If you come from the C/C++ world like me, you would prefer to use lapacke, which is a C wrapper for the standard lapack library. In Lapacke, routines are row-order by default. Let's install both libraries.

Let's begin by downloading the source files

```bash
LAPACK_VERSION="3.12.1"
LAPACK_URL="https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v${LAPACK_VERSION}.tar.gz"

# download, unpack and change directory
wget ${LAPACK_URL}
tar -xvzf v${LAPACK_VERSION}.tar.gz
cd lapack-${LAPACK_VERSION}
```

and define the installation dir

```bash
INSTALL_DIR=${HOME}/libs
```

Assuming again we are on Linux:

```bash
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/lapack \
    -DCBLAS=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DLAPACKE=ON \
    -DBLAS_LIBRARIES="${LIB_DIR}/openblas/lib/libopenblas.so" \
    -DCMAKE_BUILD_TYPE=Release \
    ..
```

Finally you can check that both of your libraries, `openblas` and `lapacke` are in the subdirectoires `${INSTALL_DIR}/openblas` and `${INSTALL_DIR}/lapack`. There you should find the includes and libraries compiled, shared and static objects.


## Code examples


### BLAS examples

Now is time to use the libraries. Please check the [code](https://github.com/SebastiaAgramunt/blogging-code/tree/main/blas-lapack-install) in the main GitHub repository for full details, we will be writing the file `cblas_example.cpp` from there. We will do first a basic dot product of two vectors, take a look at the [BLAS documentation](https://www.netlib.org/blas/) and to your `cblas.h` header to see the definitions of the functions. We use is `cblas_ddot`, which is the `cblas` implementation of the double `dot` (`ddot`) function. In the BLAS documentation the signature is `double cblas_ddot(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST double *y, OPENBLAS_CONST blasint incy);` where

- `n`: number of elements of the vectors
- `x`: pointer to the first array
- `incx`: stride for first array
- `y`: pointer to the second array
- `incy`: stride for second array.

The code is something like

```cpp
#include <cblas.h>
#include <iostream>
#include <vector>

std::vector<double> x = {1, 2, 3};
std::vector<double> y = {4, 5, 6};

// BLAS double dot operation
double dot = cblas_ddot(3, x.data(), 1, y.data(), 1);
```

Since we want the dot product we set strides to 1. The result of this operation is 32.

For a second operation we will do a matrix multiplication using the function `dgemm`, which is the double eversion of the GEneral Matrix Multiplication. Checking the [BLAS documentation](https://www.netlib.org/blas/) (you can also check `cblas.h`) we see the signature of the function is 

```cpp
void cblas_dgemm(
  OPENBLAS_CONST enum CBLAS_ORDER     Order,   // memory layout: row/col-major
  OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,  // op on A: NoTrans / Trans / ConjTrans
  OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB,  // op on B: NoTrans / Trans / ConjTrans
  OPENBLAS_CONST blasint M,                    // rows of op(A) and C
  OPENBLAS_CONST blasint N,                    // cols of op(B) and C
  OPENBLAS_CONST blasint K,                    // cols of op(A) and rows of op(B)
  OPENBLAS_CONST double alpha,                 // scales A*B
  OPENBLAS_CONST double *A,                    // pointer to A
  OPENBLAS_CONST blasint lda,                  // leading dimension of A
  OPENBLAS_CONST double *B,                    // pointer to B
  OPENBLAS_CONST blasint ldb,                  // leading dimension of B
  OPENBLAS_CONST double beta,                  // scales existing C
  double *C,                                   // pointer to C (in/out)
  OPENBLAS_CONST blasint ldc                   // leading dimension of C
);
```

 and the operation is

$$C=\alpha A^*B^*+ \beta C$$


The parameters are explained in the signature here. One thing that may be consufing is the leading dimension of the matrices. If the matrix is row-major, then the leading dimension is the number of columns, and vice versa, if the matrix is column-major, the leading dimension is the number of rows. This is important as we are passing arrays and the algorithm needs to know each dimension and how the matrices are expressed. To set an example, let's muliply a matrix `A` of size `M x K = 2 x 3` and a matrix `B` of size `K x N = 3 x 2` and store the result in `C` of size `M x N = 2 x 2` using the function `cblas_dgemm`:

```cpp
#include <cblas.h>
#include <iostream>
#include <vector>

const int M = 2, K = 3, N = 2;

// Row-major layout
std::vector<double> A = {
    1, 2, 3,
    4, 5, 6
}; // 2x3=MxK

std::vector<double> B = {
    7,  8,
    9, 10,
    11, 12
}; // 3x2=K,N

// C: matrix of zeroes, we save the result there
// A (MxK), B (KxN) -> C (MxN)
std::vector<double> C(M * N, 0.0); // 2x2

// C := alpha * Op(A) * Op(B) + beta * C
cblas_dgemm(
    CblasRowMajor,    // Matrix order, our case is Row major
    CblasNoTrans,     // Transpose matrix A
    CblasNoTrans,     // Transpose matrix B
    M,                // number of rows of op(A) and C
    N,                // number of columns of op(B) and C
    K,                // number of columns of op(A) and rows of op(B)
    1.0,              // alpha
    A.data(),         // A
    K,                // for row-major, lda = #cols of A
    B.data(),         // B
    N,                // for row-major, ldb = #cols of B
    0.0,              // beta
    C.data(),         // C
    N                 // for row-major, ldc = #cols of C
);

std::cout << "C = A*B:\n";
for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j)
        std::cout << C[i * N + j] << " ";
    std::cout << "\n";
}
// Expected:
// [ 58  64 ]
// [139 154 ]
```

The result matches the expected. In the [code](https://github.com/SebastiaAgramunt/blogging-code/tree/main/blas-lapack-install) you will find a file named `src/cblas_example.cpp`. To compile it use the bash script `scripts/build-run.sh`, there you will find this code:

```bash

OPENBLAS_INC="${ROOT_DIR}/external/lib/openblas/include/openblas"
OPENBLAS_LIB="${ROOT_DIR}/external/lib/openblas/lib"

# OPENBLAS example
# compile object
g++ -O3 \
    -std=c++17 \
    -c ${ROOT_DIR}/src/cblas_example.cpp \
    -I${OPENBLAS_INC} \
    -o ${ROOT_DIR}/build/obj/cblas_example.o

# compile binary
g++ -O3 \
    ${ROOT_DIR}/build/obj/cblas_example.o \
    -L${OPENBLAS_LIB} \
    -lopenblas \
    -Wl,-rpath,${OPENBLAS_LIB} \
    -o ${ROOT_DIR}/build/bin/cblas_example
```

where we basically first compile the file to an object with the includes and then link with the openblas library. This code assumes that your openblas library has been installed in the repository directory `external/lib/openblas` directory. If this is not the case and you have installed elsewhere, just change the variables `OPENBLAS_INC` and `OPENBLAS_LIB`. Also assumes we created several directories to store our object files and binaries inside the project.

If you follow the bash script, just execute `scripts/build-run.sh` and after compilation the executables will be in `build/bin` in the same project directory.

### LAPACKE examples

In Lapacke we will do just one example in one source file that we will name `lapacke_example.cpp`. As before all signatures for this library (Lapacke) will be in `lapacke.h` in your installed directory. In this example we will use the function [dgesv](https://netlib.org/lapack/explore-html-3.6.1/d7/d3b/group__double_g_esolve_ga5ee879032a8365897c3ba91e3dc8d512.html) to calculate the soluton to areal system of linear equations.

$$A \times X = B$$

where $A$ is a $N$ by $N$ matrix, and $X$ and $B$ are vectors of size $N$ times right-hand sides (solutions). The function `dgesv` has signature:

```cpp
lapack_int LAPACKE_dgesv(
    int matrix_layout,     // LAPACK_ROW_MAJOR or LAPACK_COL_MAJOR
    lapack_int n,          // order of A (A is n×n)
    lapack_int nrhs,       // number of right-hand sides (columns of B/X)
    double* a,             // in: A (n×n); out: combined L and U factors
    lapack_int lda,        // leading dimension of A
    lapack_int* ipiv,      // out: pivot indices (size n, 1-based)
    double* b,             // in: B (n×nrhs); out: solution X (n×nrhs)
    lapack_int ldb         // leading dimension of B
);
```

This function computes the LU factorization with partial pivoting of A and solves $A \times X = B$. We already explained leading dimensions in the previous section, here we have something new, the pivot indices. These are indices that are used by the algorithm internally to pivot (permute) indices. This is part of the [LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition) algorithm, we won't extend here to explain the algorithm.

We define the following system of equations for our example

$$
\left[ {\begin{array}{ccc}
3 & 1 & 2 \\
6 & 3 & 4\\
3 & 1 & 5\\
\end{array} } \right]
\times
\left[ {\begin{array}{c}
x \\
y \\
z \\
\end{array}} \right]
=
\left[ {\begin{array}{c}
0 \\
1 \\
3 \\
\end{array}} \right]
$$

with solution

$$
\left[ {\begin{array}{c}
x \\
y \\
z \\
\end{array}} \right]
=
\left[ {\begin{array}{c}
-1 \\
1 \\
1 \\
\end{array}} \right]
$$

Let's write the source file example to solve this equation using `dgesv`:

```cpp
#include <cstdio>
#include <vector>
#include <lapacke.h>

// Solve A x = b for x, overwriting b with the solution.
// Uses LAPACKE_dgesv (LU factorization with partial pivoting).
int main() {
    // Example 3x3 system
    // A =
    // [ 3  1  2 ]
    // [ 6  3  4 ]
    // [ 3  1  5 ]
    // b = [ 0, 1, 3 ]^T
    const int n = 3;          // order of A
    const int nrhs = 1;       // number of right-hand sides
    const int lda = n;        // leading dimension of A (row-major -> lda = n)
    const int ldb = nrhs;     // leading dimension of B (row-major -> ldb = nrhs)

    // Row-major storage (C style)
    std::vector<double> A = {
        3.0, 1.0, 2.0,
        6.0, 3.0, 4.0,
        3.0, 1.0, 5.0
    };
    std::vector<double> b = { 0.0, 1.0, 3.0 };

    // Pivot indices
    std::vector<lapack_int> ipiv(n);

    // Call LAPACKE (row-major)
    lapack_int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR,
                                    n, nrhs,
                                    A.data(), lda,
                                    ipiv.data(),
                                    b.data(), ldb);
    if (info > 0) {
        std::fprintf(stderr, "U(%d,%d) is exactly zero; singular matrix.\n", info, info);
        return 1;
    } else if (info < 0) {
        std::fprintf(stderr, "Argument %d to dgesv had an illegal value.\n", -info);
        return 1;
    }

    // b now contains the solution x
    std::printf("Solution x:\n");
    for (int i = 0; i < n; ++i) {
        std::printf("x[%d] = %.2f\n", i, b[i]);
    }
    return 0;
}
```

This code prints the solution on screen. I agree that we don't need `double` or enven `float` precision for this calculation (solution is `(-1, 1, 1)`)but there are no functions for `int` precision in lapacke. For more functions check the [lapack](https://www.netlib.org/lapack/lug/) documentation and take a look at your `lapacke.h` header where you will find all the definitions.

To compile the above code run the following if you have installed `openblas` and `lapacke` in the current project directory

```bash
OPENBLAS_INC="${ROOT_DIR}/external/lib/openblas/include/openblas"
OPENBLAS_LIB="${ROOT_DIR}/external/lib/openblas/lib"

LAPACKE_INC="${ROOT_DIR}/external/lib/lapack/include"
LAPACKE_LIB="${ROOT_DIR}/external/lib/lapack/lib"

# compile object
g++ -O3 \
    -std=c++17 \
    -c ${ROOT_DIR}/src/lapacke_example.cpp \
    -I${LAPACKE_INC} \
    -o ${ROOT_DIR}/build/obj/lapacke_example.o

# # compile binary
g++ -O3 \
"${ROOT_DIR}/build/obj/lapacke_example.o" \
-L"${LAPACKE_LIB}" \
-L"${OPENBLAS_LIB}" \
-llapacke -lopenblas \
-Wl,-rpath,"${LAPACKE_LIB}" \
-Wl,-rpath,"${OPENBLAS_LIB}" \
-o "${ROOT_DIR}/build/bin/lapacke_example"
```

where `ROOT_DIR` is the root directory of the repository. If you have installed the libraries elsewhere, just change the include and library directories above. In any case there is a bash script in the [GitHub repository](https://github.com/SebastiaAgramunt/blogging-code/tree/main/blas-lapack-install) that compiles the executable.

## Final remarks

I hope you have enjoyed this tutorial. I believe it's important to use these two libraries for numerical computing, knowning them well can save you a lot of time (not only yours, computational time!). Besides these libraries are really well optimized, they are old and still well maintained, it's the standard.
