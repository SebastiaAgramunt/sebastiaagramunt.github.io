---
title: CUDA Python package
author: sebastia
date: 2025-12-30 10:28:00 +0800
categories: [C++, Python, CUDA]
tags: [computer science]
pin: true
toc: true
render_with_liquid: false
math: true
---

## TLDR

In this post we will learn how to expose CUDA functionality in Python so effectively calling your custom CUDA code from Python without much hassle.

## Introduction

CUDA is not the simplest framework to learn, first you need to know C++ and then understand the Nvidia GPU internals. Then you can write kernels and parallelize your calculations. People like to code in Python as it is a super easy interpreted language to learn but it's simplicity sometimes is a drawback, i.e. you can't code very specific instructions in Python that interacts with the GPU, right?. Well, the Nvidia community is putting a lot of effort in creating tools for python to write efficient code for your GPU, at least that's one of the main takeaways I got from [Nvidia GTC conference 2025](https://www.nvidia.com/gtc/). For python you have [CuPy](https://cupy.dev/), [Numba](https://numba.pydata.org/), [JAX](https://docs.jax.dev/en/latest/), [Triton](https://openai.com/index/triton/), the [RAPIDS](https://docs.rapids.ai/) ecosystem containing cuDF, cuML, cuGraph, etc. Also in that conference they spoke a lot about [cuTile](https://docs.nvidia.com/cuda/cutile-python/) that finally has been released this month!.

All these libraries are super promissing, specially cuTile (I will write some posts about these libraries soon!). However sometimes one needs to have full control of the CUDA code and write directly the CUDA kernels. In my current position at Eikon Therapeutics I coded algorithms for detection and localization of proteins in images using CUDA kernels and reduced the calculation time from 3 minutes (CPU) to 5 seconds (GPU). Apart from the technical difficulty of coding the kernels and handling memory, one difficult part was to expose this functionality to a regular user that codes in Python. For that I learned how to compile the CUDA code with `nvcc` and create the bindings for Python. Also how to package this on a wheel for specific architecture and run the tests in an automated pipeline.

In this post we will learn how to expose CUDA functionality in Python so effectively calling your custom CUDA code from Python without much hassle. The code for this project can be found in my [python-cuda](https://github.com/SebastiaAgramunt/python-cuda) GitHub repository.

## The structure

Our project will have this structure

```
.
├── MANIFEST.in
├── README.md
├── cuda
│   ├── include
│   │   ├── cuBLASMultiply.h
│   │   ├── tiledMultiply.h
│   │   └── utils.h
│   └── src
│       ├── cuBLASMultiply.cu
│       └── tiledMultiply.cu
├── pyproject.toml
├── scripts
│   ├── build.sh
│   ├── launch.sh
│   └── script.py
├── setup.py
├── src
│   └── bindings.cpp
└── tests
    └── test_matmul.py
```

We will expose two functions, `matmul`, a tiled matrix multiplication, and `matmul_cublas`, a matrix multiplication using the library [cuBLAS](https://docs.nvidia.com/cuda/cublas/).


## CUDA specific files

The `tiledMultiply.cu` file has this contents:

```cpp
#include "tiledMultiply.h"

__global__ void tiledMultiply(const float* __restrict__ A, // M x K
                              const float* __restrict__ B, // K x N
                              float* __restrict__ C,       // M x N
                              std::size_t M,
                              std::size_t K,
                              std::size_t N) {

    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // global row/col this thread is responsible for
    int i = by * TILE + ty;  // row in C
    int j = bx * TILE + tx;  // col in C

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float value = 0.0f;

    // number of tiles along K
    int numTiles = (K + TILE - 1) / TILE;

    for (int ph = 0; ph < numTiles; ++ph) {
        // column in A, row in B that this thread wants to load
        int aCol = ph * TILE + tx;  // along K
        int bRow = ph * TILE + ty;  // along K

        // load A tile (row = i, col = aCol)
        if (i < M && aCol < K)
            As[ty][tx] = A[i * K + aCol];
        else
            As[ty][tx] = 0.0f;

        // load B tile (row = bRow, col = j)
        if (bRow < K && j < N)
            Bs[ty][tx] = B[bRow * N + j];
        else
            Bs[ty][tx] = 0.0f;

        // sync all threads to make sure the tiles are loaded
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < TILE; ++t) {
            value += As[ty][t] * Bs[t][tx];
        }

        // sync before loading the next tile
        __syncthreads();
    }

    // write back only if in-bounds
    if (i < (int)M && j < (int)N) {
        C[i * N + j] = value;
    }
}

void tiledMultiply_call(const float* __restrict__ A,
                    const float* __restrict__ B,
                    float* __restrict__ C,
                    std::size_t M,
                    std::size_t K,
                    std::size_t N){
    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    tiledMultiply<<<blocks, threads>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
```

This is, a CUDA kernel `tiledMultiply` and a C++ function that calls that kernel `tiledMultiply_call`. The header `tiledMultiply.h` includes just the last function exposed:

```cpp
#ifndef TILEDMULTIPLY_H
#define TILEDMULTIPLY_H

#include <iostream>
#include <cuda_runtime.h>

# define TILE 16
void tiledMultiply_call(const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    const std::size_t M,
    const std::size_t K,
    const std::size_t N);

#endif
```
These functions have been explained in the <a href="../cuda-matrix-multiplication">CUDA Matrix Multiplication</a> post. Also we included the `cuBLAS` equivalent `cuBLASMultiply.cu`:

```cpp
#include "cuBLASMultiply.h"

void cuBLASmultiply_call(const float* __restrict__ A,
                    const float* __restrict__ B,
                    float* __restrict__ C,
                    std::size_t M,
                    std::size_t K,
                    std::size_t N,
                    cudaStream_t stream){

    
    cublasHandle_t handle;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    CHECK_CUBLAS_ERROR(cublasSetStream(handle, stream));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // A: M x K (row-major)
    // B: K x N (row-major)
    // C: M x N (row-major)

    // We ask cuBLAS to compute: C^T = (B^T) * (A^T)
    CHECK_CUBLAS_ERROR(
        cublasSgemm(handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,               // m = rows of C^T
            M,               // n = cols of C^T
            K,               // k
            &alpha,
            B, N,            // matrix A is B, leading dimension N
            A, K,            // matrix B is A, leading dimension K
            &beta,
            C, N)
        );
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUBLAS_ERROR(cublasDestroy(handle));
}
```
And the corresponding header `cuBLASMultiply.h`:

```cpp
#ifndef CUBLASMATMULTIPLY_H
#define CUBLASMATMULTIPLY_H

#include "utils.h"
#include<cublas_v2.h>
#include <cuda_runtime.h>


void cuBLASmultiply_call(const float* __restrict__ A,
                    const float* __restrict__ B,
                    float* __restrict__ C,
                    std::size_t M,
                    std::size_t K,
                    std::size_t N,
                    cudaStream_t stream);

#endif
```

The `utils.h` contain a set of very useful functions for flagging errors in the cuda runtime execution. Won't explain but is a great resource to copy paste in any CUDA project.

## The bindings

Out of the two exposed functions I'm just going to explain the tiled, the cuBLAS is equivalent.

```cpp
py::array_t<float> matmul_tiled(
    py::array_t<float, py::array::c_style | py::array::forcecast> A,
    py::array_t<float, py::array::c_style | py::array::forcecast> B)
{
    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::runtime_error("A and B must be 2D arrays");
    }

    const auto M  = static_cast<std::size_t>(A.shape(0));
    const auto K  = static_cast<std::size_t>(A.shape(1));
    const auto Kb = static_cast<std::size_t>(B.shape(0));
    const auto N  = static_cast<std::size_t>(B.shape(1));

    if (K != Kb) {
        throw std::runtime_error("Inner dimensions must match: A(M,K) @ B(K,N)");
    }

    py::array_t<float> C({static_cast<py::ssize_t>(M),
                          static_cast<py::ssize_t>(N)});

    const float* hA = A.data();
    const float* hB = B.data();
    float* hC       = C.mutable_data();

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;

    std::size_t bytesA = M * K * sizeof(float);
    std::size_t bytesB = K * N * sizeof(float);
    std::size_t bytesC = M * N * sizeof(float);

    cuda_check(cudaMalloc(&dA, bytesA), "cudaMalloc dA failed");
    cuda_check(cudaMalloc(&dB, bytesB), "cudaMalloc dB failed");
    cuda_check(cudaMalloc(&dC, bytesC), "cudaMalloc dC failed");

    cuda_check(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice),
               "cudaMemcpy A failed");
    cuda_check(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice),
               "cudaMemcpy B failed");

    tiledMultiply_call(dA, dB, dC, M, K, N);

    cuda_check(cudaGetLastError(), "Kernel launch failed");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

    cuda_check(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost),
               "cudaMemcpy C failed");

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return C;
}
```

In the function we allocate the memory for the matrices `A`, `B` and `C` and

```cpp
cuda_check(cudaMalloc(&dA, bytesA), "cudaMalloc dA failed");
cuda_check(cudaMalloc(&dB, bytesB), "cudaMalloc dB failed");
cuda_check(cudaMalloc(&dC, bytesC), "cudaMalloc dC failed");
```

we copy the data:

```cpp
cuda_check(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice),
               "cudaMemcpy A failed");
cuda_check(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice),
               "cudaMemcpy B failed");
```

launch the calculation in GPU

```cpp
tiledMultiply_call(dA, dB, dC, M, K, N);
```

and copy back to host the calculated matrix `C`:

```cpp
cuda_check(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost),
               "cudaMemcpy C failed");
```

before freeing the memory. We decided to encapsualte all the logic of memory management here in the bindings file, however we could have written another c++ function to have this code and call it directly on the bindings. I'm trying to make things simpler for this example.

## Python project specifics

The python project needs for a `pyproject.toml` first just to indicate the build system:

```
[build-system]
requires = ["setuptools==70.3.0", "wheel", "pybind11>=2.6", "numpy"]
build-backend = "setuptools.build_meta"
```

Then we need to specify the `setup.py`, this file will specify how to compile and build the code. It's the default in Python. Let's show the file by parts, at the beginning we have

```python
import os
import sys
import subprocess
import sysconfig
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

REPO_PATH = Path(__file__).resolve().parent

python_include_path = sysconfig.get_path("include")

CUDA_HOME = "/usr/local/cuda"
CUDA_INCLUDE_DIR = os.path.join(CUDA_HOME, "include")
CUDA_LIB_DIR = os.path.join(CUDA_HOME, "lib64")
PACKAGE_NAME = "matmul"

INCLUDE_DIRS = [
    "cuda/include",
    CUDA_INCLUDE_DIR,
    python_include_path,
    pybind11.get_include(),
]

LIBRARY_DIRS = [CUDA_LIB_DIR]
LIBRARIES = ["cudart", "cublas"]

CXX_FLAGS = ["-std=c++17", "-O3"]
NVCC_FLAGS = [
    "-std=c++17",
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-arch=sm_80",
]

SRC_FILES = [
    "src/bindings.cpp",
    "cuda/src/tiledMultiply.cu",
    "cuda/src/cuBLASMultiply.cu",
]
```

The `python_include_path` is the path to the includes of python (bascially `Python.h`). The `CUDA_HOME` is the path to the CUDA libraries and includes, it can change depending on the system, adjust accordingly (you can also use CMAKE if you are up for it). Then we join all the include directories in `INCLUDE_DIRS` list, then the library directories in `LIBRARY_DIRS`, then libraries in `LIBRARIES` and then the flags for the C++ and nvcc compilers. Finally the `SRC_FILES` that we are going to compile.

Next we need to check that the machine has `nvcc` compiler

```python
try:
    subprocess.check_call(["nvcc", "--version"])
except Exception as e:
    print(f"nvcc compiler for CUDA not found: {e}; exiting")
    sys.exit(1)
```

And now we need to write the compiling logic

```python
class BuildExtCUDA(build_ext):
    """Compile .cu files with nvcc, others with the normal C++ compiler."""

    def build_extensions(self):
        from distutils.sysconfig import customize_compiler

        compiler = self.compiler
        customize_compiler(compiler)

        # Let distutils know about .cu files
        if ".cu" not in compiler.src_extensions:
            compiler.src_extensions.append(".cu")

        default_compile = compiler._compile
        nvcc = "nvcc"

        def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith(".cu"):
                # nvcc compile
                cmd = [nvcc, "-c", src, "-o", obj] + NVCC_FLAGS
                for inc in INCLUDE_DIRS:
                    cmd.append(f"-I{inc}")
                print("NVCC:", " ".join(cmd))
                self.spawn(cmd)
            else:
                # normal C++ compile
                extra_postargs = list(extra_postargs or []) + CXX_FLAGS
                default_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        compiler._compile = _compile
        super().build_extensions()
```

This is a sublass of `build_ext` class. We overload the function `_compile` from the parent class. In this case, if the file ends with `.cu` we build the command `cmd` to execute in a subprocess `cmd = [nvcc, "-c", src, "-o", obj] + NVCC_FLAGS` then include the includes one by one in a for loop. Finally this command is launched in bash. If the file is not ending with `.cu` and is listed in the source files, then we assume its C++ and compile it with the default compiler (usucally `g++` or `gcc`).

Now we define the `ext_modules` and the `setup` file

```python
ext_modules = [
    Extension(
        PACKAGE_NAME,
        sources=SRC_FILES,
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES,
        language="c++",
    )
]

setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    description="CUDA tiled matrix multiplication exposed to Python via pybind11",
    author="Sebastia Agramunt Puig",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtCUDA},
    zip_safe=False,
    install_requires=[
        "numpy",
    ],
)
```

In the setup we specify how to build the external modules, and its through the class `BuildExtCUDA`. That's it!, that makes it compilable and pip installable.

## The manifest

The file `MANIFEST.in` is crucial when creating wheels, in this we tell the python build to include certain files:

```
recursive-include cuda/include *.h
recursive-include cuda/src *.cu
recursive-include src *.h *.cpp
```

specially we need the headers, otherwise the code won't work as the binaries need for the function definitions there.

## Install the package

Just create a new environment and `pip install` the package

```bash
rm -rf .venv
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install .
```

Now you can run the script to test your code:

```bash
.venv/bin/python scripts/script.py
```

## Testing

As usual I include some testing. It's very important to test your code always!. The tests are very simple, just check that the `matmul` and `matmul_cublas` yield the same result. To execute the tests run

```bash
.venv/bin/python -m pip install pytest
.venv/bin/pytest -v .
```

after installing the python environment.


## Building a wheel

For convenience I included a bash script `scripts/build.sh` to build the wheels. Just execute the following

```bash
TASK=install_environment  ./scripts/build.sh
TASK=run_tests  ./scripts/build.sh
TASK=build_wheel  ./scripts/build.sh
TASK=test_install_wheel  ./scripts/build.sh
TASK=cleanup  ./scripts/build.sh
```

Obviously the `build_wheel` task will build the wheel. It places it in the `wheelhouse` directory. Let's inspect this function

```bash
build_wheel(){
    
    rm -rf ${ROOT_DIR}/dist
    rm -rf ${ROOT_DIR}/build

    # create blank environment
    rm -rf ${ROOT_DIR}/${ENV_NAME}
    python -m venv ${ROOT_DIR}/${ENV_NAME}
    ${ROOT_DIR}/${ENV_NAME}/bin/python -m pip install --upgrade pip

    # activate, install pkgs and build wheel
    source ${ROOT_DIR}/${ENV_NAME}/bin/activate
    pip install wheel pybind11 auditwheel repairwheel patchelf build
    pip install setuptools==70.3.0
    python -m build

    if [ $(arch) = "x86_64" ]; then
        platform="manylinux_2_34_x86_64"
    elif [ $(arch) = "aarch64" ]; then
        platform="manylinux_2_34_aarch64"
    else
        echo "ERROR: Unknown architecture"
        exit 1;
    fi

    auditwheel repair --exclude libcu* \
                      $(ls dist/*.whl | head -n 1) \
                      --plat ${platform} \
                      -w wheelhouse
}
```

The function removes the directories `dist` and `build` to start fresh. Then creates a new python environment and activates it. After that installs `wheel`, `pybind11`, `auditwheel`, `repairwheel`, `patchelf`, `build` and `setuptools`.

The step that really builds the wheel is `python -m build`, this will create the wheel directly in the `build` directory. At this point we need to repair the wheel: Linux systems have different versions of the library `GLIBC`, in this case we want to make it compatible from version 2.34 (see [list of versions](https://ftp.gnu.org/gnu/glibc/)) onwards. For that we indicate the plaform `platform="manylinux_2_34_x86_64"`. The next instruction will "repair" this wheel 

```bash
auditwheel repair --exclude libcu* \
                    $(ls dist/*.whl | head -n 1) \
                    --plat ${platform} \
                    -w wheelhouse
```

and exclude all libraries starting from `libcu`. That's key in the auditwheel, this program incldues all the libraries in the wheel so that it is complete and therefore there's no need to install any external libraries. We decide to exclude the cuda libraries because to run any cuda program you need to have those libraries installed... It would be duplicated, besides they are quite heavy in memory usage.

After running this `auditwheel` the wheel will appear in the directory we indicated `wheelhouse`.

In the official documentation of [auditwheel](https://github.com/pypa/auditwheel) the developers use docker images listed in [https://quay.io](https://quay.io/organization/pypa). Those work well for C++ only code and not for CUDA code, this is why I had to come up with a manual way to build and repair the wheel.

Another problem we are having with CUDA extensions is that for now GitHub won't have CUDA agents (i.e. machines with GPUs able to run CUDA code) so if you want to implement proper CI/CD you need to create your own pipeline in a custom machine. [Jenkins](https://www.jenkins.io/) could be a good tool for that. I used a comertial software that my employer provided but It's essentially the same (bash scripts here and there).

## Conclusions

This is a very simple example on how to create a Python package that uses CUDA in the backend. You can complicate this further, add other C++ implementation, more functions, more tests... But I hope by reading this you could understand the basics and have the tooling to build your first python bindings for CUDA.