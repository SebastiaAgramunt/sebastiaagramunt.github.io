---
title: CUDA utils
author: sebastia
date: 2025-07-13 12:35:00 +0800
categories: [C++, CUDA]
tags: [computer science, GPU]
pin: true
toc: true
render_with_liquid: false
math: true
---

This is my first post in CUDA, I have been working for a while using this technology and want to share some utilities that can be useful for newcomers to the field. All the code will be available in my [github repository](https://github.com/SebastiaAgramunt/blogging-code) subdirectory [cuda-utils](https://github.com/SebastiaAgramunt/blogging-code/tree/main/cuda-utils). I would like to acqnowledge [Lambda.ai](https://cloud.lambda.ai/) for providing me with free credits for my blog. I will be testing this code with a machine `gpu_1x_a100_sxm4` which has an A100 GPU, the Ampere GPU architecture. This GPU is a bit old these days but we won't be doing any heavy compute so this will suffice.

## Project structure

Files in this project will be structured as follows

```
.
├── CMakeLists.txt
├── README.md
└── src
    ├── gpu_allocate.cu
    └── gpu_info.cu
```

## GPU info tool

The first tool just displays some basic information of the GPUs available in the system, create a file `gpu_info.cu` in with the code:

```cpp
#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device count: " 
                  << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Detected " << deviceCount << " CUDA Capable Device(s)\n\n";

    for (int dev = 0; dev < deviceCount; ++dev) {
        // Select device
        cudaSetDevice(dev);

        // Query device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        // Query memory info
        size_t freeBytes = 0, totalBytes = 0;
        cudaMemGetInfo(&freeBytes, &totalBytes);

        std::cout << "Device " << dev << ": " << prop.name << "\n";
        std::cout << "  PCI Domain/Bus/Device ID: " 
                  << prop.pciDomainID << "/" 
                  << prop.pciBusID    << "/" 
                  << prop.pciDeviceID << "\n";
        std::cout << "  Compute capability: " 
                  << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total global memory: " 
                  << (prop.totalGlobalMem  / (1024.0 * 1024.0)) << " MB\n";
        std::cout << "  Free memory (current): " 
                  << (freeBytes  / (1024.0 * 1024.0)) << " MB\n";
        std::cout << "  Total allocatable memory (current): " 
                  << (totalBytes / (1024.0 * 1024.0)) << " MB\n";
        std::cout << "  Memory clock rate: " 
                  << (prop.memoryClockRate * 1e-3) << " MHz\n";
        std::cout << "  Memory bus width: " 
                  << prop.memoryBusWidth << " bits\n";
        std::cout << "  L2 cache size: " 
                  << prop.l2CacheSize / 1024 << " KB\n";
        std::cout << "  Max shared memory per block: " 
                  << prop.sharedMemPerBlock / 1024 << " KB\n";
        std::cout << "  Total constant memory: " 
                  << prop.totalConstMem / 1024 << " KB\n";
        std::cout << "  Warp size: " 
                  << prop.warpSize << "\n";
        std::cout << "  Max threads per block: " 
                  << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Max threads per multiprocessor: " 
                  << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "  Multiprocessor count: " 
                  << prop.multiProcessorCount << "\n";
        std::cout << "  Max grid dimensions: [" 
                  << prop.maxGridSize[0] << ", " 
                  << prop.maxGridSize[1] << ", " 
                  << prop.maxGridSize[2] << "]\n";
        std::cout << "  Max block dimensions: [" 
                  << prop.maxThreadsDim[0] << ", " 
                  << prop.maxThreadsDim[1] << ", " 
                  << prop.maxThreadsDim[2] << "]\n";
        std::cout << "  Clock rate: " 
                  << (prop.clockRate * 1e-3) << " MHz\n";
        std::cout << "  Concurrent kernels: " 
                  << (prop.concurrentKernels ? "Yes" : "No") << "\n";
        std::cout << "  ECC enabled: " 
                  << (prop.ECCEnabled ? "Yes" : "No") << "\n";
        std::cout << "  Integrated device: " 
                  << (prop.integrated ? "Yes" : "No") << "\n";
        std::cout << "  Can map host memory: " 
                  << (prop.canMapHostMemory ? "Yes" : "No") << "\n";
        std::cout << "  Compute mode: ";
        switch (prop.computeMode) {
            case cudaComputeModeDefault:      std::cout << "Default\n"; break;
            case cudaComputeModeExclusive:    std::cout << "Exclusive\n"; break;
            case cudaComputeModeProhibited:   std::cout << "Prohibited\n"; break;
            case cudaComputeModeExclusiveProcess:
                                              std::cout << "Exclusive Process\n"; break;
            default:                          std::cout << "Unknown\n"; break;
        }
        std::cout << "  Unified addressing: " 
                  << (prop.unifiedAddressing ? "Yes" : "No") << "\n";
        std::cout << "  Async engines: " 
                  << prop.asyncEngineCount << "\n";
        std::cout << "  Device overlap: " 
                  << (prop.deviceOverlap ? "Yes" : "No") << "\n";
        std::cout << "  PCI bus ID: " 
                  << prop.pciBusID << "\n";
        std::cout << "  PCI device ID: " 
                  << prop.pciDeviceID << "\n";
        std::cout << "\n";
    }

    return 0;
}
```

The main ingredient is `cudaDeviceProp` a struct defined in `cuda_runtime.h` (see documentation [here](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html)) that contains properites of the devices. Before printing out on screen properties we count the devices and then loop over all of them to print out he properites using the device propery variable. Let's see what is the output of an A100 gpu from lambda.ai:

```bash
Detected 1 CUDA Capable Device(s)

Device 0: NVIDIA A100-PCIE-40GB
  PCI Domain/Bus/Device ID: 0/7/0
  Compute capability: 8.0
  Total global memory: 40442.4 MB
  Free memory (current): 40019.6 MB
  Total allocatable memory (current): 40442.4 MB
  Memory clock rate: 1215 MHz
  Memory bus width: 5120 bits
  L2 cache size: 40960 KB
  Max shared memory per block: 48 KB
  Total constant memory: 64 KB
  Warp size: 32
  Max threads per block: 1024
  Max threads per multiprocessor: 2048
  Multiprocessor count: 108
  Max grid dimensions: [2147483647, 65535, 65535]
  Max block dimensions: [1024, 1024, 64]
  Clock rate: 1410 MHz
  Concurrent kernels: Yes
  ECC enabled: Yes
  Integrated device: No
  Can map host memory: Yes
  Compute mode: Default
  Unified addressing: Yes
  Async engines: 3
  Device overlap: Yes
  PCI bus ID: 7
  PCI device ID: 0
```

It tells us the memory (global) is around 40GB and it is mostly free. Warp size is 32, which is quite usual in many architectures. Maximum threads per block 1024, also very common, and maximum block dimensions [1024, 1024, 64]. I like this tool just to know my limits when I code cuda kernels (a high level API to interact with the Nvidia card).


## GPU allocate tool

This tool is a bit different, it can be used to block a chunk of gpu memory and serves as a hello wolrd example on how to code basic cuda. Write into `gpu_allocate.cu` the content:

```cpp
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <cstdlib>
#include <chrono>
#include <thread>

// Helper to parse size strings like 1024, 100M, 2G, etc.
size_t parseSize(const std::string& s) {
    char unit = s.back();
    std::string num = s;
    size_t multiplier = 1;
    if (unit == 'K' || unit == 'k') {
        multiplier = 1024ULL;
        num = s.substr(0, s.size() - 1);
    } else if (unit == 'M' || unit == 'm') {
        multiplier = 1024ULL * 1024ULL;
        num = s.substr(0, s.size() - 1);
    } else if (unit == 'G' || unit == 'g') {
        multiplier = 1024ULL * 1024ULL * 1024ULL;
        num = s.substr(0, s.size() - 1);
    }
    return static_cast<size_t>(std::stoull(num) * multiplier);
}

// Helper to parse time strings like 10s, 5m, 1h, or raw seconds
long parseTime(const std::string& s) {
    char unit = s.back();
    std::string num = s;
    long multiplier = 1;
    if (unit == 's' || unit == 'S') {
        multiplier = 1;
        num = s.substr(0, s.size() - 1);
    } else if (unit == 'm' || unit == 'M') {
        multiplier = 60;
        num = s.substr(0, s.size() - 1);
    } else if (unit == 'h' || unit == 'H') {
        multiplier = 3600;
        num = s.substr(0, s.size() - 1);
    }
    return static_cast<long>(std::stol(num) * multiplier);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <gpu_id> <memory_amount (e.g., 512M, 1G, or bytes)> <duration (e.g., 10s, 5m, 1h)>" << std::endl;
        return EXIT_FAILURE;
    }

    int gpuId = std::stoi(argv[1]);
    size_t bytes = parseSize(argv[2]);
    long duration = parseTime(argv[3]);

    cudaError_t err = cudaSetDevice(gpuId);
    if (err != cudaSuccess) {
        std::cerr << "Error setting GPU device " << gpuId << ": " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    void* d_ptr = nullptr;
    err = cudaMalloc(&d_ptr, bytes);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating " << bytes << " bytes on GPU " << gpuId
                  << ": " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Successfully allocated " << bytes << " bytes on GPU " << gpuId
              << ", holding for " << duration << " seconds..." << std::endl;

    // Keep the allocation alive for the specified duration
    std::this_thread::sleep_for(std::chrono::seconds(duration));

    // Free the allocation and exit
    cudaFree(d_ptr);
    std::cout << "Freed memory and exiting." << std::endl;
    return EXIT_SUCCESS;
}
```

Let's take a look at the main, it is a command line tool whith three inputs in the `argv` argument, the `gpuId`, the number of `bytes` and the `duration` in seconds. Basically we want to allocate a number of bytes in a specific gpu during a certain ammount of time. The next part is sleecting the gpu with `cudaSetDevice` function and allocate the memory with `cudaMalloc(&d_ptr, bytes)` where `d_ptr` is a pointer to void. Then on the cpu side we tell it to sleep for the ammount of seconds we selected with `std::this_thread::sleep_for(std::chrono::seconds(duration))`, and finally after that time is elapsed we dealocate the memory with `cudaFree` and exit the program with success code. The functions `ParseSize` and `ParseTime` are just two helpers to match the sizes kilobytes, megabytes, gigabytes to bytes and the times hours, minutes, seconds to seconds.

## The CMakeLists.txt file

Cmake is a super powerful command line tool that creates a make for your project. It is very convenient in C++ and CUDA projects. Write a `CMakeLists.txt` file with this content

```bash
cmake_minimum_required(VERSION 3.10)
project(GPUTools LANGUAGES CXX CUDA)

# default exec names
set(GPU_INFO_OUT_NAME    "gpu_info"
    CACHE STRING "Name of the gpu_info executable")
set(GPU_ALLOC_OUT_NAME   "gpu_allocate"
    CACHE STRING "Name of the gpu_allocate executable")

# Restore old FindCUDA behavior if needed
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

# Language standards
set(CMAKE_CXX_STANDARD      14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING
    "List of CUDA architectures to build for (e.g. 61;70;75;86)")

# Find CUDA (for older CMake) or you can use find_package(CUDAToolkit) in 3.17+
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(gpu_info
  src/gpu_info.cu
)
target_link_libraries(gpu_info
  PRIVATE ${CUDA_CUDART_LIBRARY}
)
set_target_properties(gpu_info
  PROPERTIES OUTPUT_NAME ${GPU_INFO_OUT_NAME}
)

add_executable(gpu_allocate
  src/gpu_allocate.cu
)
target_link_libraries(gpu_allocate
  PRIVATE ${CUDA_CUDART_LIBRARY}
)
set_target_properties(gpu_allocate
  PROPERTIES OUTPUT_NAME ${GPU_ALLOC_OUT_NAME}
)

# (Optional) If you want to give a different on-disk name:
# set(EXE_NAME alloc_mem)
# set_target_properties(allocate_gpu_memory PROPERTIES OUTPUT_NAME ${EXE_NAME})

# Installation
install(TARGETS
  gpu_info
  gpu_allocate
  RUNTIME DESTINATION bin
)
```

There are two variables that are set by default but can be changed when we call cmake command line. `GPU_INFO_OUT_NAME` and `GPU_ALOC_OUT_NAME`, those two are the names of the executables. We set the `C++` standard and the `CMAKE_CUDA_ARCHITECTURES` is the architecture of the GPU we are compiling for:

| GPU Architecture | NVCC Arch Flag | Compute Capability | Example GPUs                         |
|------------------|----------------|---------------------|--------------------------------------|
| Kepler           | `sm_30`        | 3.0                 | GTX 780, Tesla K20                   |
| Maxwell          | `sm_50`        | 5.0                 | GTX 970, Tesla M60                   |
| Pascal           | `sm_60`        | 6.0                 | GTX 1080, Tesla P100                 |
| Volta            | `sm_70`        | 7.0                 | Tesla V100                           |
| **Turing**       | `sm_75`        | 7.5                 | RTX 2080, Quadro RTX 6000           |
| Ampere (A100)    | `sm_80`        | 8.0                 | A100, RTX A6000                      |
| **Ampere (GA10x)** | `sm_86`      | 8.6                 | RTX 3090, 3080, 3070, A10            |
| Ada Lovelace     | `sm_89`        | 8.9                 | RTX 4090, 4080                       |
| Hopper           | `sm_90`        | 9.0                 | H100                                 |

In our case the arcithectures is defined by `Compute capability: 8.0` from the information printed on screen in the previous section. This is, our GPU is an A100. A general solution is to set `set(CMAKE_CUDA_ARCHITECTURES) all CACHE STRING "Target all architectures)`  making the code compatible with any card but this increases compilation time and also is slower at runtime. For modern GPUs you can do `set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 CACHE STRING "Target common modern architectures")`. Then we need to find the cuda libraries with

```bash
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
```

and adds the headers to the project. For `cmake>3.17` we would only need to define `project(GPUTools LANGUAGES CXX CUDA)` without the need to even include the cuda headers through `include_directories`. Finally we tell cmake which are the executables to be compiled, the libraries to link and the executable name. Then just the install instruction with the two executables.

To compile the two executables we need to run

```bash
rm -rf build && mkdir build && cd build
cmake ..
cmake --build .
```

which creates them in `build` directory. But, if you want to make them execcutable in all the system by installing them in `$HOME/.local/bin` you can do


```bash
rm -rf build && mkdir build && cd build
cmake \
  -DGPU_INFO_OUT_NAME=gpu_info \
  -DGPU_ALLOC_OUT_NAME=gpu_allocate \
  -DCMAKE_CUDA_ARCHITECTURES="70;75;80" \
  -DCMAKE_INSTALL_PREFIX=${HOME}/.local \
  ..
cmake --build .
cmake --install .
```

where `GPU_INFO_OUT_NAME` and `GPU_ALLOC_OUT_NAME` are the names of the executables (should we want to change them), `CMAKE_CUDA_ARCHITECTURES` are the GPU architectures we want to compile for. And `CMAKE_INSTALL_PREFIX` the install directory. In this last one even though we set it to `${HOME}/.local`, the binaries will be installed in `${HOME}/.local/bin` since we have the condition `RUNTIME destination bin` in the cmake.

## Bonus: nvitop

[nvitop](https://github.com/XuehaiPan/nvitop) is a great tool to monitor your GPU. I personally like it better than `nvidia-smi` which is the nvidia default "top". This tool comes in a python package so to install it it's best to create a new python virtual environment. We have covered this before in this blog so I am not going to extend. Just create a virtual envirionment in `$HOME/.venvs` called `nvitop` and pip install the tool:


```bash
mkdir -p $HOME/.venvs
python -m venv $HOME/.venvs/nvitop
$HOME/.venvs/nvitop/bin/pip install nvitop
```

Now you can exectute `nvitop` with

```bash
$HOME/.venvs/nvitop/bin/nvitop
```

or activating your environment and running `nvitop` in the command line. It is better to create a symlink to `${HOME}/.local/bin` directory so that the command line is in your `$PATH`:

```bash
ln -s  $HOME/.venvs/nvitop/bin/nvitop $HOME/.local/bin/nvitop
```

Seems that the python executable nvitop is not platform specific as the build wheels i find currently for the most recent version `1.5.1` in [PyPi](https://pypi.org/project/nvitop/1.5.1/#files) is `nvitop-1.5.1-py3-none-any.whl`. So this wheel should work for ARM64 (New generation of Grace Hoppers and Blackwell with GPU and CPU integrated) as well as for x86 CPU architectures.

Now we have the execs in `${HOME}/.local/bin` that should be in our `$PATH`.


## Conclusion

Hope you liked these tools I built, so far they have been useful for me. I will probably build more in the near future so I will post again about this.