---
title: Nvidia-GPU Performance
author: sebastia
date: 2025-07-13 12:35:00 +0800
categories: [C++, CUDA]
tags: [computer science, GPU]
pin: true
toc: true
render_with_liquid: false
math: true
---

GPUs or Graphical Processing Units have become essential in high performance computing these days. They are efficient hardware that can parallelize small calculations. For instance, in Machine Learning (ML) and Artificial Intelligence (AI), GPUs are ubiquitous, as almost all operations are matrix multiplications, convolutions, max pooling... Operations that can be paralellized easily. However, GPUs are not suitable for any kind of calculation, in this post we will understand when it pays off to bring the calculation to GPU for a very simple example.

I thank [Lambda AI](https://lambda.ai/) for providing free credit to run the experiments described in the post. Throughout this post we will be using `gpu_1x_a100_sxm4`, an Ampere 100 GPU. As always the code can be found in my [github repository]([Code](https://github.com/SebastiaAgramunt/blogging-code/tree/main/cuda-performance)).

## The problem

Given two arrays of `float`s of length `N`, their sum `m`  times. This is, for each element `a` in `\textbf{a}` and each element `b` of vector `\textbf{b}` we make the sum `m` times. A total of `N \times m` floating point sums. We can write the CPU code in C++ as

```cpp
// Create host a, b, c
std::vector<float> h_a(N), h_b(N), h_c_cpu(N);

// Fill a and b with random floats 0 to 1
for (int i = 0; i < N; ++i) {
    h_a[i] = static_cast<float>(std::rand()) / RAND_MAX;
    h_b[i] = static_cast<float>(std::rand()) / RAND_MAX;
}

// Sum each element of the arrays i...N, a total of m times
for (int i = 0; i < N; ++i) {
    double acc = 0;
    double s = h_a[i] + h_b[i];
    for (int j = 0; j < m; ++j) {
        acc = acc + s;
    }
    h_c_cpu[i] = acc;
}
```

I agree the code seems a bit useless, why don't we do...

```cpp
double s = (h_a[i] + h_b[1]) * m;
```

So that it would be only `N` additions?. We want to explicitely make the sum `m` times per array element. Also we want to avoid adding constants to this loop, e.g. `acc = acc + 1`. That would make the compiler to optimize the code and run much faster, the point of this calculation is to perform `m` sums for each array element.

## GPU Characteristics

As mentioned above, we will be using `gpu_1x_a100_sxm4` from [Lambda AI](https://lambda.ai/). Running `gpu_info` command line tool described in another post...


## GPU kernel for vector addition

This is one of the simplests CUDA kernels one can write, but before starting to explain it, if you are new into CUDA programming, please take a look at the [CUDA programming model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/). Make sure you understand what is a [thread](https://modal.com/gpu-glossary/device-software/thread), a [block](https://modal.com/gpu-glossary/device-software/thread-block) and a [grid](https://modal.com/gpu-glossary/device-software/thread-block-grid). A good visualization of the model can be found [here](https://harmanani.github.io/classes/csc447/Notes/Lecture15.pdf). Another great resource is [Programming Massively Parallel Processors: A Hands-on Approach](https://www.goodreads.com/book/show/7659954-programming-massively-parallel-processors), this is my reference book for GPU programming.

Our CUDA kernel for addition of two vectors hould look like this:

```cpp
template<typename T>
__global__ void vectorAdd(const T* a, const T* b, T* c, int n, int m=1) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
        T s = a[idx] + b[idx];
        T acc = T(0);
        for (int j = 0; j < m; ++j) {
            acc = acc + s;
        }
        c[idx] = acc;
}
```

We define a template to adapt in case we want to test for other types like `int` or `double` instead of `float`. Every CUDA kernel has to start with `__global__`, that tells the `nvcc` (the compiler) that this is code to be executed at the GPU. Then, inside the function we have the global index of the thread, `idx`. We will launch 1D blocks in one grid so we are working only with the `x` dimension, in this case the `idx` can be written as the block dimension times the block index plus the thread index within the block, `blockDim.x * blockIdx.x + threadIdx.x;`. Then we have the conditional on the global thread index, a condition that, even though not mandatory, it is very recommended to add; the index cannot exeed the total number of elements of the array. If the thread is larger, no worries, we just don't do anything and we leave the function for that thread.