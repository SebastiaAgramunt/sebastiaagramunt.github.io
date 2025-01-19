---
title: C++ Basics
author: sebastia
date: 2025-01-19 6:02:00 +0800
categories: [C++]
tags: [computer science]
pin: true
toc: true
render_with_liquid: false
math: true
---


This post is about C++ and the way we can compile a C++ program. I remember in the early days of my PhD in physics that I used to write all my code into one `main.cpp` and compile it into an executable. Over the months and years I understood that it is more efficient to write in different files serving different purposes and compile and link your libraries. I write this post as a reminder for me but also for these people that are starting in C++ as a guide.

# TLDR
```bash 

# step by step compilation
g++ -E main.cpp -o main.i # preprocessing
g++ -S main.cpp -o main.s # compilation: assembly
g++ -c main.s -o main.o   # compilation: object file
g++ main.o -o hello_world # linking

# compilation all at once (one file)
g++ main.cpp -o hello_world
```


## Compilation

Compilation is a process by which we translate from human language (code) to machine language, or instructions that the machine can understand. There are several steps, let's explain it using a file example saved as `main.cpp`

```cpp
#include <iostream> 

int main() { 
	std::cout << "Hello, World!" << std::endl; return 0;
}
```

## Installing the compiler

We will isntall `gcc` (GNU compiler collection), the compiler for C and `g++`, the compiler for C++. In linux install with

```bash
sudo apt update
sudo apt install gcc g++
```

In MacOS install using brew:

```bash
brew install gcc
```

The brew formulae alrady has `gcc` and `g++`, check both compilers have been successfully installed with

```bash
gcc --version
g++ --version
```

### Preprocessing

The preprocessing step has several functions:
* File inclusion: When encountering `#include`, the preprocessor replaces it with the actual content of the included file
* Macro expansion: Macros defined with `#define` are replaced wherever they appear in the code. E.g. `#define PI 3.14159; double area = PI * r * r;` is after preprocessing `double area = 3.14159 * r * r;` on the variable `area`.
* Conditional compilation: IFDEFs are included/not included in this step, for instance `#ifdef DEBUG std::cout << "Debugging enabled" << std::endl; #endif`. if `DEBUG` is added in the flag `-DDEBUG` it will add these lines.
* Removes comments: For instance comments in `//this is a comment` won't be included after the source file is processed
`
The command to generate a `main.i` preprocessed file is:

```bash
g++ -E main.cpp -o main.i
```

Try to take a look at the file. There's many things modified. The preprocessing is useful to add all the necessary code in the preprocessed file, also the compiler will get a unique file with code to compile to an object. It is useful to add or delete configurations and other directives using `ifdef` and other commands, this way we can compile for `DEBUG` or maybe `CUDA` only code.


### Compilation: Generate assembly code

This step instructs the compiler to translate the preprocessed C++ source file (`main.i`) into **assembly code**, which is a low-level, human-readable representation of the instructions for the target machine's CPU.

The following command precompiles and then generates the assembly code for your specific CPU:

```bash
g++ -S main.cpp -o main.s
```

The compiler reads the preprocessed file and checks for errors in variables, types correctness, compliance of the C++ standard. Then it constructs the abstract syntax tree, that is a tree structure of the program's execution. Then it optimizes the instructions by removing unnecessary operations or simplifying operations. Finally writes the hardware specific instructions in human readable code to the file.

### Compilation: Generate object file

The following step is produced by the assembler, that translates assembly code produced in the previous step to object code or compiled code that the CPU can execute. However it is not yet a complete executable.

The code for this step is

```bash
g++ -c main.s -o main.o
```

We create a source binary file that is not linked to any executable. This is useful for modular projects with multiple source files. 

### Linking: Executable generation

Now that we have machine interpretable code we need to link all the objects to generate an executable that can be called from the machine and produces an output or calculation. In our simple case

```bash
g++ main.o -o hello_world
```

The linking phase is simple here because we are just dealing with a single object file but in general we have many. We will see how to deal with that in a follow-up post. What we end up having from the command is a `hello_world` file that can be executed as `./hello_world` to output Hello World! in your screen.