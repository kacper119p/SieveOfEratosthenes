# Sieve of Eratosthenes

A high-performance, GPU-accelerated implementation of the Sieve of Eratosthenes written in CUDA C++23.

## Features

- Wheel 30 Factorization
- Segmented Sieving
- Multi-Stream Pipelining
- Multi-Tiered Prime Sifting
- Asynchronous Data Transfer

## Usage

Run the compiled executable from the command line:

```bash
.\SieveOfEratosthenes.exe
```

The program checks for CUDA support, initializes device contexts, sieves all prime numbers up to $N = 70,000,000$, benchmarks initialization and GPU/CPU runtimes, and calculates the sum of all discovered primes.

## Building the project

Requires CMake 4.2+, NVIDIA CUDA Toolkit with C++23 support, and a CUDA-capable GPU.

To build the project, clone the repository using:

```bash
git clone 'https://github.com/kacper119p/SieveOfEratosthenes.git'
```

Enter the cloned repository's root folder:

```bash
cd ./SieveOfEratosthenes
```

Run the following commands to build the project (assumes `cmake` is in the `PATH`):

```bash
cmake -B $BuildDirectoryName

cmake --build $BuildDirectoryName --target SieveOfEratosthenes
```

## Tech Stack

- **Language**: CUDA C++ (CUDA Standard 23 / C++23)
- **Build System**: CMake 4.2+
- **Compiler**: NVIDIA CUDA Compiler (`nvcc`)