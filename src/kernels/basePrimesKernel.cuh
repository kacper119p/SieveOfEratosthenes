#pragma once
#include "common/utility.cuh"

__global__ void BasePrimesKernel(uint4*__restrict__ Primes, PrimeCounts*__restrict__ Counts, uint32_t MaxValue);