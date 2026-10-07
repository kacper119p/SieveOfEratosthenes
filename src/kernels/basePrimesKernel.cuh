#pragma once
#include "common/basePrime.cuh"
#include "common/math.cuh"

namespace SieveOfEratosthenes
{
    __global__ void BasePrimesKernel(BasePrime*__restrict__ BasePrimes, PrimeCounts*__restrict__ Counts, uint32_t MaxValue);
}
