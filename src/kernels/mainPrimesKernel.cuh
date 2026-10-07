#pragma once
#include "common/basePrime.cuh"
#include "common/math.cuh"

namespace SieveOfEratosthenes
{
    __global__ void MainPrimesKernel(uint32_t*__restrict__ BitArray, const BasePrime*__restrict__ BasePrimes,
                               const PrimeCounts*__restrict__ Counts, uint32_t FirstSegment);
}
