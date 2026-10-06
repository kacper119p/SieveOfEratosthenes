#pragma once
#include "common/utility.cuh"

namespace SieveOfEratosthenes
{
    __global__ void MainKernel(uint32_t*__restrict__ BitArray, const uint4*__restrict__ Primes,
                               const PrimeCounts*__restrict__ Counts, uint32_t FirstSegment);
}
