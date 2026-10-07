#pragma once
#include "constants.cuh"

namespace SieveOfEratosthenes
{
    struct PrimeCounts
    {
        uint32_t WarpEnd; // entries [0, warpEnd)             : warp mode
        uint32_t LargeBegin; // entries [warpEnd, largeBegin) : thread mode, multi-hit
        uint32_t Total; // entries [largeBegin, total)        : thread mode, single-hit
    };

    struct alignas(16) BasePrime
    {
        uint32_t StepAndClass;
        uint32_t MagicLo;
        uint32_t MagicHi;
        uint32_t SquareIdx;
    };

    __host__ __device__ __forceinline__ constexpr uint32_t BasePrimeStep(const BasePrime Prime){
        return Prime.StepAndClass & ~7u;
    }

    __host__ __device__ __forceinline__ constexpr uint32_t BasePrimeValue(const BasePrime Prime){
        return Prime.StepAndClass >> 3;
    }

    __host__ __device__ __forceinline__ constexpr uint32_t BasePrimeWheelClass(const BasePrime Prime){
        return Prime.StepAndClass & 7u;
    }

    __device__ __forceinline__ BasePrime LoadBasePrime(const BasePrime*__restrict__ Prime){
        const uint4 v = __ldg(reinterpret_cast<const uint4 *>(Prime));
        return {v.x, v.y, v.z, v.w};
    }

    static_assert(sizeof(BasePrime) == 16);
}
