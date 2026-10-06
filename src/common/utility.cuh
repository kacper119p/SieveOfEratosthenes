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

    __host__ __device__ constexpr uint32_t WheelResidue(const uint32_t Bit){
        constexpr uint64_t packed = 1ull | 7ull << 5 | 11ull << 10 | 13ull << 15 | 17ull << 20 | 19ull << 25 |
                                    23ull << 30 | 29ull << 35;
        return static_cast<uint32_t>(packed >> (5u * Bit)) & 31u;
    }

    __host__ __device__ constexpr uint32_t ResidueToBit(const uint32_t Residue){
        switch (Residue)
        {
            case 1u: return 0u;
            case 7u: return 1u;
            case 11u: return 2u;
            case 13u: return 3u;
            case 17u: return 4u;
            case 19u: return 5u;
            case 23u: return 6u;
            default: return 7u; // 29
        }
    }

    __host__ __device__ constexpr uint32_t ValueToBit(const uint32_t V){
        return WheelBits * (V / WheelSpan) + ResidueToBit(V % WheelSpan);
    }

    __host__ __device__ constexpr uint32_t WheelOffset(const uint32_t C, const uint32_t J){
        const uint32_t t = WheelResidue(C) * WheelResidue(J);
        return WheelBits * (t / WheelSpan) + ResidueToBit(t % WheelSpan);
    }

    __host__ __device__ constexpr uint32_t CalculateFirstSegmentFix(){
        uint32_t bits = 0;
        for (const uint32_t p: {7u, 11u, 13u, 17u, 19u, 23u, 29u, 31u, 37u, 41u, 43u, 47u, 53u})
        {
            bits |= 1u << ValueToBit(p);
        }
        return ~bits;
    }

    __device__ __forceinline__ uint32_t FastMod(const uint32_t Value, const uint32_t Step, const uint32_t MLo,
                                                const uint32_t MHi){
        const uint32_t q = static_cast<uint32_t>(
            ((static_cast<uint64_t>(Value) * MHi) + __umulhi(Value, MLo)) >> 32);
        return Value - q * Step;
    }
}
