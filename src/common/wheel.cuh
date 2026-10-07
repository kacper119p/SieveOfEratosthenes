#pragma once
#include "constants.cuh"

namespace SieveOfEratosthenes
{
    struct MaskGroup
    {
        uint32_t Offset;
        uint32_t Period;
        uint32_t Count;
        uint32_t Primes[4];
    };

    struct MaskGroupTable
    {
        MaskGroup Groups[MaskGroupsCount];
    };

    constexpr MaskGroupTable MaskGroups = {
        {
            {MaskOffset0, MaskPeriod0, 4u, {7u, 11u, 13u, 17u}},
            {MaskOffset1, MaskPeriod1, 3u, {19u, 23u, 29u, 0u}},
            {MaskOffset2, MaskPeriod2, 3u, {31u, 37u, 41u, 0u}},
            {MaskOffset3, MaskPeriod3, 3u, {43u, 47u, 53u, 0u}},
        }
    };

    inline __device__ uint32_t DMasks[MaskTotalWords];

    __device__ __forceinline__ constexpr uint32_t OffsetToResidue(const uint32_t Residue, const uint32_t Step,
                                                                  const uint32_t LowerMod){
        return Residue >= LowerMod ? Residue - LowerMod : Residue + Step - LowerMod;
    }

    template<uint32_t Period, uint32_t Offset>
    __device__ __forceinline__ void ApplyMask(uint32_t (&Acc)[WordsPerThread], const uint32_t FirstWord){
        uint32_t idx = FirstWord % Period;
#pragma unroll
        for (uint32_t j = 0; j < WordsPerThread; ++j)
        {
            Acc[j] |= __ldg(&DMasks[Offset + idx]);
            idx += ThreadsPerBlock;
            if (idx >= Period)
            {
                idx -= Period;
            }
        }
    }

    __device__ __forceinline__ constexpr uint32_t WheelByte(const uint2 Packed, const uint32_t J){
        return ((J < 4u ? Packed.x : Packed.y) >> (8u * (J & 3u))) & 0xFFu;
    }

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
}
