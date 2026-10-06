#pragma once
#include "constants.cuh"

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
