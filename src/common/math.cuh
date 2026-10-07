#pragma once

namespace SieveOfEratosthenes
{
    __device__ __forceinline__ uint32_t FastMod(const uint32_t Value, const uint32_t Step, const uint32_t MLo,
                                                const uint32_t MHi){
        const uint32_t q = static_cast<uint32_t>(
            ((static_cast<uint64_t>(Value) * MHi) + __umulhi(Value, MLo)) >> 32);
        return Value - q * Step;
    }
}
