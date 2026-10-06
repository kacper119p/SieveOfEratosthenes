#include "initializeWheelMaskKernel.cuh"

#include "common/utility.cuh"

__global__ void __launch_bounds__(256)
InitMasksKernel(const MaskGroupTable Table){
    constexpr uint32_t maxGroupPrimes = 4u;
    __shared__ uint32_t residues[maxGroupPrimes][WheelBits];

    const MaskGroup& g = Table.Groups[blockIdx.y];
    if (threadIdx.x < maxGroupPrimes * WheelBits)
    {
        const uint32_t k = threadIdx.x / WheelBits;
        const uint32_t j = threadIdx.x % WheelBits;
        residues[k][j] = k < g.Count ? ValueToBit(g.Primes[k] * WheelResidue(j)) : 0u;
    }
    __syncthreads();

    for (uint32_t w = blockIdx.x * blockDim.x + threadIdx.x; w < g.Period; w += gridDim.x * blockDim.x)
    {
        const uint32_t firstBit = w * BitsPerWord;
        uint32_t word = 0;
#pragma unroll
        for (uint32_t k = 0; k < maxGroupPrimes; ++k)
        {
            if (k < g.Count)
            {
                const uint32_t step = WheelBits * g.Primes[k];
                const uint32_t base = firstBit % step;
#pragma unroll
                for (uint32_t j = 0; j < WheelBits; ++j)
                {
                    const uint32_t r = residues[k][j];
                    const uint32_t off = r >= base ? r - base : r + step - base;
                    word |= off < BitsPerWord ? 1u << off : 0u;
                }
            }
        }
        DMasks[g.Offset + w] = word;
    }
}