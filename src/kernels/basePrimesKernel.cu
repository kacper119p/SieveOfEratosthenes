#include "basePrimesKernel.cuh"

__global__ void __launch_bounds__(BaseThreads)
BasePrimesKernel(uint4*__restrict__ Primes, PrimeCounts*__restrict__ Counts, const uint32_t MaxValue){
    __shared__ uint32_t baseSieve[BaseSieveWords];
    __shared__ uint32_t warpOffsets[32];
    __shared__ uint32_t runningCount;
    __shared__ uint32_t iterTotal;
    __shared__ uint32_t warpModeCount;
    __shared__ uint32_t mediumCount;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t numWarps = blockDim.x >> 5;

    const uint32_t numOdd = (MaxValue - 1u) / 2u;
    const uint32_t wordCount = (numOdd + BitsPerWord - 1u) / BitsPerWord;

    const uint32_t limit = static_cast<uint32_t>(sqrtf(static_cast<float>(MaxValue)));
    for (uint32_t w = tid; w < wordCount; w += blockDim.x)
    {
        uint32_t wordMask = 0;
        const uint32_t wordStartNum = 2u * (w * BitsPerWord) + 3u;
        const uint32_t wordEndNum = wordStartNum + 2u * BitsPerWord;

        for (uint32_t i = 3; i <= limit; i += 2)
        {
            const uint32_t pSq = i * i;
            if (pSq >= wordEndNum)
            {
                break;
            }

            uint32_t j = pSq;
            if (j < wordStartNum)
            {
                j = (wordStartNum + i - 1u) / i * i;
                if ((j & 1u) == 0)
                {
                    j += i;
                }
            }
            uint32_t bitIdx = (j - wordStartNum) / 2u;
            while (bitIdx < BitsPerWord)
            {
                wordMask |= 1u << bitIdx;
                bitIdx += i;
            }
        }
        baseSieve[w] = wordMask;
    }
    if (tid == 0)
    {
        runningCount = 0;
        warpModeCount = 0;
        mediumCount = 0;
    }
    __syncthreads();

    for (uint32_t base = 0; base < numOdd; base += blockDim.x)
    {
        const uint32_t i = base + tid;
        const uint32_t p = 2u * i + 3u;
        const bool keep = i < numOdd && p >= FirstSievingPrime &&
                          !((baseSieve[i / BitsPerWord] >> (i % BitsPerWord)) & 1u);

        const uint32_t ballot = __ballot_sync(0xFFFFFFFFu, keep);
        if (lane == 0)
        {
            warpOffsets[warp] = __popc(ballot);
        }
        __syncthreads();

        if (warp == 0)
        {
            const uint32_t v = lane < numWarps ? warpOffsets[lane] : 0u;
            uint32_t incl = v;
#pragma unroll
            for (uint32_t o = 1; o < 32u; o <<= 1)
            {
                const uint32_t t = __shfl_up_sync(0xFFFFFFFFu, incl, o);
                if (lane >= o)
                {
                    incl += t;
                }
            }
            if (lane < numWarps)
            {
                warpOffsets[lane] = incl - v;
            }
            if (lane == 31u)
            {
                iterTotal = incl;
            }
        }
        __syncthreads();

        if (keep)
        {
            const uint32_t idx = runningCount + warpOffsets[warp] + __popc(ballot & ((1u << lane) - 1u));
            const uint32_t step = WheelBits * p;
            const uint64_t m = (~0ULL) / step + 1u;
            const uint32_t wheelClass = ResidueToBit(p % WheelSpan);
            const uint32_t sqIdx = ValueToBit(p * p);
            Primes[idx] = make_uint4(step | wheelClass, static_cast<uint32_t>(m), static_cast<uint32_t>(m >> 32),
                                     sqIdx);
            if (p <= WarpPrimeLimit)
            {
                atomicAdd(&warpModeCount, 1u);
            } else if (p < LargePrimeMin)
            {
                atomicAdd(&mediumCount, 1u);
            }
        }
        __syncthreads();
        if (tid == 0)
        {
            runningCount += iterTotal;
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        Counts->WarpEnd = warpModeCount;
        Counts->LargeBegin = warpModeCount + mediumCount;
        Counts->Total = runningCount;
    }
}
