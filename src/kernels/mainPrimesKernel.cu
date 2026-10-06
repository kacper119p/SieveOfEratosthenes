#include "mainPrimesKernel.cuh"

#include "common/wheelMask.cuh"

namespace
{
    template<bool SingleHit>
        __device__ __forceinline__ void SieveThreadPrimes(uint32_t*__restrict__ Segment, const uint4*__restrict__ Primes,
                                                          const uint2*__restrict__ Wheel, const uint32_t Begin,
                                                          const uint32_t End, const uint32_t SegmentStartIdx,
                                                          const uint32_t SegmentEndIdx){
        for (uint32_t i = Begin + threadIdx.x; i < End; i += ThreadsPerBlock)
        {
            const uint4 pr = __ldg(&Primes[i]);
            if (pr.w >= SegmentEndIdx)
            {
                break;
            }

            const uint32_t step = pr.x & ~7u;
            const uint32_t p = pr.x >> 3;
            const uint32_t lower = pr.w > SegmentStartIdx ? pr.w : SegmentStartIdx;
            const uint32_t lowerMod = FastMod(lower, step, pr.y, pr.z);
            const uint32_t localLower = lower - SegmentStartIdx;
            const uint32_t kx8 = (p / WheelSpan) * WheelBits;
            const uint2 wheel = Wheel[pr.x & 7u];

#pragma unroll
            for (uint32_t j = 0; j < WheelBits; ++j)
            {
                const uint32_t residue = kx8 * WheelResidue(j) + WheelByte(wheel, j);
                uint32_t l = localLower + OffsetToResidue(residue, step, lowerMod);
                if constexpr (SingleHit)
                {
                    if (l < SegmentBits)
                    {
                        atomicOr(&Segment[l >> 5], 1u << (l & 31u));
                    }
                } else
                {
                    for (; l < SegmentBits; l += step)
                    {
                        atomicOr(&Segment[l >> 5], 1u << (l & 31u));
                    }
                }
            }
        }
    }
} // namespace

__global__ void __launch_bounds__(ThreadsPerBlock)
MainKernel(uint32_t*__restrict__ BitArray, const uint4*__restrict__ Primes,
           const PrimeCounts*__restrict__ Counts, const uint32_t FirstSegment){
    __shared__ alignas(16) uint32_t segment[SegmentWords];
    __shared__ alignas(8) uint8_t wheel[WheelBits * WheelBits];

    const uint32_t tid = threadIdx.x;
    const uint32_t segmentIdx = FirstSegment + blockIdx.x;
    const uint32_t segmentStartIdx = segmentIdx * SegmentBits;
    const uint32_t segmentEndIdx = segmentStartIdx + SegmentBits;

    if (tid < WheelBits * WheelBits)
    {
        wheel[tid] = static_cast<uint8_t>(WheelOffset(tid >> 3, tid & 7u));
    }

    uint32_t acc[WordsPerThread];
#pragma unroll
    for (uint32_t j = 0; j < WordsPerThread; ++j)
    {
        acc[j] = 0u;
    }
    const uint32_t firstWord = segmentIdx * SegmentWords + tid;
    ApplyMask<MaskPeriod0, MaskOffset0>(acc, firstWord);
    ApplyMask<MaskPeriod1, MaskOffset1>(acc, firstWord);
    ApplyMask<MaskPeriod2, MaskOffset2>(acc, firstWord);
    ApplyMask<MaskPeriod3, MaskOffset3>(acc, firstWord);

    if (segmentIdx == 0 && tid == 0)
    {
        acc[0] = (acc[0] | 1u) & CalculateFirstSegmentFix();
    }

#pragma unroll
    for (uint32_t j = 0; j < WordsPerThread; ++j)
    {
        segment[tid + j * ThreadsPerBlock] = acc[j];
    }

    const PrimeCounts counts = *Counts;
    __syncthreads(); {
        const uint32_t lane = tid & 31u;
        const uint32_t warp = tid >> 5;
        const uint32_t wheelJ = lane & 7u;
        const uint32_t residueJx8 = WheelBits * WheelResidue(wheelJ);
        const uint32_t phase = lane >> 3;

        for (uint32_t i = warp; i < counts.WarpEnd; i += WarpsPerBlock)
        {
            const uint4 pr = __ldg(&Primes[i]);
            if (pr.w >= segmentEndIdx)
            {
                break;
            }

            const uint32_t step = pr.x & ~7u;
            const uint32_t p = pr.x >> 3;
            const uint32_t lower = pr.w > segmentStartIdx ? pr.w : segmentStartIdx;
            const uint32_t lowerMod = FastMod(lower, step, pr.y, pr.z);
            const uint32_t residue = (p / WheelSpan) * residueJx8 + wheel[(pr.x & 7u) * WheelBits + wheelJ];

            const uint32_t start = (lower - segmentStartIdx) + OffsetToResidue(residue, step, lowerMod) +
                                   phase * step;
            const uint32_t mask = 1u << (start & 31u);
            for (uint32_t w = start >> 5; w < SegmentWords; w += p)
            {
                atomicOr(&segment[w], mask);
            }
        }
    }

    const uint2* wheel2 = reinterpret_cast<const uint2 *>(wheel);
    SieveThreadPrimes<false>(segment, Primes, wheel2, counts.WarpEnd, counts.LargeBegin, segmentStartIdx,
                             segmentEndIdx);
    SieveThreadPrimes<true>(segment, Primes, wheel2, counts.LargeBegin, counts.Total, segmentStartIdx,
                            segmentEndIdx);
    __syncthreads();

    const uint4* segment4 = reinterpret_cast<const uint4 *>(segment);
    uint4* out4 = reinterpret_cast<uint4 *>(BitArray) + segmentIdx * (SegmentWords / 4u);
    for (uint32_t i = tid; i < SegmentWords / 4u; i += ThreadsPerBlock)
    {
        out4[i] = segment4[i];
    }
}
