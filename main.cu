#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>

#ifdef NDEBUG
#define CUDA_CHECK(call) call
#else
#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t error = call;                                                                                      \
        if (error != cudaSuccess) {                                                                                    \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error));              \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)
#endif

namespace
{
    constexpr uint32_t BitsPerWord = 32u;
    constexpr uint32_t WheelSpan = 30u;
    constexpr uint32_t WheelBits = 8u;
    constexpr uint32_t SegmentWords = 2048u;
    constexpr uint32_t SegmentBits = SegmentWords * BitsPerWord;
    constexpr uint32_t ThreadsPerBlock = 256u;
    constexpr uint32_t WarpsPerBlock = ThreadsPerBlock / 32u;
    constexpr uint32_t WordsPerThread = SegmentWords / ThreadsPerBlock;
    constexpr uint32_t BaseThreads = 1024u;
    constexpr uint32_t BaseSieveWords = 1024u;
    constexpr uint32_t MaxBaseValue = 2u * BaseSieveWords * BitsPerWord + 1u;
    constexpr uint32_t NumBatches = 8u; // compute / D2H copy pipelining

    constexpr uint32_t NumMaskGroups = 4u;
    constexpr uint32_t MaskPeriod0 = 7u * 11u * 13u * 17u;
    constexpr uint32_t MaskPeriod1 = 19u * 23u * 29u;
    constexpr uint32_t MaskPeriod2 = 31u * 37u * 41u;
    constexpr uint32_t MaskPeriod3 = 43u * 47u * 53u;
    constexpr uint32_t MaskOffset0 = 0u;
    constexpr uint32_t MaskOffset1 = MaskOffset0 + MaskPeriod0;
    constexpr uint32_t MaskOffset2 = MaskOffset1 + MaskPeriod1;
    constexpr uint32_t MaskOffset3 = MaskOffset2 + MaskPeriod2;
    constexpr uint32_t MaskTotalWords = MaskOffset3 + MaskPeriod3;
    constexpr uint32_t MaxMaskPeriod = std::max({MaskPeriod0, MaskPeriod1, MaskPeriod2, MaskPeriod3});
    constexpr uint32_t FirstSievingPrime = 59u;

    // Prime classes for the segment sieve.
    //   [59, WarpPrimeLimit]            : one warp per prime, 4 lanes per wheel progression
    //   (WarpPrimeLimit, LargePrimeMin) : one thread per prime, loop over hits
    //   [LargePrimeMin, sqrt(N)]        : one thread per prime, at most one hit per progression per segment
    constexpr uint32_t WarpPrimeLimit = 1024u;
    constexpr uint32_t LargePrimeMin = SegmentBits / WheelBits;

    static_assert(BaseThreads % 32u == 0 && BaseThreads <= 1024u,
                  "base kernel block size must be warp multiple <= 1024");
    static_assert(SegmentWords % ThreadsPerBlock == 0, "segment must split evenly across threads");
    static_assert(ThreadsPerBlock % 32u == 0, "block size must be a multiple of the warp size");
    static_assert(SegmentWords % 4u == 0, "segment must be a multiple of 4 words for uint4 stores");
    static_assert(ThreadsPerBlock >= 64u, "wheel table is filled by the first 64 threads");
    static_assert(MaskPeriod0 > ThreadsPerBlock && MaskPeriod1 > ThreadsPerBlock && MaskPeriod2 > ThreadsPerBlock &&
                  MaskPeriod3 > ThreadsPerBlock, "mask index wrap assumes period > block size");
    static_assert(FirstSievingPrime <= WarpPrimeLimit && WarpPrimeLimit < LargePrimeMin, "invalid prime classes");

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

    constexpr uint32_t Segment0Fixup = []
    {
        uint32_t bits = 0;
        for (const uint32_t p: {7u, 11u, 13u, 17u, 19u, 23u, 29u, 31u, 37u, 41u, 43u, 47u, 53u})
        {
            bits |= 1u << ValueToBit(p);
        }
        return bits;
    }();
    static_assert(ValueToBit(53u) < BitsPerWord, "mask primes must lie in word 0");

    struct MaskGroup
    {
        uint32_t Offset;
        uint32_t Period;
        uint32_t Count;
        uint32_t Primes[4];
    };

    struct MaskGroupTable
    {
        MaskGroup Groups[NumMaskGroups];
    };

    constexpr MaskGroupTable MaskGroups = {
        {
            {MaskOffset0, MaskPeriod0, 4u, {7u, 11u, 13u, 17u}},
            {MaskOffset1, MaskPeriod1, 3u, {19u, 23u, 29u, 0u}},
            {MaskOffset2, MaskPeriod2, 3u, {31u, 37u, 41u, 0u}},
            {MaskOffset3, MaskPeriod3, 3u, {43u, 47u, 53u, 0u}},
        }
    };

    __device__ uint32_t DMasks[MaskTotalWords];

    struct PrimeCounts
    {
        uint32_t WarpEnd; // entries [0, warpEnd)          : warp mode
        uint32_t LargeBegin; // entries [warpEnd, largeBegin) : thread mode, multi-hit
        uint32_t Total; // entries [largeBegin, total)    : thread mode, single-hit
    };

    __global__ void __launch_bounds__(256)
    InitMasksKernel(const MaskGroupTable Table){
        const MaskGroup& g = Table.Groups[blockIdx.y];
        for (uint32_t w = blockIdx.x * blockDim.x + threadIdx.x; w < g.Period; w += gridDim.x * blockDim.x)
        {
            uint32_t word = 0;
            for (uint32_t b = 0; b < BitsPerWord; ++b)
            {
                const uint32_t i = w * BitsPerWord + b;
                const uint32_t v = WheelSpan * (i / WheelBits) + WheelResidue(i % WheelBits);
                for (uint32_t k = 0; k < g.Count; ++k)
                {
                    if (v % g.Primes[k] == 0)
                    {
                        word |= 1u << b;
                        break;
                    }
                }
            }
            DMasks[g.Offset + w] = word;
        }
    }

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

    __device__ __forceinline__ uint32_t OffsetToResidue(const uint32_t Residue, const uint32_t Step,
                                                        const uint32_t LowerMod){
        return Residue >= LowerMod ? Residue - LowerMod : Residue + Step - LowerMod;
    }

    __device__ __forceinline__ uint32_t FastMod(const uint32_t Value, const uint32_t Step, const uint32_t MLo,
                                                const uint32_t MHi){
        const uint32_t q = static_cast<uint32_t>(
            ((static_cast<uint64_t>(Value) * MHi) + __umulhi(Value, MLo)) >> 32);
        return Value - q * Step;
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

    __device__ __forceinline__ uint32_t WheelByte(const uint2 Packed, const uint32_t J){
        return ((J < 4u ? Packed.x : Packed.y) >> (8u * (J & 3u))) & 0xFFu;
    }

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
            acc[0] = (acc[0] | 1u) & ~Segment0Fixup;
        }

#pragma unroll
        for (uint32_t j = 0; j < WordsPerThread; ++j)
        {
            segment[tid + j * ThreadsPerBlock] = acc[j];
        }

        const PrimeCounts counts = *Counts;
        __syncthreads();

        {
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
} // namespace

int main(){
    const auto programStartTime = std::chrono::system_clock::now();
    cudaFree(nullptr); // Initializes CUDA runtime.
    const auto processingStartTime = std::chrono::system_clock::now();

    cudaEvent_t gpuStartEvent;
    cudaEventCreate(&gpuStartEvent);

    cudaEvent_t gpuFinishEvent;
    cudaEventCreate(&gpuFinishEvent);

    cudaStream_t streams[NumBatches];
    cudaEvent_t batchDone[NumBatches];
    for (uint32_t k = 0; k < NumBatches; ++k)
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[k], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&batchDone[k], cudaEventDisableTiming));
    }
    cudaEvent_t masksReady;
    cudaEvent_t primesReady;
    CUDA_CHECK(cudaEventCreateWithFlags(&masksReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&primesReady, cudaEventDisableTiming));

    cudaEventRecord(gpuStartEvent, streams[0]);

    constexpr uint32_t n = 70'000'000u;
    const uint32_t maxBaseNum = static_cast<uint32_t>(std::sqrt(n)) + 1u;

    constexpr uint32_t totalBits = (n / WheelSpan + 1u) * WheelBits;
    constexpr uint32_t totalWords = (totalBits + BitsPerWord - 1) / BitsPerWord;
    constexpr uint32_t totalSegments = (totalWords + SegmentWords - 1) / SegmentWords;
    constexpr uint32_t segmentsPerBatch = (totalSegments + NumBatches - 1) / NumBatches;

    printf("Configuring GPU Sieve for N = %llu\n", static_cast<unsigned long long>(n));
    printf("Total VRAM required: ~%.2f MB\n", static_cast<float>(totalWords * 4) / (1024 * 1024));

    if (maxBaseNum > MaxBaseValue)
    {
        fprintf(stderr, "N too large: base sieve supports values up to %u\n", MaxBaseValue);
        return EXIT_FAILURE;
    }
    const uint32_t maxBasePrimes =
            static_cast<uint32_t>(1.26 * maxBaseNum / std::log(std::max(maxBaseNum, 2u))) + 1u;

    uint4* dPrimes;
    PrimeCounts* dCounts;
    uint32_t* dFinalSieve;
    CUDA_CHECK(cudaMalloc(&dPrimes, maxBasePrimes * sizeof(uint4)));
    CUDA_CHECK(cudaMalloc(&dCounts, sizeof(PrimeCounts)));
    CUDA_CHECK(cudaMalloc(&dFinalSieve, static_cast<size_t>(totalSegments) * SegmentWords * sizeof(uint32_t)));

    // Masks and base primes are independent: build them concurrently.
    CUDA_CHECK(cudaStreamWaitEvent(streams[1], gpuStartEvent));
    InitMasksKernel<<<dim3((MaxMaskPeriod + 255u) / 256u, NumMaskGroups), 256, 0, streams[1]>>>(MaskGroups);
    CUDA_CHECK(cudaEventRecord(masksReady, streams[1]));

    BasePrimesKernel<<<1, BaseThreads, 0, streams[0]>>>(dPrimes, dCounts, maxBaseNum);
    CUDA_CHECK(cudaEventRecord(primesReady, streams[0]));

    for (uint32_t k = 0; k < NumBatches; ++k)
    {
        const uint32_t first = k * segmentsPerBatch;
        const uint32_t last = std::min(first + segmentsPerBatch, totalSegments);
        if (first >= last)
        {
            continue;
        }
        CUDA_CHECK(cudaStreamWaitEvent(streams[k], masksReady));
        CUDA_CHECK(cudaStreamWaitEvent(streams[k], primesReady));
        MainKernel<<<last - first, ThreadsPerBlock, 0, streams[k]>>>(dFinalSieve, dPrimes, dCounts, first);
    }
    cudaStreamQuery(streams[NumBatches - 1]); // flush the launch queue (WDDM batching) before blocking host work

    uint32_t* finalSieve;
    CUDA_CHECK(cudaMallocHost(&finalSieve, totalWords * sizeof(uint32_t)));

    for (uint32_t k = 0; k < NumBatches; ++k)
    {
        const uint32_t firstWord = std::min(k * segmentsPerBatch * SegmentWords, totalWords);
        const uint32_t lastWord = std::min((k + 1) * segmentsPerBatch * SegmentWords, totalWords);
        if (firstWord < lastWord)
        {
            CUDA_CHECK(cudaMemcpyAsync(finalSieve + firstWord, dFinalSieve + firstWord,
                (lastWord - firstWord) * sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[k]));
        }
        if (k != 0)
        {
            CUDA_CHECK(cudaEventRecord(batchDone[k], streams[k]));
            CUDA_CHECK(cudaStreamWaitEvent(streams[0], batchDone[k]));
        }
    }

    cudaEventRecord(gpuFinishEvent, streams[0]);
    cudaEventSynchronize(gpuFinishEvent);

    const auto programTotalTime = std::chrono::system_clock::now() - programStartTime;
    std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(programTotalTime).count()
            << " ms\n";
    const auto processingTime = std::chrono::system_clock::now() - processingStartTime;
    std::cout << "Total time (without CUDA initialization): "
            << std::chrono::duration_cast<std::chrono::milliseconds>(processingTime).count() << " ms\n";


    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, gpuStartEvent, gpuFinishEvent);
    printf("GPU Execution Time: %.3f ms\n", milliseconds);

    uint64_t sum = 0;
    for (const uint32_t p: {2u, 3u, 5u})
    {
        if (p <= n)
        {
            sum += p;
        }
    }
    for (uint64_t w = 0; w < totalWords; ++w)
    {
        uint32_t bits = ~finalSieve[w];
        while (bits)
        {
            const uint64_t i = w * BitsPerWord + static_cast<uint64_t>(std::countr_zero(bits));
            bits &= bits - 1;
            const uint64_t p = WheelSpan * (i / WheelBits) + WheelResidue(static_cast<uint32_t>(i % WheelBits));
            if (p > 5 && p <= n)
            {
                sum += p;
            }
        }
    }

    printf("Sum of primes up to %llu: %llu\n", static_cast<unsigned long long>(n), (sum));

    cudaFreeHost(finalSieve);
    cudaFree(dPrimes);
    cudaFree(dCounts);
    cudaFree(dFinalSieve);
    for (uint32_t k = 0; k < NumBatches; ++k)
    {
        cudaStreamDestroy(streams[k]);
        cudaEventDestroy(batchDone[k]);
    }
    cudaEventDestroy(masksReady);
    cudaEventDestroy(primesReady);
    cudaEventDestroy(gpuStartEvent);
    cudaEventDestroy(gpuFinishEvent);

    return 0;
}
