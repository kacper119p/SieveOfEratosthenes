#include <algorithm>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <new>

#include "common/check.cuh"
#include "common/constants.cuh"
#include "common/utility.cuh"
#include "common/wheelMask.cuh"
#include "kernels/basePrimesKernel.cuh"
#include "kernels/initializeWheelMaskKernel.cuh"
#include "kernels/mainPrimesKernel.cuh"

namespace
{
    __forceinline__ bool CheckCudaSupport()
    {
        int deviceCount = 0;
        const cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess || deviceCount == 0)
        {
            fprintf(stderr, "No CUDA device found.\n");
            return false;
        }

        cudaDeviceProp properties{};
        cudaGetDeviceProperties(&properties, 0);
        printf("Using %s\n", properties.name);

        return true;
    }

    __forceinline__ void InitializeCuda(){
        cudaFree(nullptr);
    }

    __forceinline__ uint64_t CalculateSum(const uint32_t N, const uint32_t* const PrimesMask, uint32_t SizeInWords)
    {
        uint64_t sum = 0;
        for (const uint32_t p: {2u, 3u, 5u})
        {
            if (p <= N)
            {
                sum += p;
            }
        }
        for (uint64_t w = 0; w < SizeInWords; ++w)
        {
            uint32_t bits = ~PrimesMask[w];
            while (bits)
            {
                const uint64_t i = w * BitsPerWord + static_cast<uint64_t>(std::countr_zero(bits));
                bits &= bits - 1;
                const uint64_t p = WheelSpan * (i / WheelBits) + WheelResidue(static_cast<uint32_t>(i % WheelBits));
                if (p > 5 && p <= N)
                {
                    sum += p;
                }
            }
        }
        return sum;
    }
} // namespace

int main(){
    constexpr uint32_t n = 70'000'000u;
    printf("N = %" PRIu32 "\n", (n));

    if (!CheckCudaSupport())
    {
        return EXIT_FAILURE;
    }
    const auto programStartTime = std::chrono::steady_clock::now();
    InitializeCuda();
    const auto processingStartTime = std::chrono::steady_clock::now();

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

    const uint32_t maxBaseNum = static_cast<uint32_t>(std::sqrt(n)) + 1u;

    constexpr uint32_t totalBits = (n / WheelSpan + 1u) * WheelBits;
    constexpr uint32_t totalWords = (totalBits + BitsPerWord - 1u) / BitsPerWord;
    constexpr uint32_t totalSegments = (totalWords + SegmentWords - 1u) / SegmentWords;
    constexpr uint32_t segmentsPerBatch = (totalSegments + NumBatches - 1u) / NumBatches;

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

    CUDA_CHECK(cudaStreamWaitEvent(streams[1], gpuStartEvent));
    InitMasksKernel<<<dim3((MaxMaskPeriod + 255u) / 256u, MaskGroupsCount), 256, 0, streams[1]>>>(MaskGroups);
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
    cudaStreamQuery(streams[NumBatches - 1]);

    constexpr size_t sieveBytes = static_cast<size_t>(totalWords) * sizeof(uint32_t);
    constexpr size_t hostBytes = (sieveBytes + PageBytes - 1) / PageBytes * PageBytes;
    uint32_t* finalSieve = static_cast<uint32_t *>(::operator new(hostBytes, std::align_val_t{PageBytes}));
    std::byte* hostBytesPtr = reinterpret_cast<std::byte*>(finalSieve);
    const std::byte* deviceBytesPtr = reinterpret_cast<const std::byte*>(dFinalSieve);

    void* pinnedPieces[NumBatches * PinPiecesPerBatch];
    uint32_t numPinnedPieces = 0;

    for (uint32_t k = 0; k < NumBatches; ++k)
    {
        const size_t firstWord = std::min<size_t>(static_cast<size_t>(k) * segmentsPerBatch * SegmentWords,
                                                  totalWords);
        const size_t lastWord = std::min<size_t>(static_cast<size_t>(k + 1) * segmentsPerBatch * SegmentWords,
                                                 totalWords);
        if (firstWord < lastWord)
        {
            const size_t batchBegin = firstWord * sizeof(uint32_t);
            const size_t batchEnd = lastWord == totalWords ? hostBytes : lastWord * sizeof(uint32_t);
            const size_t batchSize = batchEnd - batchBegin;
            for (uint32_t j = 0; j < PinPiecesPerBatch; ++j)
            {
                const size_t pieceBegin = batchBegin + batchSize * j / PinPiecesPerBatch / PageBytes * PageBytes;
                const size_t pieceEnd = j + 1 == PinPiecesPerBatch
                                            ? batchEnd
                                            : batchBegin + batchSize * (j + 1) / PinPiecesPerBatch / PageBytes *
                                              PageBytes;
                if (pieceBegin >= pieceEnd)
                {
                    continue;
                }
                CUDA_CHECK(cudaHostRegister(hostBytesPtr + pieceBegin, pieceEnd - pieceBegin,
                    cudaHostRegisterDefault));
                pinnedPieces[numPinnedPieces++] = hostBytesPtr + pieceBegin;

                const size_t copyEnd = std::min(pieceEnd, sieveBytes);
                if (pieceBegin < copyEnd)
                {
                    CUDA_CHECK(cudaMemcpyAsync(hostBytesPtr + pieceBegin, deviceBytesPtr + pieceBegin,
                        copyEnd - pieceBegin, cudaMemcpyDeviceToHost, streams[k]));
                    cudaStreamQuery(streams[k]);
                }
            }
        }
        if (k != 0)
        {
            CUDA_CHECK(cudaEventRecord(batchDone[k], streams[k]));
            CUDA_CHECK(cudaStreamWaitEvent(streams[0], batchDone[k]));
        }
    }

    cudaEventRecord(gpuFinishEvent, streams[0]);
    cudaEventSynchronize(gpuFinishEvent);

    const auto cudaInitializationTime = processingStartTime - programStartTime;
    std::cout << "CUDA Initialization Time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(cudaInitializationTime).count() << " ms\n";

    const auto finishTime = std::chrono::steady_clock::now();
    const auto processingTime = std::chrono::steady_clock::now() - processingStartTime;
    std::cout << "CPU Processing Time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(processingTime).count() << " ms\n";


    float gpuMilliseconds = 0;
    cudaEventElapsedTime(&gpuMilliseconds, gpuStartEvent, gpuFinishEvent);
    printf("GPU Processing Time: %.0f ms\n", gpuMilliseconds);

    const auto programTotalTime = finishTime - programStartTime;
    std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(programTotalTime).count()
            << " ms\n";

    const uint64_t sum = CalculateSum(n, finalSieve, totalWords);

    printf("Sum: %" PRIu64, sum);

    for (uint32_t i = 0; i < numPinnedPieces; ++i)
    {
        cudaHostUnregister(pinnedPieces[i]);
    }
    ::operator delete(finalSieve, std::align_val_t{PageBytes});
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
