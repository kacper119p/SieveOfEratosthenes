#pragma once
#include <algorithm>

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

constexpr uint32_t NumBatches = 8u;
constexpr uint32_t PinPiecesPerBatch = 4u;
constexpr size_t PageBytes = 4096u;

constexpr uint32_t MaskGroupsCount = 4u;
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
