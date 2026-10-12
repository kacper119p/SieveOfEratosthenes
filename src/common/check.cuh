#pragma once

#ifdef NDEBUG
#define SIEVE_OF_ERATOSTHENES_CUDA_CHECK(call) call
#else
#define SIEVE_OF_ERATOSTHENES_CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t error = call;                                                                                      \
        if (error != cudaSuccess) {                                                                                    \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error));              \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)
#endif
