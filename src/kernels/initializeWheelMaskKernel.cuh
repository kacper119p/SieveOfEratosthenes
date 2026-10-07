#pragma once
#include "common/wheel.cuh"

namespace SieveOfEratosthenes
{
    __global__ void InitializeWheelsMaskKernel(MaskGroupTable Table);
}
