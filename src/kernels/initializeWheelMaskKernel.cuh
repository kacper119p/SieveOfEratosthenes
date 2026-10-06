#pragma once
#include "common/wheelMask.cuh"

namespace SieveOfEratosthenes
{
    __global__ void InitMasksKernel(MaskGroupTable Table);
}
