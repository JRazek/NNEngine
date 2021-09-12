//
// Created by user on 12.09.2021.
//

#ifndef NEURALNETLIBRARY_CUDAUTILS_CUH
#define NEURALNETLIBRARY_CUDAUTILS_CUH
#include "../Utils/dataStructures/Bitmap.h"

namespace cn {
    class CUDAUtils {
        __host__
        static void cudaConvolveKernel(double *data, double *kernel, dim3 dataSize, dim3 kernelSize, int strideX = 1, int strideY = 1, int paddingX = 0, int paddingY = 0);
        static void *fixedCudaMalloc(const size_t size);
    public:
        static Bitmap<double> cudaConvolve(const Bitmap<double> &kernel, const Bitmap<double> &input, int paddingX = 0, int paddingY = 0, int strideX = 1, int strideY = 1);
    };
}

#endif //NEURALNETLIBRARY_CUDAUTILS_CUH
