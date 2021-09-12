//
// Created by user on 12.09.2021.
//

#ifndef NEURALNETLIBRARY_CUDAUTILS_CUH
#define NEURALNETLIBRARY_CUDAUTILS_CUH
#include <vector>

namespace cn {
    template<typename T>
    class Bitmap;

    class CUDAUtils {
        static void *fixedCudaMalloc(const size_t size);

    public:
        static Bitmap<double> cudaConvolve(const std::vector<cn::Bitmap<double>> &kernels, const Bitmap<double> &input, int paddingX = 0, int paddingY = 0, int strideX = 1, int strideY = 1);
    };
}

#endif //NEURALNETLIBRARY_CUDAUTILS_CUH
