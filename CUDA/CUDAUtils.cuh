//
// Created by user on 12.09.2021.
//

#ifndef NEURALNETLIBRARY_CUDAUTILS_CUH
#define NEURALNETLIBRARY_CUDAUTILS_CUH
#include <vector>

namespace cn {
    template<typename T>
    class Bitmap;

    constexpr u_int THREADS_PER_BLOCK = 1024;
    class CUDAUtils {

    public:
        static Bitmap<double> cudaConvolve(const std::vector<cn::Bitmap<double>> &kernels, const Bitmap<double> &input, int paddingX = 0, int paddingY = 0, int strideX = 1, int strideY = 1);

        static void *fixedCudaMalloc(const size_t size);
    };
}

#endif //NEURALNETLIBRARY_CUDAUTILS_CUH
