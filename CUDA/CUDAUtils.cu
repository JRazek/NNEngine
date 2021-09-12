//
// Created by user on 12.09.2021.
//

#include "CUDAUtils.cuh"
#include "../Utils/dataStructures/Bitmap.h"

namespace cn {
    __device__
    int afterConvolutionSize(int kernelSize, int inputSize, int padding, int stride) {
        return (inputSize + 2 * padding - kernelSize) / stride + 1;
    }
    __global__
    void cudaConvolveKernel(double *data, double *kernel, dim3 dataSize, dim3 kernelSize, int strideX, int strideY,
                            int paddingX, int paddingY) {
        int sX = afterConvolutionSize(kernelSize.x, dataSize.x, paddingX, strideX);
        int sY = afterConvolutionSize(kernelSize.y, dataSize.y, paddingY, strideY);
        u_int index = blockIdx.x * blockDim.x + threadIdx.x;
        //todo
    }
}

cn::Bitmap<double> cn::CUDAUtils::cudaConvolve(const std::vector<cn::Bitmap<double>> &kernels, const cn::Bitmap<double> &input, int paddingX, int paddingY, int strideX, int strideY) {
    double *kernelDev, *dataDev, *resDev;

    int sX = cn::Utils::afterConvolutionSize(kernels[0].w(), input.w(), paddingX, strideX);
    int sY = cn::Utils::afterConvolutionSize(kernels[0].h(), input.h(), paddingY, strideY);

    u_int kerSize = kernels[0].size().multiplyContent() * sizeof(double) * kernels.size();
    u_int dataSize = input.size().multiplyContent() * sizeof(double);
    u_int resultSize = sX * sY * kernels.size() * sizeof(double);

    kernelDev = (double *) fixedCudaMalloc(kerSize);
    dataDev = (double *) fixedCudaMalloc(dataSize);
    resDev = (double *) fixedCudaMalloc(resultSize);

    for(int i = 0; i < kernels.size(); i ++){
        cudaMemcpy(kernelDev + sX * sY * i, kernels[i].data(), kernels[i].size().multiplyContent() * sizeof(double), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(dataDev, input.data(), dataSize, cudaMemcpyHostToDevice);

    int threadsCount = sX * sY * kernels.size();

    constexpr int threadsPerBlock = 1024;

    cudaConvolveKernel<<<threadsCount/threadsCount + 1, std::min(threadsCount, threadsPerBlock)>>>(dataDev, kernelDev,
            {static_cast<u_int>(input.w()), static_cast<u_int>(input.h()), static_cast<u_int>(input.d())},
            {static_cast<u_int>(kernels[0].w()), static_cast<u_int>(kernels[0].h()), static_cast<u_int>(kernels[0].d())},
            strideX, strideY, paddingX, paddingY);



    double *hostRes = new double[kernels[0].size().multiplyContent() * kernels.size()];
    cudaMemcpy(hostRes, resDev, resultSize, cudaMemcpyDeviceToHost);

    Bitmap<double> result(sX, sY, kernels.size(), hostRes);

    delete[] hostRes;

    return result;
}



void *cn::CUDAUtils::fixedCudaMalloc(size_t size) {
    void* tmp;
    return cudaMalloc(&tmp, size) == cudaError::cudaSuccess ? tmp : nullptr;
}
