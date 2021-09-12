//
// Created by user on 12.09.2021.
//

#include "CUDAUtils.cuh"
#include "../Utils/dataStructures/Bitmap.h"

namespace cn {
    __device__
    int getDataIndex(dim3 bitmapSize, int col, int row, int depth){
        if(col < 0 || col >= bitmapSize.x || row < 0 || row >= bitmapSize.y || depth < 0 || depth >= bitmapSize.z)
            printf("zły arg1 :P");
        return depth * bitmapSize.x * bitmapSize.y + row * bitmapSize.x + col;
    }
    __device__
    dim3 getDataPos(dim3 bitmapSize, int index){
        if(index >= bitmapSize.x * bitmapSize.y * bitmapSize.z)
            printf("zły arg2 :P");
        return dim3(index % bitmapSize.x, (index / bitmapSize.x) % bitmapSize.x, index / (bitmapSize.x * bitmapSize.y));
    }
    __device__
    int afterConvolutionSize(int kernelSize, int inputSize, int padding, int stride) {
        return (inputSize + 2 * padding - kernelSize) / stride + 1;
    }
    __global__
    void cudaConvolveKernel(double *data, double *kernel, dim3 outputSize, dim3 dataSize, dim3 kernelSize, int strideX, int strideY,
                            int paddingX, int paddingY) {
        u_int index = blockIdx.x * blockDim.x + threadIdx.x;
        dim3 kernelPos = getDataPos(outputSize, index);

    }
}

cn::Bitmap<double> cn::CUDAUtils::cudaConvolve(const std::vector<cn::Bitmap<double>> &kernels, const cn::Bitmap<double> &input, int paddingX, int paddingY, int strideX, int strideY) {
    double *kernelDev, *dataDev, *resDev;

    Bitmap<double> paddedInput = cn::Utils::addPadding(input, paddingX, paddingY);

    int sX = cn::Utils::afterConvolutionSize(kernels[0].w(), paddedInput.w(), paddingX, strideX);
    int sY = cn::Utils::afterConvolutionSize(kernels[0].h(), paddedInput.h(), paddingY, strideY);

    u_int kerSize = kernels[0].size().multiplyContent() * sizeof(double) * kernels.size();
    u_int dataSize = paddedInput.size().multiplyContent() * sizeof(double);
    u_int resultSize = sX * sY * kernels.size() * sizeof(double);

    kernelDev = (double *) fixedCudaMalloc(kerSize);
    dataDev = (double *) fixedCudaMalloc(dataSize);
    resDev = (double *) fixedCudaMalloc(resultSize);

    for(int i = 0; i < kernels.size(); i ++){
        cudaMemcpy(kernelDev + sX * sY * i, kernels[i].dataConst(), kernels[i].size().multiplyContent() * sizeof(double), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(dataDev, paddedInput.dataConst(), dataSize, cudaMemcpyHostToDevice);

    Bitmap<double> result(sX, sY, kernels.size());

    int threadsCount = result.w() * result.h() * kernels.size();

    constexpr int threadsPerBlock = 1024;


    cudaConvolveKernel<<<threadsCount/threadsPerBlock + 1, std::min(threadsCount, threadsPerBlock)>>>(dataDev, kernelDev,
            {static_cast<u_int>(result.w()), static_cast<u_int>(result.h()), static_cast<u_int>(result.d())},
            {static_cast<u_int>(paddedInput.w()), static_cast<u_int>(paddedInput.h()), static_cast<u_int>(paddedInput.d())},
            {static_cast<u_int>(kernels[0].w()), static_cast<u_int>(kernels[0].h()), static_cast<u_int>(kernels[0].d())},
            strideX, strideY, paddingX, paddingY);



    double *hostRes = new double[sX * sY * kernels.size()];
    cudaMemcpy(hostRes, resDev, resultSize, cudaMemcpyDeviceToHost);

    result.setData(hostRes);

    delete[] hostRes;

    cudaFree(kernelDev);
    cudaFree(dataDev);
    cudaFree(resDev);

    return result;
}



void *cn::CUDAUtils::fixedCudaMalloc(size_t size) {
    void* tmp;
    return cudaMalloc(&tmp, size) == cudaError::cudaSuccess ? tmp : nullptr;
}
