//
// Created by user on 12.09.2021.
//

#include "CUDAUtils.cuh"

int test(){
    return 0;
}

cn::Bitmap<double>
cn::CUDAUtils::cudaConvolve(const cn::Bitmap<double> &kernel, const cn::Bitmap<double> &input, int paddingX, int paddingY, int strideX, int strideY) {
    double *kData,*dData, *result;
    u_int kerSize = kernel.size().multiplyContent() * sizeof(double);
    int dataSize = input.size().multiplyContent() * sizeof(double);

    kData = (double *) fixedCudaMalloc(kerSize);
    dData = (double *) fixedCudaMalloc(dataSize);

    cudaMemcpy(kData, kernel.data(), kerSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dData, input.data(), dataSize, cudaMemcpyHostToDevice);



    return cn::Bitmap<double>();
}

__host__
void cn::CUDAUtils::cudaConvolveKernel(double *data, double *kernel, dim3 dataSize, dim3 kernelSize, int strideX,
                                       int strideY, int paddingX,
                                       int paddingY) {
    int sX = cn::Utils::afterConvolutionSize(kernelSize.x, dataSize.x, paddingX, strideX);
    int sY = cn::Utils::afterConvolutionSize(kernelSize.y, dataSize.y, paddingY, strideY);
    u_int index = blockIdx.x * blockDim.x + threadIdx.x;
}

void *cn::CUDAUtils::fixedCudaMalloc(size_t size) {
    void* tmp;
    return cudaMalloc(&tmp, size) == cudaError::cudaSuccess ? tmp : nullptr;
}

