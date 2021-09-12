//
// Created by user on 12.09.2021.
//

#include "CUDAUtils.cuh"

int test(){
    return 0;
}

cn::Bitmap<double> cn::CUDAUtils::cudaConvolve(const std::vector<const Bitmap<double> *> &kernels, const cn::Bitmap<double> &input, int paddingX, int paddingY, int strideX, int strideY) {
    double *kData,*dData, *resData;

    int sX = cn::Utils::afterConvolutionSize(kernels[0]->w(), input.w(), paddingX, strideX);
    int sY = cn::Utils::afterConvolutionSize(kernels[0]->h(), input.h(), paddingY, strideY);

    u_int kerSize = kernels[0]->size().multiplyContent() * sizeof(double) * kernels.size();
    u_int dataSize = input.size().multiplyContent() * sizeof(double);
    u_int resultSize = sX * sY * kernels.size() * sizeof(double);

    kData = (double *) fixedCudaMalloc(kerSize);
    dData = (double *) fixedCudaMalloc(dataSize);
    resData = (double *) fixedCudaMalloc(resultSize);

    cudaMemcpy(kData, kernels.data(), kerSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dData, input.data(), dataSize, cudaMemcpyHostToDevice);

    //run kernel here...
    //<<< >>>


    double *hostRes = new double[kernels[0]->size().multiplyContent() * kernels.size()];
    cudaMemcpy(hostRes, resData, resultSize, cudaMemcpyDeviceToHost);

    Bitmap<double> result(sX, sY, kernels.size(), hostRes);

    delete[] hostRes;

    return result;
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

