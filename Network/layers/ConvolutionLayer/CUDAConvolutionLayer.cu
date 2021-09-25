//
// Created by user on 17.09.2021.
//

#include "CUDAConvolutionLayer.cuh"
#include "ConvolutionLayer.h"
#include <memory>
#include "../../../CUDA/CUDAUtils.cuh"

namespace cn{

    __device__
    inline int afterConvolutionSize(int kernelSize, int inputSize, int padding, int stride) {
        return (inputSize + 2 * padding - kernelSize) / stride + 1;
    }
    __device__
    inline dim3 getDataPos(dim3 bitmapSize, int index){
        if(index >= bitmapSize.x * bitmapSize.y * bitmapSize.z)
            printf("zły arg2 :P");
        return dim3(index % bitmapSize.x, (index % bitmapSize.x * bitmapSize.y) / bitmapSize.x, index / (bitmapSize.x * bitmapSize.y));
    }
    __device__
    inline int getDataIndex(dim3 bitmapSize, dim3 pos){
        if(pos.x >= bitmapSize.x) {
            printf("x, %d %d\n", pos.x, bitmapSize.x);
        }
        if(pos.y >= bitmapSize.y) {
            printf("y, %d %d\n", pos.y, bitmapSize.y);
        }
        if(pos.z >= bitmapSize.z) {
            printf("z, %d %d\n", pos.z, bitmapSize.z);
        }
        return pos.z * bitmapSize.x * bitmapSize.y + pos.y * bitmapSize.x + pos.x;
    }
    __global__
    void
    CUDAConvAutoGrad(double *inputValues, double *kernelValues, double *resultChainValues, double *nextLayerChains, int kernelsCount, const Vector3<int> &inputSize, const Vector3<int> &kernelSize, const Vector2<int> &stride) {
        u_int index = blockDim.x * blockIdx.x + threadIdx.x;
        if(index >= inputSize.x * inputSize.y * inputSize.z)
            return;

    }
}
cn::Tensor<double> cn::CUDAConvolutionLayer::CUDARun(cn::ConvolutionLayer &convolutionLayer, const cn::Tensor<double> &_input) {
    convolutionLayer.output = std::make_unique<Tensor<double>>(
            CUDAUtils::cudaConvolve(
                convolutionLayer.kernels, _input,
                convolutionLayer.padding.x, convolutionLayer.padding.y,
                convolutionLayer.stride.x, convolutionLayer.stride.y
            )
    );

    return *convolutionLayer.output.get();
}

void cn::CUDAConvolutionLayer::CUDAAutoGrad(cn::ConvolutionLayer &convolutionLayer) {
    u_int inputSize = convolutionLayer.inputSize.multiplyContent();
    double *inputDev, *kernelDev, *chainValuesDev;


    Tensor<double> paddedInput = Utils::addPadding(*convolutionLayer.getInput().get(), convolutionLayer.padding.x, convolutionLayer.padding.y);

    u_int paddedInputBytes = paddedInput.size().multiplyContent() * sizeof(double);
    u_int combinedKernelsBytes = convolutionLayer.kernels[0].size().multiplyContent() * convolutionLayer.kernelsCount * sizeof(double);

    inputDev = (double *)CUDAUtils::fixedCudaMalloc(paddedInputBytes);
    kernelDev = (double *)CUDAUtils::fixedCudaMalloc(combinedKernelsBytes);
    chainValuesDev = (double *)CUDAUtils::fixedCudaMalloc(paddedInputBytes);


    std::vector<double> kernelsCombinedData(convolutionLayer.kernels[0].size().multiplyContent() * convolutionLayer.kernelsCount);
    for(u_int i = 0; i < convolutionLayer.kernelsCount; i ++){
        Tensor<double> &kernel = convolutionLayer.kernels[i];
        std::copy(kernel.dataConst(), kernel.dataConst() + kernel.size().multiplyContent(), kernelsCombinedData.begin() + kernel.size().multiplyContent());
    }

    cudaMemcpy(inputDev, paddedInput.data(), paddedInputBytes, cudaMemcpyHostToDevice);

    //todo
//    CUDAConvAutoGrad<<<inputSize/cn::THREADS_PER_BLOCK+1, cn::THREADS_PER_BLOCK>>>(inputDev, kernelDev, chainValuesDev, paddedInput.size(), convolutionLayer.kernels[0].size());

    cudaMemcpy(convolutionLayer.memoizationTable->data(), chainValuesDev, paddedInputBytes, cudaMemcpyDeviceToHost);

    std::fill(convolutionLayer.memoizationStates->data(), convolutionLayer.memoizationStates->data() + paddedInput.size().multiplyContent(), true);

    cudaFree(inputDev);
    cudaFree(kernelDev);
    cudaFree(chainValuesDev);

}
