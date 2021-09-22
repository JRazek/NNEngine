//
// Created by user on 21.09.2021.
//

#include "CUDAOutputLayer.cuh"
#include "OutputLayer.h"
#include "../../../CUDA/CUDAUtils.cuh"

namespace cn{
    __global__
    void CUDACalcOutputGradients(double *output, double *target, double *result, dim3 outputDims){
        u_int index = blockDim.x * blockIdx.x + threadIdx.x;
        if(index >= outputDims.x * outputDims.y * outputDims.z)
            return;
        result[index] = output[index] - target[index];
    }
    inline dim3 vec3ToDim3(const cn::Vector3<int> &vec){
        return dim3(static_cast<u_int>(vec.x), static_cast<u_int>(vec.y), static_cast<u_int>(vec.z));
    }
}

void cn::CUDAOutputLayer::CUDAAutoGrad(cn::OutputLayer &outputLayer) {
    double *outputDev, *targetDev, *resultDev;

    outputDev = (double *) CUDAUtils::fixedCudaMalloc(outputLayer.output->size().multiplyContent() * sizeof(double));
    targetDev = (double *) CUDAUtils::fixedCudaMalloc(outputLayer.target->size().multiplyContent() * sizeof(double));
    resultDev = (double *) CUDAUtils::fixedCudaMalloc(outputLayer.target->size().multiplyContent() * sizeof(double));

    u_int threadsCount = outputLayer.output->size().multiplyContent();

    cudaMemcpy(outputDev, outputLayer.output->dataConst(), outputLayer.output->size().multiplyContent() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(targetDev, outputLayer.target->dataConst(), outputLayer.target->size().multiplyContent() * sizeof(double), cudaMemcpyHostToDevice);

    CUDACalcOutputGradients<<<threadsCount / cn::THREADS_PER_BLOCK + 1, cn::THREADS_PER_BLOCK>>> (outputDev, targetDev, resultDev, vec3ToDim3(outputLayer.outputSize));

    cudaMemcpy(outputLayer.memoizationTable->data(), resultDev, outputLayer.target->size().multiplyContent() * sizeof(double), cudaMemcpyDeviceToHost);
    std::fill(outputLayer.memoizationStates->data(), outputLayer.memoizationStates->data() + outputLayer.outputSize.multiplyContent(), true);

    cudaFree(outputDev);
    cudaFree(targetDev);
    cudaFree(resultDev);
}
