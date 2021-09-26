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


}
