//
// Created by user on 12.09.2021.
//

#include <thread>
#include "CUDAUtils.cuh"
#include "../Utils/dataStructures/Bitmap.h"

namespace cn {
    __device__
    int getDataIndex(dim3 bitmapSize, dim3 pos){
//        if(pos.x >= bitmapSize.x) {
//            printf("x, %d %d\n", pos.x, bitmapSize.x);
//        }
//        if(pos.y >= bitmapSize.y) {
//            printf("y, %d %d\n", pos.y, bitmapSize.y);
//        }
//        if(pos.z >= bitmapSize.z) {
//            printf("z, %d %d\n", pos.z, bitmapSize.z);
//        }
        //    return depth * _w * _h + row * _w + col;
        return pos.z * bitmapSize.x * bitmapSize.y + pos.y * bitmapSize.x + pos.x;
    }
    __device__
    dim3 getDataPos(dim3 bitmapSize, int index){
//        if(index >= bitmapSize.x * bitmapSize.y * bitmapSize.z)
//            printf("zły arg2 :P");
        return dim3(index % bitmapSize.x, (index / bitmapSize.x) % bitmapSize.x, index / (bitmapSize.x * bitmapSize.y));
    }
    __device__
    int afterConvolutionSize(int kernelSize, int inputSize, int padding, int stride) {
        return (inputSize + 2 * padding - kernelSize) / stride + 1;
    }


    __global__
    void cudaConvolveKernel(double *input, const double *kernel, double *result, int strideX, int strideY, dim3 inputSize, dim3 outputSize, dim3 kernelSize) {
        //each index corresponds to output cell.
        u_int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= outputSize.x * outputSize.y * kernelSize.z)
            return;
        u_int posXInput = (index % outputSize.x) * strideX;
        u_int posYInput = ((index % (outputSize.x * outputSize.y)) / outputSize.x) * strideY;
        u_int posZInput = (index % (outputSize.x * outputSize.y * outputSize.z)) / (outputSize.x * outputSize.y);

        u_int posZKernel = index / (outputSize.x * outputSize.y);

        double sum = 0;
        for(u_int ky = 0; ky < kernelSize.y; ky++){
            for(u_int kx = 0; kx < kernelSize.x; kx++){
                sum += kernel[getDataIndex(kernelSize, {kx, ky, posZKernel})] * input[getDataIndex(inputSize, {posXInput + kx, posYInput + ky, posZInput})];
            }
        }

        result[index] = sum;

//      printf("index:%d x:%d y:%d z:%d kID:%d  sum:%g\n", index, kPosX, kPosY, kPosZ, kID, sum);
    }
    __global__
    void cudaCombineResult(const double *resultRaw, double *resultCombined, u_int kernelDepth, dim3 resultCombinedDim){
        dim3 resultRawDim(resultCombinedDim.x, resultCombinedDim.y, kernelDepth * resultCombinedDim.z);
        u_int kSizeZ = resultRawDim.z / resultCombinedDim.z;
        u_int index = blockIdx.x * blockDim.x + threadIdx.x;

        if(index >= resultCombinedDim.x * resultCombinedDim.y * resultCombinedDim.z)
            return;


        u_int resultCombinedPosX = index % resultCombinedDim.x;
        u_int resultCombinedPosY = (index % (resultCombinedDim.x * resultCombinedDim.y)) / resultCombinedDim.x;
        u_int resultCombinedPosZ = index / (resultCombinedDim.x * resultCombinedDim.y);



        resultCombined[index] = 0;
        if(index == 0){
//            printf("%.15f\n%.15f\n%.15f\n", resultRaw[2], resultRaw[164], resultRaw[83]);
        }

        for(u_int zRaw = resultCombinedPosZ * kSizeZ; zRaw < (resultCombinedPosZ + 1) * kSizeZ; zRaw++){
            u_int indexRaw = getDataIndex(resultRawDim, {resultCombinedPosX, resultCombinedPosY, zRaw});
            double val = resultRaw[indexRaw];
            resultCombined[index] += val;
            if(resultCombinedPosX == 2 && resultCombinedPosY == 0){
//                printf("index:%d, val:%.15f \n", indexRaw,  val);
            }
        }
//        printf("index:%d x:%d y:%d z:%d res:%f\n", index, resultCombinedPosX, resultCombinedPosY, resultCombinedPosZ, resultCombined[index]);
    }
}

cn::Bitmap<double> cn::CUDAUtils::cudaConvolve(const std::vector<cn::Bitmap<double>> &kernels, const cn::Bitmap<double> &input, int paddingX, int paddingY, int strideX, int strideY) {
    double *kernelDev, *dataDev, *resRawDev, *resCombinedDev;

    Bitmap<double> paddedInput = cn::Utils::addPadding(input, paddingX, paddingY);

    int sX = cn::Utils::afterConvolutionSize(kernels[0].w(), input.w(), paddingX, strideX);
    int sY = cn::Utils::afterConvolutionSize(kernels[0].h(), input.h(), paddingY, strideY);

    u_int kerBytes = kernels[0].size().multiplyContent() * sizeof(double);
    u_int inputBytes = paddedInput.size().multiplyContent() * sizeof(double);
    u_int resultRawBytes = sX * sY * paddedInput.d() * kernels.size() * sizeof(double);
    u_int resultCombinedBytes = sX * sY * kernels.size() * sizeof(double);

    kernelDev = (double *) fixedCudaMalloc(kerBytes * kernels.size());
    dataDev = (double *) fixedCudaMalloc(inputBytes);
    resRawDev = (double *) fixedCudaMalloc(resultRawBytes);
    resCombinedDev = (double *) fixedCudaMalloc(resultCombinedBytes);
    if(!kernelDev || !dataDev || !resRawDev || !resCombinedDev){
        throw std::logic_error("BAD ALLOC");
    }

    for(int i = 0; i < kernels.size(); i ++){
        cudaMemcpy(kernelDev + i * kernels[i].size().multiplyContent(), kernels[i].dataConst(), kerBytes, cudaMemcpyHostToDevice);
    }

    cudaMemcpy(dataDev, paddedInput.dataConst(), inputBytes, cudaMemcpyHostToDevice);

    Bitmap<double> result(sX, sY, kernels.size());

    constexpr int threadsPerBlock = 1024;

    dim3 inputDim = {static_cast<u_int>(paddedInput.w()), static_cast<u_int>(paddedInput.h()), static_cast<u_int>(paddedInput.d())};
    dim3 convOutputDim = {static_cast<u_int>(result.w()), static_cast<u_int>(result.h()), static_cast<u_int>(paddedInput.d())};
    dim3 finalOutputDim = {static_cast<u_int>(result.w()), static_cast<u_int>(result.h()), static_cast<u_int>(result.d())};
    dim3 kernelDim = {static_cast<u_int>(kernels[0].w()), static_cast<u_int>(kernels[0].h()), static_cast<u_int>(kernels[0].d() * kernels.size())};

    int threadsRawCount = result.w() * result.h() * kernels.size() * paddedInput.d();
    cudaConvolveKernel<<<threadsRawCount/threadsPerBlock + 1, threadsPerBlock>>>
    (
            dataDev,
            kernelDev,
            resRawDev,
            strideX,
            strideY,
            inputDim,
            convOutputDim,
            kernelDim
    );


    cudaDeviceSynchronize();

    int threadsCombinedCount = result.w() * result.h() * kernels.size();
    cudaCombineResult<<<threadsCombinedCount / threadsPerBlock + 1, threadsPerBlock>>>
    (
            resRawDev,
            resCombinedDev,
            kernels[0].d(),
            finalOutputDim
    );


    double *hostRes = new double[sX * sY * kernels.size()];
    cudaMemcpy(hostRes, resCombinedDev, resultCombinedBytes, cudaMemcpyDeviceToHost);

    result.setData(std::move(hostRes));

    delete[] hostRes;

    cudaFree(kernelDev);
    cudaFree(dataDev);
    cudaFree(resRawDev);

    return result;
}



void *cn::CUDAUtils::fixedCudaMalloc(size_t size) {
    void* tmp;
    return cudaMalloc(&tmp, size) == cudaError::cudaSuccess ? tmp : nullptr;
}
