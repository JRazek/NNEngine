//
// Created by user on 12.09.2021.
//

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
    void cudaConvolveKernel(double *input, double *kernel, double *result, int strideX, int strideY, dim3 inputSize, dim3 outputSize, dim3 kernelSize) {
        u_int index = blockIdx.x * blockDim.x + threadIdx.x;
        u_int posXOutput = index % outputSize.x;
        u_int posYOutput = (index % (outputSize.x * outputSize.y)) / outputSize.x;

        u_int kID = index / (outputSize.x * outputSize.y * inputSize.z); //same as posZOutput
//        printf("kID:%d index:%d \n", kID, index);

        u_int kPosX = posXOutput * strideX;
        u_int kPosY = posYOutput * strideY;
        u_int kPosZ = index / (outputSize.x * outputSize.y);

        const double *kernelStart = kernel + kID * (kernelSize.x * kernelSize.y * kernelSize.z);

        double sum = 0;
        for(u_int ky = 0; ky < kernelSize.y; ky++){
            for(u_int kx = 0; kx < kernelSize.x; kx++){
                double v = kernelStart[getDataIndex(kernelSize, {kx, ky, kPosZ})];
                double in = input[getDataIndex(inputSize, {kPosX + kx, kPosY + ky, kPosZ})];
                sum += v * in;
            }
        }
       // u_int resultIndex = getDataIndex(outputSize, {posXOutput, posYOutput, kID});
        result[index] = sum;

//        printf("index:%d x:%d y:%d z:%d kID:%d  sum:%g\n", index, kPosX, kPosY, kPosZ, kID, sum);
    }
    __global__
    void cudaCombineResult(double *resultRaw, double *resultCombined, u_int kernelDepth, dim3 resultCombinedDim){
        dim3 resultRawDim(resultCombinedDim.x, resultCombinedDim.y, kernelDepth * resultCombinedDim.z);
        u_int kSizeZ = resultRawDim.z / resultCombinedDim.z;
        u_int index = blockIdx.x * blockDim.x + threadIdx.x;
        u_int resultCombinedPosX = index % resultCombinedDim.x;
        u_int resultCombinedPosY = (index % (resultCombinedDim.x * resultCombinedDim.y)) / resultCombinedDim.x;
        u_int resultCombinedPosZ = index / (resultCombinedDim.x * resultCombinedDim.y);



        resultCombined[index] = 0;

        for(u_int zRaw = resultCombinedPosZ * kSizeZ; zRaw < (resultCombinedPosZ + 1) * kSizeZ; zRaw++){
            u_int indexRaw = getDataIndex(resultRawDim, {resultCombinedPosX, resultCombinedPosY, zRaw});
            float val = resultRaw[indexRaw];
            resultCombined[index] += val;
//            if(resultCombinedPosX == 0 && resultCombinedPosY == 0){
//                printf("val:%.15f res:%.15f\n", val, resultCombined[index]);
//            }
        }
//        printf("index:%d x:%d y:%d z:%d res:%f\n", index, resultCombinedPosX, resultCombinedPosY, resultCombinedPosZ, resultCombined[index]);

    }
}

cn::Bitmap<double> cn::CUDAUtils::cudaConvolve(const std::vector<cn::Bitmap<double>> &kernels, const cn::Bitmap<double> &input, int paddingX, int paddingY, int strideX, int strideY) {
    double *kernelDev, *dataDev, *resRawDev, *resCombinedDev;

    Bitmap<double> paddedInput = cn::Utils::addPadding(input, paddingX, paddingY);

    int sX = cn::Utils::afterConvolutionSize(kernels[0].w(), input.w(), paddingX, strideX);
    int sY = cn::Utils::afterConvolutionSize(kernels[0].h(), input.h(), paddingY, strideY);

    u_int kerSize = kernels[0].size().multiplyContent() * sizeof(double);
    u_int dataSize = paddedInput.size().multiplyContent() * sizeof(double);
    u_int resultRawSize = sX * sY * paddedInput.d() * kernels.size() * sizeof(double);
    u_int resultCombinedSize = sX * sY * kernels.size() * sizeof(double);

    kernelDev = (double *) fixedCudaMalloc(kerSize * kernels.size());
    dataDev = (double *) fixedCudaMalloc(dataSize);
    resRawDev = (double *) fixedCudaMalloc(resultRawSize);
    resCombinedDev = (double *) fixedCudaMalloc(resultCombinedSize);
    if(!kernelDev || !dataDev || !resRawDev || !resCombinedDev){
        throw std::logic_error("BAD ALLOC");
    }

    for(int i = 0; i < kernels.size(); i ++){
        cudaMemcpy(kernelDev + i * kerSize, kernels[i].dataConst(), kerSize, cudaMemcpyHostToDevice);
    }

//    cudaMemset(resRawDev, 0, resultRawSize);

    cudaMemcpy(dataDev, paddedInput.dataConst(), dataSize, cudaMemcpyHostToDevice);

    Bitmap<double> result(sX, sY, kernels.size());

    int threadsRawCount = result.w() * result.h() * kernels.size() * paddedInput.d();

    constexpr int threadsPerBlock = 1024;


    dim3 inputSize = {static_cast<u_int>(paddedInput.w()), static_cast<u_int>(paddedInput.h()), static_cast<u_int>(paddedInput.d())};
    dim3 outputSize = {static_cast<u_int>(result.w()), static_cast<u_int>(result.h()), static_cast<u_int>(result.d())};
    dim3 kernelSize = {static_cast<u_int>(kernels[0].w()), static_cast<u_int>(kernels[0].h()), static_cast<u_int>(kernels[0].d())};

    cudaConvolveKernel<<<threadsRawCount/threadsPerBlock + 1, std::min(threadsRawCount, threadsPerBlock)>>>
    (
            dataDev,
            kernelDev,
            resRawDev,
            strideX,
            strideY,
            inputSize,
            outputSize,
            kernelSize
    );

    int threadsCombinedCount = result.w() * result.h() * kernels.size();

    cudaCombineResult<<<threadsCombinedCount / threadsPerBlock + 1, std::min(threadsCombinedCount, threadsPerBlock)>>>
    (
            resRawDev,
            resCombinedDev,
            kernels[0].d(),
            outputSize
    );


    double *hostRes = new double[sX * sY * kernels.size()];
    cudaMemcpy(hostRes, resCombinedDev, resultCombinedSize, cudaMemcpyDeviceToHost);

    result.setData(hostRes);

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
