#include "PoolingLayer.h"
#include <math.h>
PoolingLayer::PoolingLayer(int kernelSizeX, int kernelSizeY):kernelSizeX(kernelSizeX), kernelSizeY(kernelSizeY){}
PoolingLayer::~PoolingLayer(){}
void PoolingLayer::run(const Tensor &tensor){
    int sizeX = ceil((float)tensor.getX() / (float)kernelSizeX);
    int sizeY = ceil((float)tensor.getY() / (float)kernelSizeY);
    Tensor * output = new Tensor(sizeX, sizeY, tensor.getZ());
    for(int z = 0; z < tensor.getZ(); z++){
        for(int y = 0; y < tensor.getY() - kernelSizeY; y += kernelSizeY){
            for(int x = 0; x < tensor.getX() - kernelSizeX; x+= kernelSizeX){
                int max = 0;
                for(int kY = 0; kY < kernelSizeY; kY++){
                    for(int kX = 0; kX < kernelSizeX; kX ++){
                        if(tensor.getValue(x + kX, y + kY, z) > max){
                            max = tensor.getValue(x + kX, y + kY, z);
                        }
                    }
                }
                output->edit(x / kernelSizeX, y / kernelSizeY, z, max);
            }
        }
    }
    this->outputTensor = outputTensor;
}