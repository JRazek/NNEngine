#include "PoolingLayer.h"
#include <math.h>
PoolingLayer::PoolingLayer(int id, Net * net, int kernelSizeX, int kernelSizeY):
    kernelSizeX(kernelSizeX), kernelSizeY(kernelSizeY), Layer(id, net), outputTensor(){}
PoolingLayer::~PoolingLayer(){}
void PoolingLayer::run(const Tensor &tensor){
    int sizeX = tensor.getX() / kernelSizeX;
    int sizeY = tensor.getY() / kernelSizeY;
    Tensor * output = new Tensor(sizeX, sizeY, tensor.getZ());
    for(int z = 0; z < tensor.getZ(); z++){
        for(int y = 0; y < tensor.getY() - kernelSizeY; y += kernelSizeY){
            for(int x = 0; x < tensor.getX() - kernelSizeX; x+= kernelSizeX){
                float max = 0;
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
    this->outputTensor = *output;    
    for(int z = 0; z < outputTensor.getZ(); z++){
        for(int y = 0; y < outputTensor.getY(); y ++){
            for(int x = 0; x < outputTensor.getX(); x ++){
                this->outputVector.push_back(outputTensor.getValue(x, y, z));
            }
        }
    }
}