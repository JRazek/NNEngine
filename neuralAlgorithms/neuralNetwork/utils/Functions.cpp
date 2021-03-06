#include "Functions.h"
#include <stdexcept>

Matrix Functions::convolve(const Tensor &inputTensor, const Tensor &kernel, int padding, const int stride){
        padding = 0; //change later
        if(inputTensor.getZ() != kernel.getZ()){
            throw std::invalid_argument( "depths of kernel and input does not match!\n" );
        }
        if(kernel.getX() % 2 == 0 || kernel.getY() % 2 == 0){
            throw std::invalid_argument( "kernel dimensions must be odd!\n" );
        }
        int newSizeX = afterConvolutionSize(inputTensor.getX(), kernel.getX(), padding, stride);
        int newSizeY = afterConvolutionSize(inputTensor.getY(), kernel.getY(), padding, stride);
        Matrix result = Matrix(newSizeX, newSizeY);
        
        for(int z = 0; z < inputTensor.getZ(); z++){
            const Matrix * inputMatrix = &inputTensor.getMatrix(z);
            const Matrix * kernelMatrix = &kernel.getMatrix(z);
            //convolution itself
            for(int y = 0; y < inputTensor.getY() - kernelMatrix->getY() + 1; y += stride){
                for(int x = 0; x < inputTensor.getX() - kernelMatrix->getX() + 1; x += stride){
                    //iterate over all in matrix
                    float sum = 0;
                    for(int y2 = 0; y2 < kernelMatrix->getY(); y2++){
                        for(int x2 = 0; x2 < kernelMatrix->getX(); x2++){
                            sum += inputMatrix->getValue(y + y2, x + x2) * kernelMatrix->getValue(y2, x2);
                        }
                    }
                    result.edit(x / stride, y / stride, result.getValue(x / stride, y / stride) + sum);
                }
            }
        }
        return result;//todo
    }
int Functions::afterConvolutionSize(const int inputSize, const int kernelSize, const int padding, const int stride){
    return inputSize - kernelSize + 2*padding / stride + 1;
}