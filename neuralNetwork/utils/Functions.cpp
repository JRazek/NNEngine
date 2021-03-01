#include "Functions.h"
#include <stdexcept>

Matrix Functions::convolve(const Tensor &inputTensor, const Tensor &kernel, const int padding, const int stride){
        if(inputTensor.matrices.size() != kernel.matrices.size()){
            throw std::invalid_argument( "depths of kernel and input does not match!\n" );
        }
        if(kernel.x % 2 == 0 || kernel.y % 2 == 0){
            throw std::invalid_argument( "kernel dimensions must be odd!\n" );
        }
        int newSizeX = (inputTensor.x - kernel.x + 2*padding) / stride + 1;
        int newSizeY = (inputTensor.y - kernel.y + 2*padding) / stride + 1;
        Matrix result = Matrix(newSizeX, newSizeY);
        
        for(int z = 0; z < inputTensor.matrices.size(); z++){
            const Matrix * inputMatrix = &inputTensor.matrices[z];
            const Matrix * kernelMatrix = &kernel.matrices[z];
            //convolution itself
            for(int y = 0; y < inputTensor.matrices[z].y - kernelMatrix->y + 1; y += stride){
                for(int x = 0; x < inputTensor.matrices[z].x - kernelMatrix->x + 1; x += stride){
                    //iterate over all in matrix
                    float sum = 0;
                    for(int y2 = 0; y2 < kernelMatrix->y; y2++){
                        for(int x2 = 0; x2 < kernelMatrix->x; x2++){
                            sum += inputMatrix->weights[x + x2][y + y2] * kernelMatrix->weights[x2][y2];
                        }
                    }
                    result.weights[x / stride][y / stride] = sum;
                }
            }
        }
        return result;//todo
    }