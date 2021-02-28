#include "Functions.h"
#include <stdexcept>

Matrix Functions::convolve(const Tensor &inputTensor, const Tensor &kernel, const int padding, const int stride){
        if(inputTensor.matrices.size() != kernel.matrices.size()){
            throw std::invalid_argument( "depths of kernel and input does not match!\n" );
        }
        if(kernel.x % 2 == 0 || kernel.y % 2 == 0){
            throw std::invalid_argument( "kernel dimensions must be odd!\n" );
        }
        int newSizeX = (inputTensor.x - kernel.x + 2*padding) / stride;
        int newSizeY = (inputTensor.y - kernel.y + 2*padding) / stride;
        Matrix result = Matrix(newSizeX, newSizeY);
        int resultX = 0, resultY = 0;
        for(int z = 0; z < inputTensor.matrices.size(); z++){
            const Matrix * inputMatrix = &inputTensor.matrices[z];
            const Matrix * kernelMatrix = &kernel.matrices[z];
            //convolution itself
            for(int y = 0; y < inputTensor.matrices[z].y; y += stride){
                for(int x = 0; x < inputTensor.matrices[z].x; x += stride){
                    
                }
            }
        }
        return Matrix(newSizeX, newSizeY);//todo
    }