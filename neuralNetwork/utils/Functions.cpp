#include "Functions.h"
#include <stdexcept>

Matrix Functions::convolve(const Tensor &inputTensor, const Tensor &kernel, const int padding, const int stride){
        if(inputTensor.matrices.size() != kernel.matrices.size()){
            throw std::invalid_argument( "depths of kernel and input does not match!\n" );
        }
        if(kernel.matrices[0].weights[0].size() % 2 == 0 || kernel.matrices[0].weights.size() % 2 == 0){
            throw std::invalid_argument( "kernel dimensions must be odd!\n" );
        }
        int newSizeX = (inputTensor.matrices[0].weights[0].size() - kernel.matrices[0].weights[0].size() + 2*padding) / stride;
        int newSizeY = (inputTensor.matrices[0].weights.size() - kernel.matrices[0].weights.size() + 2*padding) / stride;
        Matrix result = Matrix(newSizeX, newSizeY);
        for(int z = 0; z < inputTensor.matrices.size(); z++){
            const Matrix * inputMatrix = &inputTensor.matrices[z];
            const Matrix * kernelMatrix = &kernel.matrices[z];
            //convolution itself
        }
        return Matrix(newSizeX, newSizeY);//todo
    }