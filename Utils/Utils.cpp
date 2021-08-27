#include "Bitmap.h"
#include "Utils.h"
#include <unordered_set>
#include <stack>
#include <cmath>
#include "dataStructures/KDTree.h"
//
// Created by jrazek on 05.08.2021.
//

cn::Bitmap<float> cn::Utils::normalize(const Bitmap<byte> &input) {
    Bitmap<float> bitmap (input.w(), input.h(), input.d());
    for(int i = 0; i < input.w() * input.h() * input.d(); i ++){
        bitmap.data()[i] = ((float)input.data()[i]) / 255.f;
    }
    return bitmap;
}



cn::Bitmap<float> cn::Utils::convolve(const Bitmap<float> &kernel, const Bitmap<float> &input, int paddingX, int paddingY, int strideX, int strideY) {
    if(!(kernel.w() % 2 && kernel.h() % 2 && kernel.d() == input.d())){
        throw std::invalid_argument("wrong dimensions of kernel!");
    }
    int sizeX = Utils::afterConvolutionSize(kernel.w(), input.w(), paddingX, strideX);
    int sizeY = Utils::afterConvolutionSize(kernel.h(), input.h(), paddingY, strideY);

    if(sizeX <= 0 || sizeY <= 0){
        throw std::invalid_argument("kernel bigger than input!");
    }

    cn::Bitmap<float> output (sizeX, sizeY, input.d());

    cn::Bitmap<float> paddedInput = addPadding(input, paddingX, paddingY);

    for(int y = 0; y < paddedInput.h() - kernel.h(); y++){
        for(int x = 0; x < paddedInput.w() - kernel.w(); x++){
            for(int c = 0; c < paddedInput.d(); c++){
                Vector2<int> kernelPos(x, y);
                float sum = 0;
                for(int ky = 0; ky < kernel.h(); ky++){
                    for(int kx = 0; kx < kernel.w(); kx++){
                        sum += paddedInput.getCell(kernelPos.x + kx, kernelPos.y + ky, c) / kernel.getCell(kx, ky, c);
                    }
                }
                int outputX = kernelPos.x / strideX;
                int outputY = kernelPos.y / strideY;
                output.setCell(outputX, outputY, c, sum);
            }
        }
    }

    return output;
}

int cn::Utils::afterConvolutionSize(int kernelSize, int inputSize, int padding, int stride) {
    return (inputSize + 2*padding - kernelSize) / stride + 1;
}

float cn::Utils::distanceSquared(const std::pair<float, float> &p1, const std::pair<float, float> &p2) {
    return std::pow(p1.first - p2.first, 2) + std::pow(p1.second - p2.second, 2);
}

int cn::Utils::afterMaxPoolSize(int kernelSize, int inputSize) {
    return inputSize/kernelSize;
}

cn::Bitmap<float> cn::Utils::maxPool(const cn::Bitmap<float> &input, int kernelSizeX, int kernelSizeY) {
    Bitmap<float> output(afterMaxPoolSize(kernelSizeX, input.w()), afterMaxPoolSize(kernelSizeY, input.h()), input.d());
    for(int c = 0; c < input.d(); c++){
        for(int y = 0; y < input.h(); y += kernelSizeY){
            for(int x = 0; x < input.w(); x += kernelSizeX){
                float max = 0;
                for(int kY = 0; kY < kernelSizeY; kY++){
                    for(int kX = 0; kX < kernelSizeX; kX++){
                        max = std::max(input.getCell(x + kX, y + kY, c), max);
                    }
                }
                output.setCell(x / kernelSizeX, y / kernelSizeY, c, max);
            }
        }
    }
    return output;
}

template<typename T>
cn::Bitmap<T> cn::Utils::addPadding(const cn::Bitmap<T> &input, int paddingX, int paddingY) {
    cn::Bitmap<float> paddedInput (input.w() + paddingX * 2, input.h() + paddingY * 2, input.d());

    std::fill(paddedInput.data(), paddedInput.data() + paddedInput.w() * paddedInput.h() * paddedInput.d(), 0);

    for(int c = 0; c < input.d(); c ++){
        for(int y = 0; y < input.h(); y ++){
            for(int x = 0; x < input.w(); x ++){
                paddedInput.setCell(x + paddingX, y + paddingY, c, input.getCell(x, y, c));
            }
        }
    }
    return paddedInput;
}
