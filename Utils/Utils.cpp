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
    Bitmap<float> bitmap (input.w, input.h, input.d);
    for(int i = 0; i < input.w * input.h * input.d; i ++){
        bitmap.data()[i] = ((float)input.data()[i]) / 255.f;
    }
    return bitmap;
}



cn::Bitmap<float> cn::Utils::convolve(const Bitmap<float> &kernel, const Bitmap<float> &input, int paddingX, int paddingY, int strideX, int strideY) {
    if(!(kernel.w % 2 && kernel.h % 2 && kernel.d == input.d)){
        throw std::invalid_argument("wrong dimensions of kernel!");
    }
    int sizeX = Utils::afterConvolutionSize(kernel.w, input.w, paddingX, strideX);
    int sizeY = Utils::afterConvolutionSize(kernel.h, input.h, paddingY, strideY);

    if(sizeX <= 0 || sizeY <= 0){
        throw std::invalid_argument("kernel bigger than input!");
    }
    cn::Bitmap<float> paddedInput (input.w + paddingX * 2, input.h + paddingY * 2, input.d);

    std::fill(paddedInput.data(), paddedInput.data() + paddedInput.w * paddedInput.h * paddedInput.d, 0);

    for(int c = 0; c < input.d; c ++){
        for(int y = 0; y < input.h; y += strideY){
            for(int x = 0; x < input.w; x += strideX){
                paddedInput.setCell(x + paddingX, y + paddingY, c, input.getCell(x, y, c));
            }
        }
    }
    cn::Bitmap<float> output (sizeX, sizeY, input.d);

    for(int originY = kernel.h / 2; originY < paddedInput.h - kernel.h / 2; originY += strideY){
        for(int originX = kernel.w / 2; originX < paddedInput.w - kernel.w / 2; originX += strideX){
            int outputX = originX - kernel.w / 2;
            int outputY = originY - kernel.h / 2;
            for(int channel = 0; channel < paddedInput.d; channel++){
                float sum = 0;
                for(int ky = 0; ky < kernel.h; ky++){
                    for(int kx = 0; kx < kernel.w; kx++){
                        sum += kernel.getCell(kx, ky, channel) * paddedInput.getCell(originX - kernel.w / 2 + kx, originY - kernel.h / 2 + ky, channel);
                    }
                }
                output.setCell(outputX, outputY, channel, sum);
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

std::function<float(float)> cn::Utils::ReLU = std::function<float(float)>([](float n){
    return std::max(0.f, n);
});

std::function<float(float)> cn::Utils::Sigmoid = std::function<float(float)>([](float n){
   return (1.f/(1.f + std::pow(std::exp(1), -n)));
});

