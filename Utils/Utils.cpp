#include "dataStructures/Bitmap.h"
#include "Utils.h"
#include <unordered_set>
#include <stack>
#include <cmath>
#include "dataStructures/KDTree.h"
#include "dataStructures/PrefixSum2D.h"
//
// Created by jrazek on 05.08.2021.
//

cn::Bitmap<double> cn::Utils::normalize(const Bitmap<byte> &input) {
    Bitmap<double> bitmap (input.w(), input.h(), input.d());
    for(int i = 0; i < input.w() * input.h() * input.d(); i ++){
        bitmap.data()[i] = ((double) input.dataConst()[i]) / 255.f;
    }
    return bitmap;
}



cn::Bitmap<double> cn::Utils::convolve(const Bitmap<double> &kernel, const Bitmap<double> &input, int paddingX, int paddingY, int strideX, int strideY) {
    if(!(kernel.w() % 2 && kernel.h() % 2 && kernel.d() == input.d())){
        throw std::invalid_argument("wrong dimensions of kernel!");
    }
    int sizeX = Utils::afterConvolutionSize(kernel.w(), input.w(), paddingX, strideX);
    int sizeY = Utils::afterConvolutionSize(kernel.h(), input.h(), paddingY, strideY);

    if(sizeX <= 0 || sizeY <= 0){
        throw std::invalid_argument("kernel bigger than input!");
    }

    cn::Bitmap<double> result (sizeX, sizeY, input.d());

    cn::Bitmap<double> paddedInput = addPadding(input, paddingX, paddingY);

    for(int y = 0; y < result.h(); y++){
        for(int x = 0; x < result.w(); x++){
            for(int c = 0; c < paddedInput.d(); c++){
                Vector2<int> kernelPos(x * strideX, y * strideY);
                double sum = 0;
                for(int ky = 0; ky < kernel.h(); ky++){
                    for(int kx = 0; kx < kernel.w(); kx++){
                        double v1 = kernel.getCell(kx, ky, c);
                        double v2 = paddedInput.getCell(kernelPos.x + kx, kernelPos.y + ky, c);
                        sum += v1 * v2;
                    }
                }
                if(x == 2 && y == 0 && c == 2)
                    printf("NORM sum: %f\n", sum);
                result.setCell(x, y, c, sum);
            }
        }
    }
    return result;
}

int cn::Utils::afterConvolutionSize(int kernelSize, int inputSize, int padding, int stride) {
    return (inputSize + 2*padding - kernelSize) / stride + 1;
}

double cn::Utils::distanceSquared(const std::pair<double, double> &p1, const std::pair<double, double> &p2) {
    return std::pow(p1.first - p2.first, 2) + std::pow(p1.second - p2.second, 2);
}

int cn::Utils::afterMaxPoolSize(int kernelSize, int inputSize) {
    return inputSize/kernelSize;
}

cn::Bitmap<double> cn::Utils::maxPool(const cn::Bitmap<double> &input, int kernelSizeX, int kernelSizeY) {
    Bitmap<double> output(afterMaxPoolSize(kernelSizeX, input.w()), afterMaxPoolSize(kernelSizeY, input.h()), input.d());
    for(int c = 0; c < input.d(); c++){
        for(int y = 0; y < input.h(); y += kernelSizeY){
            for(int x = 0; x < input.w(); x += kernelSizeX){
                double max = 0;
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
    cn::Bitmap<double> paddedInput (input.w() + paddingX * 2, input.h() + paddingY * 2, input.d());

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

cn::Bitmap<cn::byte> cn::Utils::average3Layers(const cn::Bitmap<cn::byte> &input) {
    if(input.d() != 3)
        throw std::logic_error("image must have 3 layers!");
    cn::Bitmap<cn::byte> result(input.w(), input.h(), 1);
    for(int y = 0; y < result.h(); y ++){
        for(int x = 0; x < result.w(); x++){
            int res = input.getCell(x, y, 0) + input.getCell(x, y, 1) + input.getCell(x, y, 2);
            result.setCell(x, y, 0, res / 3);
        }
    }
    return result;
}
