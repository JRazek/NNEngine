//
// Created by jrazek on 27.07.2021.
//

#include <stdexcept>
#include "ConvolutionLayer.h"
#include <iostream>

void cn::ConvolutionLayer::run(cn::Bitmap<float> &bitmap) {

}

cn::ConvolutionLayer::ConvolutionLayer(int id, cn::Network *network, int kernelSizeX, int kernelSizeY, int kernelSizeZ, int kernelsCount, int paddingX, int paddingY,
                                       int strideX,
                                       int strideY) :
                                       kernelSizeX(kernelSizeX),
                                       kernelSizeY(kernelSizeY),
                                       kernelSizeZ(kernelSizeZ),
                                       kernelsCount(kernelsCount),
                                       paddingX(paddingX),
                                       paddingY(paddingY),
                                       strideX(strideX),
                                       strideY(strideY),
                                       cn::Layer(id, network) {
    kernels.reserve(kernelsCount);
    for(int i = 0; i < kernelsCount; i ++){
        kernels.emplace_back(kernelSizeX, kernelSizeY, kernelSizeZ);
        std::fill(kernels.back().data(), kernels.back().data() + kernelSizeX * kernelSizeY * kernelSizeZ, 0);
    }
}

cn::Bitmap<float> cn::ConvolutionLayer::convolve(const Bitmap<float> &kernel, const Bitmap<float> &input,
                                                 int paddingX, int paddingY, int strideX, int strideY) {
    if(!(kernel.w % 2 && kernel.h % 2 && kernel.d == input.d)){
        throw std::invalid_argument("wrong dimensions of kernel!");
    }
    int sizeX = ConvolutionLayer::afterConvolutionSize(kernel.w, input.w, paddingX, strideX);
    int sizeY = ConvolutionLayer::afterConvolutionSize(kernel.h, input.h, paddingY, strideY);

    if(sizeX <= 0 || sizeY <= 0){
        throw std::invalid_argument("kernel bigger than input!");
    }
    cn::Bitmap<float> paddedInput (input.w + 2 * paddingX, input.h + 2 * paddingY, input.d);
    for(int c = 0; c < input.d; c ++){
        for(int y = 0; y < input.h; y += strideY){
            for(int x = 0; x < input.w; x += strideX){
                paddedInput.setCell(x + paddingX, y + paddingY, c, input.getCell(x, y, c));
            }
        }
    }
    cn::Bitmap<float> output (sizeX, sizeY, input.d);
    std::fill(output.data(), output.data() + output.w * output.h * output.d, 0);

    for(int originY = kernel.h / 2; originY < paddedInput.h - kernel.h / 2; originY += strideY){
        for(int originX = kernel.w / 2; originX < paddedInput.w - kernel.w / 2; originX += strideX){
            for(int channel = 0; channel < paddedInput.d; channel++){
                float sum = 0;
                for(int ky = 0; ky < kernel.h; ky++){
                    for(int kx = 0; kx < kernel.w; kx++){
                        sum += kernel.getCell(kx, ky, channel) * paddedInput.getCell(originX - kernel.w / 2 + kx, originY - kernel.h / 2 + ky, channel);
                    }
                }
                output.setCell(originX - kernel.w / 2, originY - kernel.h / 2, channel, sum);
            }
            std::cout<<"";
        }
    }

    return output;
}

int cn::ConvolutionLayer::afterConvolutionSize(int kernelSize, int inputSize, int padding, int step) {
    return (inputSize + 2*padding - kernelSize) / step + 1;
}

