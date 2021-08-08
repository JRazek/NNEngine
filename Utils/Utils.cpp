#include "Bitmap.h"
#include "Utils.h"
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


template<typename T>
cn::Bitmap<T> cn::Utils::upsample(const cn::Bitmap<T> &input, int destSizeX, int destSizeY, int method) {
    //todo
}


template<typename T>
cn::Bitmap<T> cn::Utils::downsample(const cn::Bitmap<T> &input, int destSizeX, int destSizeY, int method) {
    if(method == 0){
        int kernelSizeX = input.w - destSizeX + 1;
        int kernelSizeY = input.h - destSizeY + 1;
        if(!(kernelSizeX % 2)){
            //extend the pic in X by one pixel
        }
        if(!(kernelSizeY % 2)){
            //extend the pic in Y by one pixel
        }
    }
    //todo
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
        }
    }

    return output;
}

int cn::Utils::afterConvolutionSize(int kernelSize, int inputSize, int padding, int step) {
    return (inputSize + 2*padding - kernelSize) / step + 1;
}

