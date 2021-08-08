//
// Created by jrazek on 05.08.2021.
//

#ifndef NEURALNETLIBRARY_UTILS_H
#define NEURALNETLIBRARY_UTILS_H


#include <bits/stdint-uintn.h>

namespace cn {
    using byte = uint8_t;

    template<typename T>
    class Bitmap;

    class Utils{
    public:
        /**
         *
         * @param input
         * @return normalized input. Each byte is now value equal to [ x / 255 ].
         */
        static Bitmap<float> normalize(const Bitmap<unsigned char> &input);

        /**
         *
         * @param input - data to convert
         * @param w - data width
         * @param h - data height
         * @param d - number of channels
         * @param inputType - format of input
         * @param outputType - format of input
         *
         * 0 - for standard (each row and channel is in ascending order)
         * {0, 1, 2},
         * {3, 4, 5},
         * {6, 7, 8}
         * indexing example.
         *
         * 1 - Ordering pixel on (x, y) pos in each channel is next to each other. Sth like RGB ordering
         */
        template<typename T>
        static void convert(const T *input, T *output, int w, int h, int d, int inputType, int outputType);


        /**
         *
         * @tparam T
         * @param input - bitmap to transform
         * @param destSizeX
         * @param destSizeY
         * @param method see below for types of methods
         * 0 - for average convolution kernel
         *
         * @return transformed bitmap
         */
        template<typename T>
        static Bitmap<T> transform(const Bitmap<T> &input, int destSizeX, int destSizeY, int method);
    };
};


template<typename T>
void cn::Utils::convert(const T *input, T *output, int w, int h, int d, int inputType, int outputType) {
    if(inputType == outputType){
        std::copy(input, input + w * h * d, output);
    }
    if(inputType == 1 && outputType == 0){
        for(int c = 0; c < d; c ++){
            for(int i = 0; i < w * h; i ++){
                output[w * h * c + i] = input[d * i + c];
            }
        }
    }else if(inputType == 0 && outputType == 1){
        for(int c = 0; c < d; c ++){
            for(int i = 0; i < w * h; i ++){
                output[d * i + c] = input[w * h * c + i];
            }
        }
    }
}

template<typename T>
cn::Bitmap<T> cn::Utils::transform(const cn::Bitmap<T> &input, int destSizeX, int destSizeY, int method) {
    if(method == 0){
        int kernelSizeX = input.w - destSizeX + 1;
        int kernelSizeY = input.h - destSizeY + 1;

    }
    return cn::Bitmap<T>();
}

#endif //NEURALNETLIBRARY_UTILS_H
