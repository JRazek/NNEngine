//
// Created by jrazek on 05.08.2021.
//

#ifndef NEURALNETLIBRARY_UTILS_H
#define NEURALNETLIBRARY_UTILS_H


#include <bits/stdint-uintn.h>
#include "dataStructures/PointData.h"
#include <iostream>
#include <queue>
#include <unordered_set>
#include "dataStructures/TMatrix.h"


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
         * @param input - bitmap to downsample
         * @param destSizeX
         * @param destSizeY
         * @param method see below for types of methods
         * 0 - for average convolution kernel
         *
         * @return transformed bitmap
         */
        template<typename T>
        static Bitmap<T> downsample(const Bitmap<T> &input, int destSizeX, int destSizeY, int method);


        /**
         *
         * @tparam T - template param
         * @param input - bitmap to resize
         * @param destSizeX - wanted size in X axis
         * @param destSizeY - wanted size in Y axis
         * @return resampled bitmap
         */
        template<typename T>
        static Bitmap<T> resize(const Bitmap<T> &input, int destSizeX, int destSizeY);


        /**
         *
         * @tparam T
         * @param input - bitmap to downsample
         * @param destSizeX
         * @param destSizeY
         * @param method see below for types of methods
         * 0 - nearest neighbour
         * 1 - bilinear interpolation [not supported yet]
         * @return transformed bitmap
         */
        template<typename T>
        static Bitmap<T> upsample(const Bitmap<T> &input, int destSizeX, int destSizeY, int method);

        /**
         *
         * @tparam T
         * @param input image to transform
         * @param tMatrix matrix for linear transformation of an image
         * @return transformed image
         */
        template<typename T>
        static Bitmap<T> transform(const Bitmap<T> &input, const TMatrix &tMatrix);


        static int afterConvolutionSize(int kernelSize, int inputSize, int padding, int stride);
        static Bitmap<float> convolve(const Bitmap<float> &kernel, const Bitmap<float> &input, int paddingX = 0, int paddingY = 0, int strideX = 1, int strideY = 1);

        static float distanceSquared(const std::pair<float, float> &p1, const std::pair<float, float> &p2);
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
cn::Bitmap<T> cn::Utils::resize(const cn::Bitmap<T> &input, int destSizeX, int destSizeY) {
    auto max = [](float x, float y){
        return x > y ? x : y;
    };

    auto min = [](float x, float y){
        return x < y ? x : y;
    };

    cn::Bitmap<T> sampled = upsample<T>(input, max(input.w, destSizeX), max(input.h, destSizeY), 0);
    return downsample<T>(sampled, min(input.w, destSizeX), min(input.h, destSizeY), 0);
}



template<typename T>
cn::Bitmap<T> cn::Utils::downsample(const cn::Bitmap<T> &input, int destSizeX, int destSizeY, int method) {
    float factorX = (float)destSizeX / (float)input.w;
    float factorY = (float)destSizeY / (float)input.h;

    if(factorX == 1 && factorY == 1)
        return input;

    std::vector<int> avgCount (destSizeX * destSizeY * input.d, 0);

    cn::Bitmap<T> result(destSizeX, destSizeY, input.d);

    if(method == 0){
        for(int c = 0; c < result.d;  c++){
            for(int y = 0; y < result.h; y++){
                for(int x = 0; x < result.w; x++){
                    avgCount[result.getDataIndex(x, y, c)] += 1;
                    int corrX = (int)((float)x / factorX);
                    int corrY = (int)((float)y / factorY);
                    result.setCell(x, y, c, result.getCell(x, y, c) + input.getCell(corrX, corrY, c));
                }
            }
        }
        for(int c = 0; c < result.d;  c++){
            for(int y = 0; y < result.h; y++){
                for(int x = 0; x < result.w; x++){
                    result.setCell(x, y, c, result.getCell(x, y, c) / avgCount[result.getDataIndex(x, y, c)]);
                }
            }
        }
    }

    for(auto n : avgCount){
        if(n != 1){
            std::cout<<"";
        }
    }

    return result;
}


template<typename T>
cn::Bitmap<T> cn::Utils::upsample(const cn::Bitmap<T> &input, int destSizeX, int destSizeY, int method) {
    float factorX = (float)destSizeX / (float)input.w;
    float factorY = (float)destSizeY / (float)input.h;

    if(factorX == 1 && factorY == 1)
        return input;

    cn::Bitmap<T> result(destSizeX, destSizeY, input.d);

    if(method == 0){
        for(int c = 0; c < result.d;  c++){
            for(int y = 0; y < result.h; y++){
                int corrY = (int)((float)y / factorY);
                for(int x = 0; x < result.w; x++){
                    int corrX = (int)((float)x / factorX);
                    result.setCell(x, y, c, input.getCell(corrX, corrY, c));
                }
            }
        }
    }
    return result;
}

template<typename T>
cn::Bitmap<T> cn::Utils::transform(const cn::Bitmap<T> &input, const TMatrix &tMatrix) {
    return cn::Bitmap<T>();
}

#endif //NEURALNETLIBRARY_UTILS_H
