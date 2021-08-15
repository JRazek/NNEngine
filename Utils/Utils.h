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


namespace cn {
    using byte = uint8_t;

    template<typename T>
    class Bitmap;

    class Utils{
    public:
        static inline int maxStackSize = 0;
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
         * 1 - bilinear interpolation
         * @return transformed bitmap
         */
        template<typename T>
        static Bitmap<T> upsample(const Bitmap<T> &input, int destSizeX, int destSizeY, int method);



        static int afterConvolutionSize(int kernelSize, int inputSize, int padding, int stride);
        static Bitmap<float> convolve(const Bitmap<float> &kernel, const Bitmap<float> &input, int paddingX = 0, int paddingY = 0, int strideX = 1, int strideY = 1);

        static float distanceSquared(const std::pair<float, float> &p1, const std::pair<float, float> &p2);


        /**
         *
         * @tparam T
         * @param bitmap - bitmap to search nn
         * @param point - point from where the bfs starts
         * @param filledArr - pointer to array of flags with filled pixels
         * @return  returns nearest filled neighbour
         */
        template<typename T>
        static std::pair<int, int>
        nearestNeighbour(const Bitmap <T> &bitmap, const std::pair<int, int> &point, int channel,
                         const bool *filledArr);

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
    cn::Bitmap<T> sampled = upsample<T>(input, destSizeX, destSizeY, 0);
    return downsample<T>(sampled, destSizeX, destSizeY, 0);
}



template<typename T>
cn::Bitmap<T> cn::Utils::downsample(const cn::Bitmap<T> &input, int destSizeX, int destSizeY, int method) {
    float factorX = (float)destSizeX / (float)input.w;
    float factorY = (float)destSizeY / (float)input.h;

    if(factorX == 1 && factorY == 1)
        return input;

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
    return input;
}



template<typename T>
cn::Bitmap<T> cn::Utils::upsample(const cn::Bitmap<T> &input, int destSizeX, int destSizeY, int method) {
    float factorX = (float)destSizeX / (float)input.w;
    float factorY = (float)destSizeY / (float)input.h;
    cn::Bitmap<T> result(destSizeX, destSizeY, input.d);

    //otherwise stack overflow occurs
    bool * filled = new bool [destSizeX * destSizeY * input.d];
    std::fill(filled, filled + destSizeX * destSizeY * input.d, false);


    for(int c = 0; c < input.d;  c++){
        for(int y = 0; y < input.h; y++){
            int corrY = y * factorY;
            if(corrY >= result.h)
                break;
            for(int x = 0; x < input.w; x++){
                int corrX = x * factorX;
                if(corrX >= result.w)
                    break;
                result.setCell(corrX, corrY, c, input.getCell(x, y, c));
                filled[result.getDataIndex(corrX, corrY, c)] = true;
            }
        }
    }

    if(method == 0){
        for(int c = 0; c < result.d;  c++){
            for(int y = 0; y < result.h; y++){
                for(int x = 0; x < result.w; x++){
                    if(!filled[result.getDataIndex(x, y, c)]){
                        auto nn = nearestNeighbour(result, {x, y}, c, filled);
                        result.setCell(x, y, c, result.getCell(nn.first, nn.second, c));
                        //result.setCell(x, y, c, 255);
                        //filled[result.getDataIndex(x, y, c)] = true;
                    }
                }
            }
        }
    }
    delete [] filled;
    //todo
    return result;
}

template<typename T>
std::pair<int, int> cn::Utils::nearestNeighbour(const Bitmap <T> &bitmap, const std::pair<int, int> &point, int channel, const bool *filledArr) {
    auto belongs = [](const cn::Bitmap<T> &bitmap, const std::pair<int, int> &point){
        return point.first >= 0 && point.first < bitmap.w && point.second >= 0 && point.second < bitmap.h;
    };
    if(!(belongs(bitmap, point))){
        throw std::out_of_range("This point does not belong to bitmap!");
    }
    auto hash = [](std::pair<int, int> const &pair){
        std::size_t h1 = std::hash<int>()(pair.first);
        std::size_t h2 = std::hash<int>()(pair.second);
        return h1 ^ h2;
    };

    //from where, point
    std::queue<std::pair<int, int>> queue;
    queue.push(point);

    std::unordered_set<std::pair<int, int>, decltype(hash)> visited(10, hash);

    while (!queue.empty()){
        if(queue.size() > maxStackSize){
            maxStackSize = queue.size();
        }
        auto p = queue.front();
        queue.pop();
        visited.insert(p);
        if(belongs(bitmap, p)){
            if(filledArr[bitmap.getDataIndex(p.first, p.second, channel)]){
                return {p.first, p.second};
            }
            std::vector<std::pair<int,int>> neighbors(4);
            neighbors[0] = {p.first - 1, p.second};
            neighbors[1] = {p.first + 1, p.second};
            neighbors[2] = {p.first, p.second - 1};
            neighbors[3] = {p.first, p.second + 1};

            for(auto n : neighbors) {
                if (visited.find(n) == visited.end()) {
                    queue.push(n);
                }
            }
        }
    }
    return std::pair<int, int>({-1, -1});
}

#endif //NEURALNETLIBRARY_UTILS_H
