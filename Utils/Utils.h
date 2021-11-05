//
// Created by jrazek on 05.08.2021.
//

#ifndef NEURALNETLIBRARY_UTILS_H
#define NEURALNETLIBRARY_UTILS_H


#include <bits/stdint-uintn.h>
#include "dataStructures/PointData.h"
#include "dataStructures/Vector3.h"
#include <iostream>
#include <queue>
#include <unordered_set>
#include "dataStructures/TMatrix.h"
#include <cmath>

namespace cn {
    using byte = uint8_t;
    constexpr static double e = M_E;

    template<typename T>
    class Tensor;

    template<typename T>
    class PrefixSum2D;

    class Utils{
    public:


        /**
         *
         * @param input
         * @return normalized input. Each byte is now value equal to [ x / 255 ].
         */
        static Tensor<double> normalize(const Tensor<unsigned char> &input);

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
        static Tensor<T> downsample(const Tensor<T> &input, int destSizeX, int destSizeY, int method);


        /**
         *
         * @tparam T - template param
         * @param input - bitmap to resize
         * @param destSizeX - wanted size in X axis
         * @param destSizeY - wanted size in Y axis
         * @return resampled bitmap
         */
        template<typename T>
        static Tensor<T> resize(const Tensor<T> &input, int destSizeX, int destSizeY);


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
        static Tensor<T> upsample(const Tensor<T> &input, int destSizeX, int destSizeY, int method);

        /**
         *
         * @tparam T
         * @param input image to transform
         * @param tMatrix matrix for linear transformation of an image
         * @return transformed image
         */
        template<typename T>
        static Tensor<T> transform(const Tensor<T> &input, const TMatrix<double> &tMatrix);

        /**
         *
         * @tparam T
         * @param input image to rotate
         * @param rad radians to rotate anticlockwise
         * @return rotated bitmap
         */
        template<typename T>
        static Tensor<T> rotate(const Tensor<T> &input, double rad);

        template<typename T>
        static Tensor<T> addPadding(const Tensor<T> &input, int paddingX, int paddingY);


        static int afterConvolutionSize(int kernelSize, int inputSize, int padding, int stride);
        static Tensor<double> convolve(const Tensor<double> &kernel, const Tensor<double> &input, int paddingX = 0, int paddingY = 0, int strideX = 1, int strideY = 1);

        static Tensor<cn::byte> average3Layers(const Tensor<cn::byte> &input);

        static int afterMaxPoolSize(int kernelSize, int inputSize);
        static Tensor<double> maxPool(const Tensor<double> &input, int kernelSizeX, int kernelSizeY);

        static double distanceSquared(const std::pair<double, double> &p1, const std::pair<double, double> &p2);

        template<typename T>
        static Tensor<T> sumBitmapLayers(const Tensor <T> &input);

        template<typename T>
        static Tensor<T> elementWiseProduct(const Tensor<T> &v1, const Tensor<T> &v2);

        template<typename T>
        static Tensor<T> elementWiseSum(const Tensor<T> &v1, const Tensor<T> &v2);
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
cn::Tensor<T> cn::Utils::resize(const cn::Tensor<T> &input, int destSizeX, int destSizeY) {
    auto max = [](double x, double y){
        return x > y ? x : y;
    };

    auto min = [](double x, double y){
        return x < y ? x : y;
    };

    cn::Tensor<T> sampled = upsample<T>(input, max(input.w(), destSizeX), max(input.h(), destSizeY), 0);
    return downsample<T>(sampled, min(sampled.w(), destSizeX), min(sampled.h(), destSizeY), 0);
}


template<typename T>
cn::Tensor<T> cn::Utils::downsample(const cn::Tensor<T> &input, int destSizeX, int destSizeY, int method) {
    double factorX = ((double)destSizeX) / (double)input.w();
    double factorY = (double)destSizeY / (double)input.h();

    if(factorX == 1 && factorY == 1)
        return input;

    std::vector<int> avgCount (destSizeX * destSizeY * input.d(), 0);

    cn::Tensor<T> output(destSizeX, destSizeY, input.d());
    std::fill(output.data(), output.data() + output.w() * output.h() * output.d(), 0);

    if(method == 0){
        for(int c = 0; c < output.d(); c++){
            for(int y = 0; y < output.h(); y++){
                for(int x = 0; x < output.w(); x++){
                    avgCount[output.getDataIndex(x, y, c)] += 1;
                    int corrX = (int)((double)x / factorX);
                    int corrY = (int)((double)y / factorY);
                    output.setCell(x, y, c, output.getCell(x, y, c) + input.getCell(corrX, corrY, c));
                }
            }
        }
        for(int c = 0; c < output.d(); c++){
            for(int y = 0; y < output.h(); y++){
                for(int x = 0; x < output.w(); x++){
                    output.setCell(x, y, c, output.getCell(x, y, c) / avgCount[output.getDataIndex(x, y, c)]);
                }
            }
        }
    }
    return output;
}


template<typename T>
cn::Tensor<T> cn::Utils::upsample(const cn::Tensor<T> &input, int destSizeX, int destSizeY, int method) {
    double factorX = (double)destSizeX / (double)input.w();
    double factorY = (double)destSizeY / (double)input.h();

    if(factorX == 1 && factorY == 1)
        return input;

    cn::Tensor<T> result(destSizeX, destSizeY, input.d());

    if(method == 0){
        for(int c = 0; c < result.d();  c++){
            for(int y = 0; y < result.h(); y++){
                int corrY = (int)((double)y / factorY);
                for(int x = 0; x < result.w(); x++){
                    int corrX = (int)((double)x / factorX);
                    result.setCell(x, y, c, input.getCell(corrX, corrY, c));
                }
            }
        }
    }
    return result;
}

template<typename T>
cn::Tensor<T> cn::Utils::sumBitmapLayers(const cn::Tensor<T> &input){
    cn::Tensor<T> output(input.w(), input.h(), 1);
    std::fill(output.data(), output.data() + output.w() * output.h(), 0);
    for(int c = 0; c < input.d();  c++){
        for(int y = 0; y < input.h(); y++){
            for(int x = 0; x < input.w(); x++){
                output.setCell(x, y, 0, output.getCell(x, y, 0) + input.getCell(x, y, c));
            }
        }
    }
    return output;
}

template<typename T>
cn::Tensor<T> cn::Utils::transform(const cn::Tensor<T> &input, const TMatrix<double> &tMatrix) {
    int maxX = 0, minX = INT32_MAX, maxY = 0, minY = INT32_MAX;
    std::vector<Vector2<int>> edges(4);
    edges[0] = {0,0};
    edges[1] = {input.w() - 1, 0};
    edges[2] = {input.w() - 1, input.h() - 1};
    edges[3] = {0, input.h() - 1};
    for(auto &e : edges){
        e = tMatrix * e;
        int x = e.x, y = e.y;
        maxX = std::max(x, maxX);
        minX = std::min(x, minX);
        maxY = std::max(y, maxY);
        minY = std::min(y, minY);
    }
    int w = (int)(maxX - minX) + 1;
    int h = (int)(maxY - minY) + 1;
    cn::Tensor<T> result(w, h, input.d());
    std::fill(result.data(), result.data() + w * h * result.d(), 0);

    Vector2<int> shift = {minX, minY};
    shift = shift * -1;

    for(int y = 0; y < input.h(); y++){
        for(int x = 0; x < input.w(); x++){
            Vector2<int> v(x,y);
            v = v * tMatrix + shift;
            for(int c = 0; c < input.d(); c++){
                T cell = input.getCell(x, y, c);
                result.setCell(v.x, v.y, c, cell);
            }
        }
    }


    return result;
}

template<typename T>
cn::Tensor<T> cn::Utils::rotate(const cn::Tensor<T> &input, double rad) {
    double sin = std::sin(rad);
    double cos = std::cos(rad);
    TMatrix<double> rotationMatrix(cos, -sin, sin, cos);
    return transform<T>(input, rotationMatrix);
}

template<typename T>
cn::Tensor<T> cn::Utils::elementWiseProduct(const cn::Tensor<T> &v1, const cn::Tensor<T> &v2) {
    if(v1.size() != v2.size())
        throw std::logic_error("incorrect input sizes for element wise multiplication!");

    cn::Tensor<T> res(v1);

    for(int i = 0; i < res.size().multiplyContent(); i ++){
        res.data()[i] *= v2.dataConst()[i];
    }

    return res;
}
template<typename T>
cn::Tensor<T> cn::Utils::elementWiseSum(const cn::Tensor<T> &v1, const cn::Tensor<T> &v2) {
    if(v1.size() != v2.size())
        throw std::logic_error("incorrect input sizes for element wise multiplication!");

    cn::Tensor<T> res(v1);

    for(int i = 0; i < res.size().multiplyContent(); i ++){
        res.data()[i] += v2.dataConst()[i];
    }

    return res;
}

#endif //NEURALNETLIBRARY_UTILS_H
