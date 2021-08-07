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
         * @return normalized input. Each byte is now value equal to [ 1 / (256 - previousValue) ].
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
         * @returns pointer to an array of converted data.
         * @warning the array must be deallocated by the user.
         * 0 - for standard (each column and channel is in ascending order)
         * 1 - Ordering pixel on (x, y) pos in each channel is next to each other. Sth like RGB ordering
         */
        template<typename T>
        static T *convert(const T *input, int w, int h, int d, int inputType, int outputType);

    };
};


template<typename T>
T *cn::Utils::convert(const T *input, int w, int h, int d, int inputType, int outputType) {
    T *data = new T [w * h * d];
    if(inputType == 1 && outputType == 0){
        for(int c = 0; c < d; c ++){
            for(int i = 0; i < w * h; i ++){
                data[w * h * c + i] = input[ d * i + c];
            }
        }
    }
    return data;
}

#endif //NEURALNETLIBRARY_UTILS_H
