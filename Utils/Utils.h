//
// Created by jrazek on 05.08.2021.
//

#ifndef NEURALNETLIBRARY_UTILS_H
#define NEURALNETLIBRARY_UTILS_H

#include "Bitmap.h"

namespace cn {
    using byte = unsigned char;

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
        static T *convert(T *input, int w, int h, int d, int inputType, int outputType);
    };
};


#endif //NEURALNETLIBRARY_UTILS_H
