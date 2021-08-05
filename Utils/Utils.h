//
// Created by jrazek on 05.08.2021.
//

#ifndef NEURALNETLIBRARY_UTILS_H
#define NEURALNETLIBRARY_UTILS_H

#include "Bitmap.h"

namespace cn {

    /**
     *
     * @param input
     * @return normalized input. Each byte is now value equal to [ 1 / (256 - previousValue) ].
     */
    static Bitmap<float> * normalize(const Bitmap<unsigned char> &input);
};


#endif //NEURALNETLIBRARY_UTILS_H
