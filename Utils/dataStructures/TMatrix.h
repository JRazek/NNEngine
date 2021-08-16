//
// Created by user on 15.08.2021.
//

#ifndef NEURALNETLIBRARY_TMATRIX_H
#define NEURALNETLIBRARY_TMATRIX_H
#include <functional>

struct TMatrix {
    using func = std::function<float(float)> ;
    /**
     * matrix of a functions for space transformation
     * | a, b |
     * | c, d |
     */
    std::function<float(float)> a, b, c, d;
    TMatrix(const func &_a, const func &_b, const func &_c, const func &_d);
};


#endif //NEURALNETLIBRARY_TMATRIX_H
