//
// Created by user on 15.08.2021.
//

#ifndef NEURALNETLIBRARY_TMATRIX_H
#define NEURALNETLIBRARY_TMATRIX_H
#include <functional>
#include "Vector2f.h"

struct TMatrix {
    using func = std::function<float(float)> ;
    /**
     * matrix for space transformation
     * | a, b |
     * | c, d |
     */
    float a, b, c, d;
    TMatrix(float _a, float _b, float _c, float _d);
    [[nodiscard]] Vector2f getIHat() const;
    [[nodiscard]] Vector2f getJHat() const;
};


#endif //NEURALNETLIBRARY_TMATRIX_H
