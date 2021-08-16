//
// Created by user on 15.08.2021.
//

#include "TMatrix.h"
TMatrix::TMatrix(float _a, float _b, float _c, float _d)
    :
    a(_a),
    b(_b),
    c(_c),
    d(_d){}

Vector2f TMatrix::getIHat() const {
    return {a, c};
}

Vector2f TMatrix::getJHat() const {
    return {b, d};
}
