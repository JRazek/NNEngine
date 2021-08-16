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

Vector2f TMatrix::transform(const Vector2f &vector2F) const {
    return {a * vector2F.x + b * vector2F.y, c * vector2F.x + d * vector2F.y};
}


TMatrix TMatrix::operator*(float scalar) const {
    return TMatrix(a * scalar, b * scalar, c * scalar, d * scalar);
}

Vector2f TMatrix::operator*(Vector2f vec) const {
    return transform(vec);
}

