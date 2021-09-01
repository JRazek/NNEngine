//
// Created by user on 15.08.2021.
//

#ifndef NEURALNETLIBRARY_TMATRIX_H
#define NEURALNETLIBRARY_TMATRIX_H
#include <functional>
#include "Vector2.h"


/**
 * 2 x 2 2D transformation matrix
 */
template<typename T>
struct TMatrix {
    using func = std::function<double(double)> ;
    /**
     * matrix for space transformation
     * | a, b |
     * | c, d |
     */
    double a, b, c, d;
    TMatrix(double _a, double _b, double _c, double _d);
    [[nodiscard]] Vector2<T> getIHat() const;
    [[nodiscard]] Vector2<T> getJHat() const;

    template<typename Y>
    [[nodiscard]] Vector2<T> transform(const Vector2<Y> &vector2) const;

    template<typename Y>
    Vector2<T> operator *(const Vector2<Y> &vec) const;
    TMatrix operator *(T scalar) const;
    TMatrix operator *(TMatrix other) const;
};


template<typename T>
TMatrix<T>::TMatrix(double _a, double _b, double _c, double _d)
        :
        a(_a),
        b(_b),
        c(_c),
        d(_d){}

template<typename T>
Vector2<T> TMatrix<T>::getIHat() const {
    return {a, c};
}

template<typename T>
Vector2<T> TMatrix<T>::getJHat() const {
    return {b, d};
}

template<typename T>
template<typename Y>
Vector2<T> TMatrix<T>::transform(const Vector2<Y> &vector2) const {
    return {a * vector2.x + b * vector2.y, c * vector2.x + d * vector2.y};
}

template<typename T>
TMatrix<T> TMatrix<T>::operator*(T scalar) const {
    return TMatrix(a * scalar, b * scalar, c * scalar, d * scalar);
}
template<typename T>
template<typename Y>
Vector2<T> TMatrix<T>::operator*(const Vector2<Y> &vec) const {
    return transform(vec);
}



#endif //NEURALNETLIBRARY_TMATRIX_H
