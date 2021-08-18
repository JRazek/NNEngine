//
// Created by user on 16.08.2021.
//

#ifndef NEURALNETLIBRARY_VECTOR2_H
#define NEURALNETLIBRARY_VECTOR2_H
#include <vector>

template<typename T>
struct TMatrix;

template<typename T>
struct Vector2 {
    T x, y;
    explicit Vector2<T>(const std::pair<T, T> &p);
    Vector2<T>();
    Vector2<T>(T _x, T _y);

    template<typename Y>
    Vector2<T>(const Vector2<Y> &other);

    Vector2<T> operator*(const T &scalar);
    template<typename Y>
    Vector2<T> operator*(const TMatrix<Y> &tMatrix);
    Vector2<T> operator+(const Vector2<T> &other);
    Vector2<T> operator-(const Vector2<T> &other);
};

template<typename T>
Vector2<T>::Vector2(const std::pair<T, T> &p): x(p.first), y(p.second) {}

template<typename T>
Vector2<T>::Vector2(T _x, T _y): x(_x), y(_y) {}

template<typename T>
Vector2<T>::Vector2(): x(0), y(0) {}


template<typename T>
Vector2<T> Vector2<T>::operator*(const T &scalar) {
    return Vector2<T>(scalar * x, scalar * y);
}

template<typename T>
Vector2<T> Vector2<T>::operator+(const Vector2<T> &other) {
    return Vector2<T>(x + other.x, y + other.y);
}

template<typename T>
Vector2<T> Vector2<T>::operator-(const Vector2<T> &other) {
    return Vector2<T>(x - other.x, y - other.y);
}

template<typename T>
template<typename Y>
Vector2<T> Vector2<T>::operator*(const TMatrix<Y> &tMatrix) {
    return Vector2<T>(x * tMatrix.a + y * tMatrix.b, x * tMatrix.c + y * tMatrix.d);
}

template<typename T>
template<typename Y>
Vector2<T>::Vector2(const Vector2<Y> &other): Vector2<T>((T)other.x, (T)other.y) {}


#endif //NEURALNETLIBRARY_VECTOR2_H
