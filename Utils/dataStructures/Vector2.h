//
// Created by user on 16.08.2021.
//

#ifndef NEURALNETLIBRARY_VECTOR2_H
#define NEURALNETLIBRARY_VECTOR2_H
#include <vector>

template<typename T>
struct Vector2 {
    T x, y;
    explicit Vector2<T>(const std::pair<T, T> &p);
    Vector2<T>();
    Vector2<T>(T _x, T _y);

    template<typename Y>
    Vector2<T> operator=(const Vector2<Y> &other);
};

template<typename T>
Vector2<T>::Vector2(const std::pair<T, T> &p): x(p.first), y(p.second) {}

template<typename T>
Vector2<T>::Vector2(T _x, T _y): x(_x), y(_y) {}

template<typename T>
Vector2<T>::Vector2(): x(0), y(0) {

}

template<typename T>
template<typename Y>
Vector2<T> Vector2<T>::operator=(const Vector2<Y> &other) {
    return Vector2<T>((T)other.x, (T)other.y);
}


#endif //NEURALNETLIBRARY_VECTOR2_H
