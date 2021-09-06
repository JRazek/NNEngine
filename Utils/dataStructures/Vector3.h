//
// Created by user on 20.08.2021.
//

#ifndef NEURALNETLIBRARY_VECTOR3_H
#define NEURALNETLIBRARY_VECTOR3_H

#include <vector>
#include "../interfaces/JSONEncodable.h"

namespace cn {
    template<typename T>
    struct TMatrix;

    template<typename T>
    struct Vector3 : public cn::JSONEncodable {
        T x, y, z;

        Vector3<T>();

        Vector3<T>(T _x, T _y, T _z);

        template<typename Y>
        Vector3<T>(const Vector3<Y> &other);

        cn::JSON jsonEncode() override;

        Vector3<T> operator*(const T &scalar);

        Vector3<T> operator/(const T &scalar);

        Vector3<T> operator+(const Vector3<T> &other);

        Vector3<T> operator-(const Vector3<T> &other);

        bool operator==(const Vector3<T> &other) const;

        bool operator!=(const Vector3<T> &other) const;

        T multiplyContent() const;
    };
}
template<typename T>
cn::Vector3<T>::Vector3(T _x, T _y, T _z): x(_x), y(_y), z(_z) {}

template<typename T>
cn::Vector3<T>::Vector3(): x(0), y(0), z(0) {}


template<typename T>
cn::Vector3<T> cn::Vector3<T>::operator*(const T &scalar) {
    return Vector3<T>(scalar * x, scalar * y, scalar * z);
}

template<typename T>
cn::Vector3<T> cn::Vector3<T>::operator+(const Vector3<T> &other) {
    return Vector3<T>(x + other.x, y + other.y, z + other.z);
}

template<typename T>
cn::Vector3<T> cn::Vector3<T>::operator-(const Vector3<T> &other) {
    return Vector3<T>(x - other.x, y - other.y, z - other.z);
}

template<typename T>
template<typename Y>
cn::Vector3<T>::Vector3(const Vector3<Y> &other): Vector3<T>((T) other.x, (T) other.y, (T) other.z) {}

template<typename T>
bool cn::Vector3<T>::operator==(const Vector3<T> &other) const{
    return x == other.x && y == other.y && z == other.z;
}

template<typename T>
cn::Vector3<T> cn::Vector3<T>::operator/(const T &scalar) {
    return Vector3<T>(x / scalar, y / scalar, z / scalar);
}

template<typename T>
T cn::Vector3<T>::multiplyContent() const {
    return x * y * z;
}

template<typename T>
bool cn::Vector3<T>::operator!=(const Vector3<T> &other) const {
    return !(*this == other);
}

template<typename T>
cn::JSON cn::Vector3<T>::jsonEncode() {
    cn::JSON json;
    json["x"] = x;
    json["y"] = y;
    json["z"] = z;
    return json;
}

#endif //NEURALNETLIBRARY_VECTOR3_H
