//
// Created by user on 20.08.2021.
//

#ifndef NEURALNETLIBRARY_VECTOR3_H
#define NEURALNETLIBRARY_VECTOR3_H

#include <vector>
#include "VectorN.h"

namespace cn {
    template<typename T>
    struct TMatrix;

    template<typename T>
    struct Vector3 : public VectorN<3U, T> {
        T &x, &y, &z;

        using VectorN<3, T>::operator*;
        using VectorN<3, T>::operator=;
        using VectorN<3, T>::operator+;
        Vector3<T>();
        Vector3<T>(T _x, T _y, T _z);
        Vector3<T>(const Vector3<T> &other);
        Vector3<T>(const VectorN<3U, T> &other);

        Vector3<T> &operator=(const Vector3<T> &other);
        Vector3<T> &operator=(const VectorN<3U, T> &other) noexcept;
        cn::JSON jsonEncode() const override;
    };

    template<typename T>
    [[maybe_unused]]
    static void to_json(JSON &j, const Vector3<T> &value);

    template<typename T>
    [[maybe_unused]]
    static void from_json(const JSON &j, Vector3<T> &value);
}
template<typename T>
cn::Vector3<T>::Vector3(T _x, T _y, T _z): x(this->v[0]), y(this->v[1]), z(this->v[2]) {
    x = _x;
    y = _y;
    z = _z;
}

template<typename T>
[[maybe_unused]]
static void cn::to_json(JSON &json, const Vector3<T> &v){
    json = v.jsonEncode();
}

template<typename T>
[[maybe_unused]]
static void cn::from_json(const JSON &j, Vector3<T> &value){
    value = cn::Vector3<int>(j["x"], j["y"], j["z"]);
}

template<typename T>
cn::Vector3<T>::Vector3(): Vector3(0,0,0) {}

template<typename T>
cn::Vector3<T>::Vector3(const Vector3<T> &other): Vector3<T>(other.x, other.y, other.z) {}

template<typename T>
cn::JSON cn::Vector3<T>::jsonEncode() const{
    cn::JSON json;
    json["x"] = x;
    json["y"] = y;
    json["z"] = z;
    return json;
}

template<typename T>
cn::Vector3<T> &cn::Vector3<T>::operator=(const cn::Vector3<T> &other) {
    x = other.x;
    y = other.y;
    z = other.z;
    return *this;
}

template<typename T>
cn::Vector3<T> &cn::Vector3<T>::operator=(const cn::VectorN<3U, T> &other) noexcept {
    x = other.v[0];
    y = other.v[1];
    z = other.v[2];
    return *this;
}

template<typename T>
cn::Vector3<T>::Vector3(const cn::VectorN<3U, T> &other):
VectorN<3, T>(other),
x(this->v[0]),
y(this->v[1]),
z(this->v[2])
{}

#endif //NEURALNETLIBRARY_VECTOR3_H
