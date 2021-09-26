//
// Created by user on 26.09.2021.
//

#ifndef NEURALNETLIBRARY_VECTOR4_H
#define NEURALNETLIBRARY_VECTOR4_H

#include <vector>
#include "VectorN.h"

namespace cn {
    template<typename T>
    struct TMatrix;

    template<typename T>
    struct Vector4 : public VectorN<4U, T> {
        T &x, &y, &z, &t;

        using VectorN<4U, T>::operator*;
        using VectorN<4U, T>::operator=;
        using VectorN<4U, T>::operator+;
        Vector4<T>();
        Vector4<T>(T _x, T _y, T _z, T _t);
        Vector4<T>(const Vector4<T> &other);
        Vector4<T>(const VectorN<4U, T> &other);

        Vector4<T> &operator=(const Vector4<T> &other);
        Vector4<T> &operator=(const VectorN<4U, T> &other) noexcept;
        cn::JSON jsonEncode() const override;
    };

    template<typename T>
    [[maybe_unused]]
    static void to_json(JSON &j, const Vector4<T> &value);

    template<typename T>
    [[maybe_unused]]
    static void from_json(const JSON &j, Vector4<T> &value);
}
template<typename T>
cn::Vector4<T>::Vector4(T _x, T _y, T _z, T _t): x(this->v[0]), y(this->v[1]), z(this->v[2]), t(this->v[3]) {
    x = _x;
    y = _y;
    z = _z;
    t = _t;
}

template<typename T>
[[maybe_unused]]
static void cn::to_json(JSON &json, const Vector4<T> &v){
    json = v.jsonEncode();
}

template<typename T>
[[maybe_unused]]
static void cn::from_json(const JSON &j, Vector4<T> &value){
    value = cn::Vector4<int>(j["x"], j["y"], j["z"], j["t"]);
}

template<typename T>
cn::Vector4<T>::Vector4(): Vector4(0, 0, 0, 0) {}

template<typename T>
cn::Vector4<T>::Vector4(const Vector4<T> &other): Vector4<T>(other.x, other.y, other.z, other.t) {}

template<typename T>
cn::JSON cn::Vector4<T>::jsonEncode() const{
    cn::JSON json;
    json["x"] = x;
    json["y"] = y;
    json["z"] = z;
    json["t"] = z;
    return json;
}

template<typename T>
cn::Vector4<T> &cn::Vector4<T>::operator=(const cn::Vector4<T> &other) {
    x = other.x;
    y = other.y;
    z = other.z;
    t = other.t;
    return *this;
}

template<typename T>
cn::Vector4<T> &cn::Vector4<T>::operator=(const cn::VectorN<4U, T> &other) noexcept {
    x = other.v[0];
    y = other.v[1];
    z = other.v[2];
    t = other.v[3];
    return *this;
}

template<typename T>
cn::Vector4<T>::Vector4(const cn::VectorN<4U, T> &other):
        VectorN<4U, T>(other),
        x(this->v[0]),
        y(this->v[1]),
        z(this->v[2]),
        t(this->v[3])
{}

#endif //NEURALNETLIBRARY_VECTOR4_H
