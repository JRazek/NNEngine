//
// Created by user on 16.08.2021.
//

#ifndef NEURALNETLIBRARY_VECTOR2_H
#define NEURALNETLIBRARY_VECTOR2_H
#include <vector>
#include "../interfaces/JSONEncodable.h"
#include "VectorN.h"


namespace cn {
    template<typename T>
    struct TMatrix;

    template<typename T>
    struct Vector2 : public VectorN<2U, T> {
        T &x, &y;

        Vector2<T>(const std::pair<T, T> &p);

        using VectorN<2U, T>::operator*;
        using VectorN<2U, T>::operator=;
        using VectorN<2U, T>::operator+;
        Vector2<T>(const Vector2<T> &other);
        Vector2<T>(const VectorN<2U, T> &other);
        Vector2<T>();
        Vector2<T>(T _x, T _y);
        template<typename Y>
        Vector2<T> operator*(const TMatrix<Y> &tMatrix);

        Vector2<T> &operator=(const Vector2<T> &other) noexcept;
        Vector2<T> &operator=(const VectorN<2U, T> &other) noexcept;
        cn::JSON jsonEncode() const override;
    };

    template<typename T>
    [[maybe_unused]]
    static void to_json(JSON &j, const Vector2<T> &value);

    template<typename T>
    [[maybe_unused]]
    static void from_json(const JSON &j, Vector2<T> &value);
}
template<typename T>
cn::Vector2<T>::Vector2(const std::pair<T, T> &p): Vector2() {
    x = p.first;
    y = p.second;
}

template<typename T>
cn::Vector2<T>::Vector2(T _x, T _y): Vector2(){
    x = _x;
    y = _y;
}

template<typename T>
cn::Vector2<T>::Vector2(): x(this->v[0]), y(this->v[1])  {}


template<typename T>
template<typename Y>
cn::Vector2<T> cn::Vector2<T>::operator*(const cn::TMatrix<Y> &tMatrix) {
    T a = static_cast<T>(tMatrix.a);
    T b = static_cast<T>(tMatrix.b);
    T c = static_cast<T>(tMatrix.c);
    T d = static_cast<T>(tMatrix.d);
    return Vector2<T>(x * a + y * b, x * c + y * d);
}

template<typename T>
cn::Vector2<T>::Vector2(const Vector2<T> &other):Vector2<T>(other.x, other.y) {}


template<typename T>
cn::JSON cn::Vector2<T>::jsonEncode() const{
    cn::JSON json;
    json["x"] = x;
    json["y"] = y;
    return json;
}

template<typename T>
cn::Vector2<T> &cn::Vector2<T>::operator=(const cn::Vector2<T> &other) noexcept {
    x = other.x;
    y = other.y;
    return *this;
}

template<typename T>
cn::Vector2<T> &cn::Vector2<T>::operator=(const cn::VectorN<2U, T> &other) noexcept {
    x = other.v[0];
    y = other.v[1];
    return *this;
};

template<typename T>
cn::Vector2<T>::Vector2(const cn::VectorN<2U, T> &other):
        VectorN<2U, T>(other),
        x(this->v[0]),
        y(this->v[1])
{}

template<typename T>
[[maybe_unused]]
static void cn::to_json(JSON &json, const Vector2<T> &v){
    json = v.jsonEncode();
}

template<typename T>
[[maybe_unused]]
static void cn::from_json(const JSON &j, Vector2<T> &value){
    value = cn::Vector2<int>(j["x"], j["y"]);
}

#endif //NEURALNETLIBRARY_VECTOR2_H
