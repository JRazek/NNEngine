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
    struct Vector2 : public VectorN<2, T>{
        T &x, &y;

        Vector2();

        explicit Vector2<T>(const std::pair<T, T> &p);

        Vector2<T>(T _x, T _y);

        template<typename Y>
        Vector2<T> operator*(const TMatrix<Y> &tMatrix);

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
template<typename Y>
cn::Vector2<T> cn::Vector2<T>::operator*(const cn::TMatrix<Y> &tMatrix) {
    //todo static cast
    return Vector2<T>(x * tMatrix.a + y * tMatrix.b, x * tMatrix.c + y * tMatrix.d);
}

template<typename T>
cn::Vector2<T>::Vector2(const std::pair<T, T> &p): x(this->v[0]), y(this->v[1]) {}

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

template<typename T>
cn::Vector2<T>::Vector2(T _x, T _y): Vector2(){
    x = _x;
    y = _y;
}

template<typename T>
cn::JSON cn::Vector2<T>::jsonEncode() const{
    cn::JSON json;
    json["x"] = x;
    json["y"] = y;
    return json;
}

template<typename T>
cn::Vector2<T>::Vector2(): x(this->v[0]), y(this->v[1]) {}



#endif //NEURALNETLIBRARY_VECTOR2_H
