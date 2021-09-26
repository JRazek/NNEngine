//
// Created by user on 25.09.2021.
//

#ifndef NEURALNETLIBRARY_VECTORN_H
#define NEURALNETLIBRARY_VECTORN_H

#include <zconf.h>
#include <vector>
#include "../interfaces/JSONEncodable.h"

namespace cn {
    template<u_int N, typename T>
    struct VectorN : public cn::JSONEncodable {
        T v[N];

        template<typename Y>
        VectorN<N, T>(const VectorN<N, Y> &other);
        VectorN<N, T>();
        VectorN<N, T>(const T d[N]);
        VectorN<N, T> operator*(const T &scalar);
        VectorN<N, T> operator/(const T &scalar);
        VectorN<N, T> operator+(const VectorN<N, T> &other);
        VectorN<N, T> operator-(const VectorN<N, T> &other);
        bool operator==(const VectorN<N, T> &other) const;
        bool operator!=(const VectorN<N, T> &other) const;

        T multiplyContent() const;
        virtual cn::JSON jsonEncode() const override;
    };

    template<u_int N, typename T>
    [[maybe_unused]]
    static void to_json(JSON &json, const VectorN<N, T> &v);

    template<u_int N, typename T>
    [[maybe_unused]]
    static void from_json(const JSON &j, VectorN<N, T> &value);

}

template<u_int N, typename T>
cn::VectorN<N, T>::VectorN(const T *d) {
    std::copy(d, d + N, v);
}

template<u_int N, typename T>
static void cn::to_json(JSON &json, const VectorN<N, T> &v){
    json = v.jsonEncode();
}

template<u_int N, typename T>
[[maybe_unused]]
static void cn::from_json(const JSON &j, VectorN<N, T> &value){
    for(u_int i = 0; i < N; i ++){
        value.v[i] = j.at(i);
    }
}

template<u_int N, typename T>
cn::VectorN<N, T>::VectorN() {
    std::fill(v, v + N, 0);
}

template<u_int N, typename T>
T cn::VectorN<N, T>::multiplyContent() const {
    T res = 1;
    for(u_int i = 0; i < N; i ++){
        res *= v[i];
    }
    return res;
}

template<u_int N, typename T>
cn::VectorN<N, T> cn::VectorN<N, T>::operator*(const T &scalar) {
    VectorN<N, T> copy(*this);
    for(u_int i = 0; i < N; i ++){
        copy.v[i] *= scalar;
    }
    return copy;
}

template<u_int N, typename T>
cn::VectorN<N, T> cn::VectorN<N, T>::operator/(const T &scalar) {
    VectorN<N, T> copy(*this);
    for(u_int i = 0; i < N; i ++){
        copy.v[i] /= scalar;
    }
    return copy;
}

template<u_int N, typename T>
cn::VectorN<N, T> cn::VectorN<N, T>::operator+(const VectorN<N, T> &other) {
    VectorN<N, T> copy(*this);
    for(u_int i = 0; i < N; i ++){
        copy.v[i] += other.v[i];
    }
    return copy;
}

template<u_int N, typename T>
cn::VectorN<N, T> cn::VectorN<N, T>::operator-(const VectorN<N, T> &other) {
    VectorN<N, T> copy(*this);
    for(u_int i = 0; i < N; i ++){
        copy.v[i] -= other.v[i];
    }
    return copy;
}

template<u_int N, typename T>
bool cn::VectorN<N, T>::operator==(const VectorN<N, T> &other) const {
    VectorN<N, T> copy(*this);
    for(u_int i = 0; i < N; i ++){
        if(copy.v[i] != other.v[i]) {
            return false;
        }
    }
    return true;
}

template<u_int N, typename T>
bool cn::VectorN<N, T>::operator!=(const VectorN<N, T> &other) const {
    return !(*this == other);
}

template<u_int N, typename T>
cn::JSON cn::VectorN<N, T>::jsonEncode() const {
    cn::JSON json;
    for(u_int i = 0; i < N; i ++){
        json.push_back(v[i]);
    }
    return json;
}

template<u_int N, typename T>
template<typename Y>
cn::VectorN<N, T>::VectorN(const VectorN<N, Y> &other) {
    for(u_int i = 0; i < N; i ++){
        v[i] = static_cast<T>(other.v[i]);
    }
}


#endif //NEURALNETLIBRARY_VECTORN_H
