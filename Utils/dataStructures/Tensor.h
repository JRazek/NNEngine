//
// Created by jrazek on 30.07.2021.
//

#include <algorithm>
#include "../Utils.h"
#ifndef NEURALNETLIBRARY_BITMAP_H
#define NEURALNETLIBRARY_BITMAP_H


namespace cn {
    template<typename T>
    class Tensor : public JSONEncodable{
    protected:

        /**
         * data should be stored in 0th format described in Utils.h file
         */
//        std::unique_ptr<T[]> dataP;
        T * dataP;
        int _w, _h, _d;
    public:
        Tensor(int w, int h, int d);
        Tensor(const Vector3<int> &s);
        Tensor(int w, int h, int d, const T *data, int inputType = 0);
        Tensor(const Vector3<int> &s, const T *data, int inputType = 0);
        Tensor(const Vector3<int> &s, T *&&data);
        Tensor(const Tensor &bitmap);
        Tensor(Tensor &&bitmap);
        Tensor(const JSON &json);
        Tensor();
        ~Tensor();
        T getCell(int col, int row, int depth) const;
        T getCell(const Vector3<int> &c) const;
        void setCell(int col, int row, int depth, T val);
        void setCell(const Vector3<int> &c, T b);
        void setData(const T * data, int inputType = 0);
        /**
         *
         * @param data rvalue data in 0 format already
         */
        void setData(T *&&data);
        void setLayer(int layerID, T *input);
        [[nodiscard]] int getDataIndex(int col, int row, int depth) const;
        [[nodiscard]] int getDataIndex(const Vector3<int> &v) const;
        Vector3<int> indexToVector(int index) const;

        /**
         *
         * @param other
         * @return true if they are copies. Function works in O(N) time where N is the size of tensor.
         */
        bool operator==(const Tensor<T> &other) const noexcept;
        bool operator!=(const Tensor<T> &other) const noexcept;

        Tensor<T> &operator=(const Tensor<T> &other);
        Tensor<T> &operator=(Tensor<T> &&other);
        Tensor<T> operator*(T scalar);
        Tensor<T> &operator*=(T scalar);
        Tensor<T> operator/(T scalar);
        Tensor<T> &operator/=(T scalar);
        bool belongs(const Vector3<int> &point) const;
        const T * dataConst() const;
        T * data();
        int w() const;
        int h() const;
        int d() const;
        Vector3<int> size() const;

        JSON jsonEncode() const override;
    };
}

/////DEFINITIONS/////
template<typename T>
cn::Tensor<T>::Tensor(int w, int h, int d): _w(w), _h(h), _d(d) {
    if(_w < 1 || _h < 1 || _d < 1)
        throw std::logic_error("invalid bitmap size!");
    dataP = new T [_w * _h * _d];
}

template<typename T>
cn::Tensor<T>::Tensor(const Tensor<T> &bitmap): Tensor(bitmap.w(), bitmap.h(), bitmap.d()) {
    std::copy(bitmap.dataP, bitmap.dataP + _w * _h * _d, dataP);
}

template<typename T>
T cn::Tensor<T>::getCell(int col, int row, int depth) const {
    return dataP[getDataIndex(col, row, depth)];
}

template<typename T>
void cn::Tensor<T>::setCell(int col, int row, int depth, T val) {
    dataP[getDataIndex(col, row, depth)] = val;
}

template<typename T>
const T *cn::Tensor<T>::dataConst() const{
    return dataP;
}

template<typename T>
T *cn::Tensor<T>::data() {
    return dataP;
}

template<typename T>
cn::Tensor<T>::~Tensor() {
    delete [] dataP;
}

template<typename T>
cn::Tensor<T>::Tensor(int _w, int _h, int _d, const T *data, int inputType): Tensor(_w, _h, _d) {
    setData(data, inputType);
}

template<typename T>
int cn::Tensor<T>::getDataIndex(int col, int row, int depth) const{
    if(col < 0 || col >= _w || row < 0 || row >= _h || depth < 0 || depth >= _d)
        throw std::logic_error("invalid read!");
    return depth * _w * _h + row * _w + col;
}

template<typename T>
cn::Tensor<T> &cn::Tensor<T>::operator=(const cn::Tensor<T> &other) {
    if(&other != this) {
        delete[] dataP;
        _w = other.w();
        _h = other.h();
        _d = other.d();
        dataP = new T[_w * _h * _d];
        std::copy(other.dataConst(), other.dataConst() + _w * _h * _d, dataP);
    }
    return *this;
}

template<typename T>
cn::Tensor<T> &cn::Tensor<T>::operator=(cn::Tensor<T> &&other){
    if(&other != this) {
        delete [] dataP;
        _w = other.w();
        _h = other.h();
        _d = other.d();
        dataP = other.dataP;
        other.dataP = nullptr;
    }
    return *this;
}

template<typename T>
cn::Tensor<T>::Tensor(cn::Tensor<T> &&bitmap):_w(bitmap.w()), _h(bitmap.h()), _d(bitmap.d()){
    if(&bitmap != this){
        dataP = bitmap.dataP;
        bitmap.dataP = nullptr;
    }
}

template<typename T>
void cn::Tensor<T>::setData(const T *data, int inputType) {
    cn::Utils::convert(data, dataP, _w, _h, _d, inputType, 0);
}

template<typename T>
void cn::Tensor<T>::setLayer(int layerID, T *input) {
    std::copy(input, input + _w * _h, dataP + _w * _h * layerID);
}


template<typename T>
int cn::Tensor<T>::w() const {
    return _w;
}

template<typename T>
int cn::Tensor<T>::h() const {
    return _h;
}

template<typename T>
int cn::Tensor<T>::d() const {
    return _d;
}

template<typename T>
T cn::Tensor<T>::getCell(const Vector3<int> &c) const {
    return getCell(c.x, c.y, c.z);
}

template<typename T>
int cn::Tensor<T>::getDataIndex(const Vector3<int> &v) const {
    return getDataIndex(v.x, v.y, v.z);
}

template<typename T>
cn::Vector3<int> cn::Tensor<T>::indexToVector(int index) const{
    return {index % _w, (index / _w) % _w, index / (_w * _h)};
}

template<typename T>
void cn::Tensor<T>::setCell(const Vector3<int> &c, T b) {
    setCell(c.x, c.y, c.z, b);
}

template<typename T>
cn::Vector3<int> cn::Tensor<T>::size() const {
    return Vector3<int>(_w, _h, _d);
}

template<typename T>
cn::Tensor<T>::Tensor(const Vector3<int> &s):Tensor(s.x, s.y, s.z) {}

template<typename T>
cn::Tensor<T>::Tensor(const Vector3<int> &s, const T *data, int inputType):Tensor(s.x, s.y, s.z, data, inputType) {}

template<typename T>
cn::Tensor<T>::Tensor():Tensor(0, 0, 0) {
    dataP = nullptr;
}

template<typename T>
bool cn::Tensor<T>::belongs(const Vector3<int> &p) const {
    return p.x >= 0 && p.x < _w && p.y >= 0 && p.y < _h && p.z >= 0 && p.z < _d;
}

template<typename T>
cn::JSON cn::Tensor<T>::jsonEncode() const{
    JSON json;
    json["size"] = size().jsonEncode();
    std::vector<T> dataVec;
    dataVec.insert(dataVec.end(), dataP, dataP + size().multiplyContent());
    json["data"] = dataVec;
    return json;
}

template<typename T>
cn::Tensor<T>::Tensor(const cn::JSON &json): Tensor(Vector3<int>(json.at("size"))) {
    std::vector<T> d = json.at("data");
    std::copy(d.begin(), d.end(), dataP);
}

template<typename T>
void cn::Tensor<T>::setData(T *&&data) {
    dataP = data;
    data = nullptr;
}

template<typename T>
cn::Tensor<T>::Tensor(const cn::Vector3<int> &s, T *&&data):Tensor(s) {
    dataP = data;
    data = nullptr;
}

template<typename T>
bool cn::Tensor<T>::operator==(const cn::Tensor<T> &other) const noexcept{
    if(size() == other.size()){
        for(u_int i = 0; i < size().multiplyContent(); i++){
            if(dataConst()[i] != other.dataConst()[i])
                return false;
        }
        return true;
    }
    return false;
}

template<typename T>
bool cn::Tensor<T>::operator!=(const cn::Tensor<T> &other) const noexcept{
    return !(*this == other);
}

template<typename T>
cn::Tensor<T> cn::Tensor<T>::operator*(T scalar) {
    Tensor<T> tensor = cn::Tensor<T>(*this);
    for(auto it = tensor.data(); it != tensor.data()+tensor.size().multiplyContent(); ++it){
        (*it) *= scalar;
    }
    return tensor;
}

template<typename T>
cn::Tensor<T> &cn::Tensor<T>::operator*=(T scalar) {
    for(auto it = data(); it != data()+size().multiplyContent(); ++it){
        (*it) *= scalar;
    }
    return *this;
}

template<typename T>
cn::Tensor<T> cn::Tensor<T>::operator/(T scalar) {
    Tensor<T> tensor = cn::Tensor<T>(*this);
    for(auto it = tensor.data(); it != tensor.data()+tensor.size().multiplyContent(); ++it){
        (*it) /= scalar;
    }
    return tensor;
}

template<typename T>
cn::Tensor<T> &cn::Tensor<T>::operator/=(T scalar) {
    for(auto it = data(); it != data()+size().multiplyContent(); ++it){
        (*it) /= scalar;
    }
    return *this;
}

#endif //NEURALNETLIBRARY_BITMAP_H
