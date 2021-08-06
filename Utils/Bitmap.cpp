//
// Created by jrazek on 30.07.2021.
//

#include <algorithm>
#include "Utils.h"
template<typename T>
cn::Bitmap<T>::Bitmap(int w, int h, int d): w(w), h (h), d(d) {
    this->dataP = new T [w * h * d];
}


template<typename T>
cn::Bitmap<T>::Bitmap(const Bitmap<T> &bitmap): Bitmap(bitmap.w, bitmap.w, bitmap.h) {
    std::copy(bitmap.dataP, bitmap.dataP + w * h * d, this->dataP);
}

template<typename T>
T cn::Bitmap<T>::getByte(int col, int row, int depth) {
    if(col >= this->w or col < 0 or row >= this->h or row < 0 or depth >= this->d or depth < 0){
        throw std::invalid_argument("byte does not belong to bitmap!");
    }
    return this->dataP[depth * w * h + row * w + col];
}

template<typename T>
void cn::Bitmap<T>::setBye(int col, int row, int depth, T b) {
    if(col >= this->w or col < 0 or row >= this->h or row < 0 or depth >= this->d or depth < 0){
        throw std::invalid_argument("wrong byte to set!");
    }
    this->dataP[depth * w * h + row * w + col] = b;
}

template<typename T>
T *cn::Bitmap<T>::data() const{
    return this->dataP;
}

template<typename T>
cn::Bitmap<T>::~Bitmap() {
    delete [] dataP;
}

template<typename T>
cn::Bitmap<T>::Bitmap(int w, int h, int d, const T * data, int options): Bitmap(w, h, d) {
   // cn::Utils::convert(data, w, h, d, options, 0);
   // delete [] test;
    byte byte;
}


