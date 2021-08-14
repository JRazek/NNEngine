//
// Created by jrazek on 30.07.2021.
//

#include <algorithm>
#include "Utils.h"
#ifndef NEURALNETLIBRARY_BITMAP_H
#define NEURALNETLIBRARY_BITMAP_H


namespace cn {
    template<typename T>
    class Bitmap {
    private:

        /**
         * data should be stored in 0th format described in Utils.h file
         */
        T * dataP;
    public:
        const int w, h, d;
        Bitmap(int w, int h, int d);
        Bitmap(int w, int h, int d, const T* data, int inputType = 0);
        Bitmap(const Bitmap &bitmap);
        ~Bitmap();
        T getCell(int col, int row, int depth) const;
        void setCell(int col, int row, int depth, T b);
        [[nodiscard]] int getDataIndex(int col, int row, int depth) const;

        T * data() const;
    };
}

/////DEFINITIONS/////
template<typename T>
cn::Bitmap<T>::Bitmap(int w, int h, int d): w(w), h (h), d(d) {
    this->dataP = new T [w * h * d];
}

template<typename T>
cn::Bitmap<T>::Bitmap(const Bitmap<T> &bitmap): Bitmap(bitmap.w, bitmap.h, bitmap.d) {
    std::copy(bitmap.dataP, bitmap.dataP + w * h * d, this->dataP);
}

template<typename T>
T cn::Bitmap<T>::getCell(int col, int row, int depth) const {
    return this->dataP[getDataIndex(col, row, depth)];
}

template<typename T>
void cn::Bitmap<T>::setCell(int col, int row, int depth, T b) {
    this->dataP[getDataIndex(col, row, depth)] = b;
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
cn::Bitmap<T>::Bitmap(int w, int h, int d, const T * data, int inputType): Bitmap(w, h, d) {
    cn::Utils::convert<T>(data, this->dataP, w, h, d, inputType, 0);
}

template<typename T>
int cn::Bitmap<T>::getDataIndex(int col, int row, int depth) const{
    if(col >= this->w or col < 0 or row >= this->h or row < 0 or depth >= this->d or depth < 0){
        throw std::invalid_argument("cell does not belong to bitmap!");
    }
    return depth * w * h + row * w + col;
}
//



#endif //NEURALNETLIBRARY_BITMAP_H
